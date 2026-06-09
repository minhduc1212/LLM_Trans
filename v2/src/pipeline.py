"""
pipeline.py - Parallel translation engine using Google GenAI API (Gemma 4 31B IT).

Flow per file:
  1. Detect source language (auto, first 2000 chars)
  2. Pre-extract baseline glossary and characters from the first ~15,000 chars (1 API call)
  3. Split text into chunks
  4. Translate all chunks in parallel concurrently using asyncio.gather:
     - For context, each chunk N receives the last ~1000 characters of the original text of chunk N-1.
     - Each chunk uses the pre-extracted glossary and characters list.
  5. Assemble translated parts in the correct order.
  6. Save output and cleanup checkpoints.
"""

import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from google import genai
from google.genai.types import GenerateContentConfig
import yaml

from .chunker import chunk_text, get_chunk_stats
from .checkpoint_manager import CheckpointManager
from .glossary_manager import GlossaryManager
from .lang_detector import detect_source_language, get_source_lang_label

import dotenv

dotenv.load_dotenv()
class TranslationPipeline:
    def __init__(
        self,
        project: str,
        model: Optional[str] = None,
        genre: Optional[str] = None,
        target_lang: Optional[str] = None,
        source_lang: Optional[str] = None,   # override auto-detect
        output_dir: Optional[str] = None,
        chunk_size: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        # Load configs
        self.settings  = self._load_yaml('config/settings.yaml')
        self.genres_cfg = self._load_yaml('config/genres.yaml')
        self.prompts   = self._load_yaml('prompts/system_prompts.yaml')

        # Override with explicit args
        self.model       = model       or self.settings['model']['name']
        self.genre       = genre       or self.settings['translation']['genre']
        self.target_lang = target_lang or self.settings['translation']['target_language']
        self.chunk_size  = chunk_size  or self.settings['translation']['chunk_size']
        self.max_workers = max_workers or self.settings['translation']['max_workers']
        self.project     = project

        # source_lang: 'auto' means detect per-file; explicit value overrides
        self._source_lang_override = source_lang  # None or explicit code
        self.source_lang = source_lang or 'auto'  # shown in header

        self.output_dir = Path(output_dir or self.settings['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Managers
        genre_label = self.genres_cfg['genres'].get(self.genre, {}).get('label', self.genre)
        self.checkpoint_mgr = CheckpointManager(project, self.settings['paths']['checkpoint_dir'])
        self.glossary_mgr   = GlossaryManager(project, genre_label, self.settings['paths']['data_dir'])

        # Thread pool for blocking API calls
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # API config options
        self.model_opts = {
            'temperature': self.settings['model'].get('temperature', 0.25),
            'top_p':       self.settings['model'].get('top_p', 0.9),
        }

        # Initialize Google GenAI client (requires GEMINI_API_KEY env var)
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError(
                "❌ Environment variable GEMINI_API_KEY is not set.\n"
                "   Please set it using:\n"
                "   PowerShell: $env:GEMINI_API_KEY='your_api_key'\n"
                "   CMD: set GEMINI_API_KEY=your_api_key\n"
                "   Linux/macOS: export GEMINI_API_KEY='your_api_key'"
            )
        self.client = genai.Client()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Source language helpers
    # ------------------------------------------------------------------
    def _resolve_source_lang(self, text: str) -> str:
        """Return explicit override, or auto-detect from text."""
        if self._source_lang_override:
            return self._source_lang_override
        return detect_source_language(text)

    def _get_source_rules(self, source_lang: str) -> str:
        """Return the matching source_rules_XX block from prompts yaml."""
        key = f'source_rules_{source_lang}'
        return self.prompts.get(key, self.prompts.get('source_rules_en', '')).strip()

    @staticmethod
    def _safe_format(template: str, **kwargs) -> str:
        """
        Safe string substitution that won't choke on literal { } in prompt text
        (e.g. JSON examples inside the YAML prompts).
        Only replaces {key} tokens that are in kwargs; leaves everything else alone.
        """
        for key, value in kwargs.items():
            template = template.replace('txt' if key == 'text' else '{' + key + '}', str(value))
        return template

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _build_system_prompt(self, source_lang: str) -> str:
        base         = self.prompts['system_base'].strip()
        source_rules = self._get_source_rules(source_lang)

        genre_cfg   = self.genres_cfg['genres'].get(self.genre, {})
        genre_hint  = (f"Thể loại: {genre_cfg.get('label', self.genre)}. "
                       f"{genre_cfg.get('hint', '')}")

        lang_cfg         = self.genres_cfg['languages'].get(self.target_lang, {})
        lang_instruction = lang_cfg.get('instruction', f'Translate to {self.target_lang}.')

        return f"{base}\n\n{source_rules}\n\n{genre_hint}\n{lang_instruction}"

    def _build_user_message(
        self,
        chunk: str,
        summary: str,
        glossary_str: str,
        characters_str: str,
    ) -> str:
        parts = []

        if summary:
            tpl = self.prompts['context_section']
            # Manual format to avoid JSON template braces issue
            parts.append(tpl.replace('{summary}', summary).rstrip())

        if glossary_str:
            tpl = self.prompts['glossary_section']
            parts.append(tpl.replace('{terms}', glossary_str).rstrip())

        if characters_str:
            tpl = self.prompts['characters_section']
            parts.append(tpl.replace('{characters}', characters_str).rstrip())

        prefix = '\n\n'.join(parts)
        if prefix:
            return prefix + '\n\n[VĂN BẢN CẦN DỊCH]\n' + chunk
        return '[VĂN BẢN CẦN DỊCH]\n' + chunk

    # ------------------------------------------------------------------
    # LLM calls (blocking - run in executor)
    # ------------------------------------------------------------------
    def _call_genai(self, system: str, user: str, temperature: float = None) -> str:
        temp = temperature if temperature is not None else self.model_opts['temperature']
        top_p = self.model_opts['top_p']

        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=GenerateContentConfig(
                system_instruction=system,
                temperature=temp,
                top_p=top_p,
            )
        )
        raw = response.text or ""

        # Strip thinking tags (if any)
        if self.settings['features'].get('clean_thinking_tags', True):
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            raw = re.sub(r'<\|.*?\|>',         '', raw, flags=re.DOTALL)

        return raw.strip()

    def _translate_chunk_sync(
        self,
        chunk: str,
        summary: str,
        glossary_str: str,
        characters_str: str,
        source_lang: str,
    ) -> str:
        system = self._build_system_prompt(source_lang)
        user   = self._build_user_message(chunk, summary, glossary_str, characters_str)
        return self._call_genai(system, user)

    def _pre_extract_glossary_and_characters_sync(self, text: str, source_lang: str) -> dict:
        """Extract baseline glossary and characters from a text sample."""
        # Use first 15000 characters as a representative sample
        sample = text[:15000]
        lang_labels = {'en': 'English', 'zh': 'Tiếng Trung (Chinese)',
                       'ja': 'Tiếng Nhật (Japanese)', 'ko': 'Tiếng Hàn (Korean)'}
        user = self.prompts['pre_extract_prompt'].replace('{source_lang}', lang_labels.get(source_lang, source_lang)).replace('{text}', sample)
        try:
            raw = self._call_genai("", user, temperature=0.1)
            # Strip markdown code blocks if any
            clean_content = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', raw, flags=re.DOTALL)
            match = re.search(r'\{.*\}', clean_content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"      ⚠️  Pre-extraction failed: {e}")
        return {}

    def _parse_xml_output_only(self, response_text: str) -> str:
        """Parses only the <translation> tag from the response."""
        translation_match = re.search(r'<translation>(.*?)</translation>', response_text, re.DOTALL | re.IGNORECASE)
        if translation_match:
            return translation_match.group(1).strip()
        # Fallback: remove XML tags if any, otherwise return raw
        return re.sub(r'<.*?>', '', response_text, flags=re.DOTALL).strip()

    # ------------------------------------------------------------------
    # Core: translate one file
    # ------------------------------------------------------------------
    async def translate_file(self, input_path: Path, resume: bool = False) -> Path:
        text = input_path.read_text(encoding='utf-8')

        # ── Auto-detect source language ──────────────────────────────
        detected_lang = self._resolve_source_lang(text)
        lang_label    = get_source_lang_label(detected_lang)
        auto_marker   = '' if self._source_lang_override else ' (auto)'

        print(f"\n📖 [{self.genre.upper()}] {input_path.name}")
        print(f"   🌐 Nguồn: {lang_label}{auto_marker} → {self.target_lang.upper()}")

        chunks = chunk_text(text, self.chunk_size, source_lang=detected_lang)
        stats  = get_chunk_stats(chunks, source_lang=detected_lang)
        total  = stats['total_chunks']

        print(f"   📦 {total} chunks | avg {stats['avg_tokens']} tokens | "
              f"max {stats['max_tokens']} tokens")

        # ── Resume from checkpoint ───────────────────────────────────
        translated_chunks: dict[int, str] = {}

        if resume:
            cp = self.checkpoint_mgr.load(input_path.stem)
            if cp:
                if 'translated_chunks' in cp:
                    # new dictionary format (keys are serialized as strings in JSON)
                    translated_chunks = {int(k): v for k, v in cp['translated_chunks'].items()}
                    detected_lang = cp.get('source_lang', detected_lang)
                    if cp.get('glossary_terms'):
                        self.glossary_mgr.update_terms(cp['glossary_terms'])
                    if cp.get('characters'):
                        self.glossary_mgr.update_characters(cp['characters'])
                    print(f"   ▶️  Resuming progress: {len(translated_chunks)}/{total} chunks completed")
                elif 'translated_parts' in cp:
                    # backward compatibility with old list format
                    for idx, part in enumerate(cp['translated_parts']):
                        translated_chunks[idx] = part
                    detected_lang = cp.get('source_lang', detected_lang)
                    if cp.get('glossary_terms'):
                        self.glossary_mgr.update_terms(cp['glossary_terms'])
                    if cp.get('characters'):
                        self.glossary_mgr.update_characters(cp['characters'])
                    print(f"   ▶️  Resuming progress (list format): {len(translated_chunks)}/{total} chunks")

        loop = asyncio.get_event_loop()

        # ── Pre-extract glossary & characters ────────────────────────
        if (not resume or not self.glossary_mgr.char_count()) and len(text) > 0:
            print("   🔍 Pre-extracting baseline characters and glossary terms ... ", end='', flush=True)
            pre_data = await loop.run_in_executor(
                self.executor,
                self._pre_extract_glossary_and_characters_sync,
                text,
                detected_lang,
            )
            if pre_data:
                terms_added = self.glossary_mgr.update_terms(pre_data.get('terms', {}))
                chars_added = self.glossary_mgr.update_characters(pre_data.get('characters', {}))
                print(f"✅ (Found {chars_added} characters, {terms_added} terms)")
            else:
                print("⚠️ (No data found or extraction failed)")

        # ── Translate chunks in parallel ─────────────────────────────
        chunks_to_translate = [i for i in range(total) if i not in translated_chunks]

        if chunks_to_translate:
            # We use max_workers (or setting.yaml equivalent) for concurrent requests limit
            print(f"   ⚡ Translating {len(chunks_to_translate)} chunks concurrently (workers: {self.max_workers}) ...")
            
            semaphore = asyncio.Semaphore(self.max_workers)
            
            glossary_str   = self.glossary_mgr.format_terms_for_prompt()
            characters_str = self.glossary_mgr.format_characters_for_prompt()

            async def _translate_single_chunk(idx: int):
                async with semaphore:
                    t_start = time.time()
                    chunk = chunks[idx]

                    # Context: last 1000 characters of previous original text
                    if idx > 0:
                        prev_chunk = chunks[idx - 1]
                        context_snippet = prev_chunk[-1000:]
                    else:
                        context_snippet = "(Đây là phần đầu của tác phẩm, không có bối cảnh trước)"

                    raw_response = await loop.run_in_executor(
                        self.executor,
                        self._translate_chunk_sync,
                        chunk,
                        context_snippet,
                        glossary_str,
                        characters_str,
                        detected_lang,
                    )

                    translation = self._parse_xml_output_only(raw_response)
                    translated_chunks[idx] = translation

                    # Thread-safe save checkpoint
                    self.checkpoint_mgr.save(input_path.stem, {
                        'total_chunks': total,
                        'translated_chunks': {str(k): v for k, v in translated_chunks.items()},
                        'source_lang': detected_lang,
                        'glossary_terms': self.glossary_mgr.get_all(),
                        'characters': self.glossary_mgr.get_characters(),
                    })

                    elapsed = time.time() - t_start
                    print(f"      ✅ Chunk {idx+1:03d}/{total} completed in {elapsed:.1f}s")

            # Gather all translation tasks
            tasks = [_translate_single_chunk(idx) for idx in chunks_to_translate]
            await asyncio.gather(*tasks)

        # ── Assemble & save output ───────────────────────────────────
        final_parts = [translated_chunks[i] for i in range(total)]
        final_text  = '\n\n'.join(final_parts)
        lang_suffix = f"_{self.target_lang}"
        output_path = self.output_dir / f"{input_path.stem}{lang_suffix}{input_path.suffix}"
        output_path.write_text(final_text, encoding='utf-8')

        # Glossary report
        report_path = self.output_dir / f"{self.project}_glossary_report.md"
        report_path.write_text(self.glossary_mgr.export_report(), encoding='utf-8')

        # Clean checkpoint on success
        self.checkpoint_mgr.delete(input_path.stem)

        print(f"   ✨ Done → {output_path.name}")
        print(f"   📚 Glossary: {self.glossary_mgr.term_count()} terms | "
              f"👥 Characters: {self.glossary_mgr.char_count()} tracked")

        return output_path

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    async def run(self, input_path_str: str, resume: bool = False) -> None:
        input_path = Path(input_path_str)
        supported  = {'.txt', '.md'}

        if input_path.is_file():
            if input_path.suffix.lower() not in supported:
                print(f"⚠️  Unsupported file type: {input_path.suffix}")
                return
            await self.translate_file(input_path, resume=resume)

        elif input_path.is_dir():
            files = []
            for ext in ['.txt', '.md']:
                files.extend(sorted(input_path.glob(f'*{ext}')))

            if not files:
                print("⚠️  No .txt or .md files found in directory.")
                return

            print(f"\n📁 Found {len(files)} files | {self.max_workers} parallel workers")

            semaphore = asyncio.Semaphore(self.max_workers)

            async def _translate_guarded(f: Path):
                async with semaphore:
                    return await self.translate_file(f, resume=resume)

            results = await asyncio.gather(
                *[_translate_guarded(f) for f in files],
                return_exceptions=True,
            )

            for f, result in zip(files, results):
                if isinstance(result, Exception):
                    print(f"   ❌ FAILED: {f.name} → {result}")

        else:
            print(f"❌ Path not found: {input_path}")
            return

        print(f"\n🎉 All done! Output in: {self.output_dir}/")
        self.executor.shutdown(wait=False)