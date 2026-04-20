"""
pipeline.py - Core translation engine.

Flow per chunk:
  1. Detect source language (auto, first 2000 chars)
  2. Build system prompt (base + source rules + genre hint + target language)
  3. Build user message (context summary + glossary + characters + text)
  4. Call Ollama → get translation
  5. Clean output (strip thinking tags)
  6. Extract summary for next chunk
  7. Extract new glossary terms
  8. Save checkpoint
  9. Repeat

Source language rules:
  - EN: keep capitalized proper nouns; terms annotated in parens
  - ZH: names → Hán-Việt; terms → MUST translate (no raw Chinese chars)
  - JA: kanji names → Hán-Việt; katakana names → phonetic; terms → MUST translate
  - KO: names → Hán-Việt or phonetic; terms → MUST translate (no hangul)
"""

import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import ollama
import yaml

from .chunker import chunk_text, get_chunk_stats
from .checkpoint_manager import CheckpointManager
from .glossary_manager import GlossaryManager
from .lang_detector import detect_source_language, get_source_lang_label


class TranslationPipeline:
    def __init__(
        self,
        project: str,
        model: Optional[str] = None,
        genre: Optional[str] = None,
        target_lang: Optional[str] = None,
        source_lang: Optional[str] = None,   # NEW: override auto-detect
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

        # Thread pool for blocking Ollama calls
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Ollama model options
        self.ollama_opts = {
            'temperature':    self.settings['model']['temperature'],
            'num_ctx':        self.settings['model']['num_ctx'],
            'top_p':          self.settings['model']['top_p'],
            'repeat_penalty': self.settings['model']['repeat_penalty'],
        }

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
        (e.g. JSON examples like {"key": "value"} inside the YAML prompts).
        Only replaces {key} tokens that are in kwargs; leaves everything else alone.
        """
        for key, value in kwargs.items():
            template = template.replace('{' + key + '}', str(value))
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
            parts.append(self._safe_format(tpl, summary=summary).rstrip())

        if glossary_str:
            tpl = self.prompts['glossary_section']
            parts.append(self._safe_format(tpl, terms=glossary_str).rstrip())

        if characters_str:
            tpl = self.prompts['characters_section']
            parts.append(self._safe_format(tpl, characters=characters_str).rstrip())

        prefix = '\n\n'.join(parts)
        if prefix:
            return prefix + '\n\n[VĂN BẢN CẦN DỊCH]\n' + chunk
        return '[VĂN BẢN CẦN DỊCH]\n' + chunk

    # ------------------------------------------------------------------
    # LLM calls (blocking - run in executor)
    # ------------------------------------------------------------------
    def _call_ollama(self, system: str, user: str, temperature: float = None) -> str:
        opts = dict(self.ollama_opts)
        if temperature is not None:
            opts['temperature'] = temperature

        response = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user},
            ],
            options=opts,
        )
        raw = response['message']['content']

        # Strip thinking tags (DeepSeek-R1, QwQ, Gemma reasoning mode, etc.)
        if self.settings['features']['clean_thinking_tags']:
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
        return self._call_ollama(system, user)

    def _extract_summary_sync(self, translated_chunk: str) -> str:
        user = self._safe_format(self.prompts['summary_prompt'], text=translated_chunk[:2000])
        try:
            return self._call_ollama("", user, temperature=0.1)
        except Exception:
            return ""

    def _extract_glossary_sync(self, original: str, translated: str) -> dict:
        user = self._safe_format(
            self.prompts['glossary_extraction_prompt'],
            original=original[:800],
            translated=translated[:800],
        )
        try:
            raw   = self._call_ollama("", user, temperature=0.1)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {}

    def _extract_characters_sync(self, text: str, source_lang: str) -> dict:
        """
        Extract characters with source-language-aware naming rules.
        Passes source_lang so the prompt instructs correct romanisation/Hán-Việt.
        """
        lang_labels = {'en': 'English', 'zh': 'Tiếng Trung (Chinese)',
                       'ja': 'Tiếng Nhật (Japanese)', 'ko': 'Tiếng Hàn (Korean)'}
        user = self._safe_format(
            self.prompts['character_extraction_prompt'],
            text=text[:1500],
            source_lang=lang_labels.get(source_lang, source_lang),
        )
        try:
            raw   = self._call_ollama("", user, temperature=0.1)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # Core: translate one file sequentially
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
        start_idx        = 0
        translated_parts: list[str] = []
        summary          = ""

        if resume:
            cp = self.checkpoint_mgr.load(input_path.stem)
            if cp:
                start_idx        = cp['chunk_index'] + 1
                translated_parts = cp['translated_parts']
                summary          = cp.get('summary', '')
                detected_lang    = cp.get('source_lang', detected_lang)  # restore from cp
                if cp.get('glossary_terms'):
                    self.glossary_mgr.update_terms(cp['glossary_terms'])
                if cp.get('characters'):
                    self.glossary_mgr.update_characters(cp['characters'])
                print(f"   ▶️  Resuming from chunk {start_idx}/{total}")

        loop = asyncio.get_event_loop()

        # ── Extract characters from first chunk (if fresh start) ─────
        if start_idx == 0 and not self.glossary_mgr.char_count():
            chars = await loop.run_in_executor(
                self.executor,
                self._extract_characters_sync,
                chunks[0],
                detected_lang,
            )
            if chars:
                added = self.glossary_mgr.update_characters(chars)
                print(f"   👥 Found {added} characters")

        # ── Translate chunks ─────────────────────────────────────────
        for i in range(start_idx, total):
            chunk = chunks[i]
            t0    = time.time()
            print(f"   🔄 Chunk {i+1:03d}/{total} ... ", end='', flush=True)

            glossary_str   = self.glossary_mgr.format_terms_for_prompt()
            characters_str = self.glossary_mgr.format_characters_for_prompt()

            translated = await loop.run_in_executor(
                self.executor,
                self._translate_chunk_sync,
                chunk, summary, glossary_str, characters_str, detected_lang,
            )
            translated_parts.append(translated)

            elapsed = time.time() - t0
            print(f"✅ ({elapsed:.1f}s)", flush=True)

            # ── Post-chunk: summary + glossary IN PARALLEL ──────────
            do_summary  = self.settings['features']['auto_summary']
            do_glossary = self.settings['features']['auto_glossary']

            if do_summary and do_glossary:
                # Both enabled → run concurrently, saves ~40s per chunk
                results = await asyncio.gather(
                    loop.run_in_executor(self.executor, self._extract_summary_sync, translated),
                    loop.run_in_executor(self.executor, self._extract_glossary_sync, chunk, translated),
                    return_exceptions=True,
                )
                if not isinstance(results[0], Exception):
                    summary = results[0]
                if not isinstance(results[1], Exception) and results[1]:
                    added = self.glossary_mgr.update_terms(results[1])
                    if added:
                        print(f"      📚 +{added} new glossary terms")

            elif do_summary:
                summary = await loop.run_in_executor(
                    self.executor, self._extract_summary_sync, translated
                )
            elif do_glossary:
                new_terms = await loop.run_in_executor(
                    self.executor, self._extract_glossary_sync, chunk, translated
                )
                if new_terms:
                    added = self.glossary_mgr.update_terms(new_terms)
                    if added:
                        print(f"      📚 +{added} new glossary terms")

            # Checkpoint (includes source_lang for resume)
            self.checkpoint_mgr.save(input_path.stem, {
                'chunk_index':    i,
                'total_chunks':   total,
                'translated_parts': translated_parts,
                'summary':        summary,
                'source_lang':    detected_lang,
                'glossary_terms': self.glossary_mgr.get_all(),
                'characters':     self.glossary_mgr.get_characters(),
            })

        # ── Assemble & save output ───────────────────────────────────
        final_text  = '\n\n'.join(translated_parts)
        lang_suffix = f"_{self.target_lang}"
        output_path = self.output_dir / f"{input_path.stem}{lang_suffix}{input_path.suffix}"
        output_path.write_text(final_text, encoding='utf-8')

        # Glossary report
        report_path = self.output_dir / f"{self.project}_glossary_report.md"
        report_path.write_text(self.glossary_mgr.export_report(), encoding='utf-8')

        # Clean checkpoint on success
        self.checkpoint_mgr.delete(input_path.stem)

        print(f"   ✨ Done → {output_path.name}")
        print(f"   📚 Glossary: {self.glossary_mgr.term_count()} terms saved")

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