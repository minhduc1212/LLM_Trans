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
from google.genai.errors import APIError
import yaml

from .chunker import chunk_text, get_chunk_stats, estimate_tokens, detect_script, get_chars_per_token
from .checkpoint_manager import CheckpointManager
from .glossary_manager import GlossaryManager
from .lang_detector import detect_source_language, get_source_lang_label

import dotenv

dotenv.load_dotenv()

# Custom Exception Classes
class TranslationError(Exception):
    """Base exception for translation errors."""
    pass

class GeminiAPIError(TranslationError):
    """API error (e.g. 429, 5xx, or network timeout)."""
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

class TruncatedResponseError(TranslationError):
    """The response was truncated (finish_reason == 'MAX_TOKENS')."""
    pass

class InvalidFormatError(TranslationError):
    """The response did not contain the expected translation tags."""
    pass

class DuplicateTranslationError(TranslationError):
    """Raised when the translation is too similar to recent translations."""
    pass

class LazyTranslationError(TranslationError):
    """Raised when the translation is too similar to the source text."""
    pass


def calculate_tfidf_cosine_similarity(text1: str, text2: str) -> float:
    """Calculates TF-IDF Weighted Cosine Similarity between two texts in pure Python."""
    import re
    import math
    import collections
    
    words1 = re.findall(r'\w+', text1.lower())
    words2 = re.findall(r'\w+', text2.lower())
    
    if not words1 or not words2:
        return 0.0
        
    c1 = collections.Counter(words1)
    c2 = collections.Counter(words2)
    
    all_words = set(c1.keys()).union(set(c2.keys()))
    
    idf = {}
    for word in all_words:
        df = 0
        if word in c1:
            df += 1
        if word in c2:
            df += 1
        idf[word] = math.log(2.0 / df) + 1.0
        
    vec1 = {word: c1[word] * idf[word] for word in c1}
    vec2 = {word: c2[word] * idf[word] for word in c2}
    
    dot_product = sum(vec1[w] * vec2[w] for w in vec1 if w in vec2)
    norm_a = sum(v * v for v in vec1.values())
    norm_b = sum(v * v for v in vec2.values())
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))


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

        # Concurrency settings
        concurrency_cfg = self.settings.get('concurrency', {})
        self.use_global_semaphore = concurrency_cfg.get('use_global_semaphore', True)
        self.max_concurrent_requests = concurrency_cfg.get('max_concurrent_requests', self.max_workers)

        if max_workers is not None:
            self.max_concurrent_requests = max_workers
            self.max_workers = max_workers

        # Thread pool for blocking API calls (ensure it has at least max_concurrent_requests size)
        executor_size = max(self.max_concurrent_requests, self.max_workers)
        self.executor = ThreadPoolExecutor(max_workers=executor_size)

        self.global_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Retry settings
        retry_cfg = self.settings.get('retry', {})
        self.max_retries = retry_cfg.get('max_retries', 5)
        self.initial_delay = retry_cfg.get('initial_delay', 2.0)
        self.exponential_base = retry_cfg.get('exponential_base', 2.0)
        self.max_delay = retry_cfg.get('max_delay', 60.0)
        self.jitter = retry_cfg.get('jitter', True)

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
    def _build_system_prompt(self, source_lang: str, glossary_str: str = "", characters_str: str = "") -> str:
        base         = self.prompts['system_base'].strip()
        source_rules = self._get_source_rules(source_lang)

        genre_cfg   = self.genres_cfg['genres'].get(self.genre, {})
        genre_hint  = (f"Thể loại: {genre_cfg.get('label', self.genre)}. "
                       f"{genre_cfg.get('hint', '')}")

        lang_cfg         = self.genres_cfg['languages'].get(self.target_lang, {})
        lang_instruction = lang_cfg.get('instruction', f'Translate to {self.target_lang}.')

        system_prompt = f"{base}\n\n{source_rules}\n\n{genre_hint}\n{lang_instruction}"
        
        if glossary_str:
            system_prompt += f"\n\n[GLOSSARY - DỊCH ĐÚNG THEO BẢNG NÀY]\n{glossary_str}"
        if characters_str:
            system_prompt += f"\n\n[NHÂN VẬT & XƯNG HÔ - GIỮ NHẤT QUÁN]\n{characters_str}"
            
        return system_prompt

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

    async def _run_api_task(self, coro_func, *args, **kwargs):
        """Awaits an async API function, guarded by the global semaphore if enabled."""
        if self.use_global_semaphore:
            async with self.global_semaphore:
                return await coro_func(*args, **kwargs)
        else:
            return await coro_func(*args, **kwargs)

    async def _call_genai(self, system: str, user: str, temperature: float = None) -> str:
        temp = temperature if temperature is not None else self.model_opts['temperature']
        top_p = self.model_opts['top_p']
        
        use_async = self.settings['features'].get('use_async_client', True)

        try:
            if use_async:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=GenerateContentConfig(
                        system_instruction=system,
                        temperature=temp,
                        top_p=top_p,
                    )
                )
            else:
                # Wrap synchronous generate_content in run_in_executor
                response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.client.models.generate_content,
                    self.model,
                    user,
                    GenerateContentConfig(
                        system_instruction=system,
                        temperature=temp,
                        top_p=top_p,
                    )
                )
        except APIError as e:
            raise GeminiAPIError(f"Gemini API Error: {e}", status_code=getattr(e, 'code', None))
        except Exception as e:
            err_msg = str(e).lower()
            if "timeout" in err_msg or "time out" in err_msg or "deadline" in err_msg:
                raise GeminiAPIError(f"Timeout: {e}", status_code=408)
            raise GeminiAPIError(f"Connection/Unexpected Error: {e}")

        raw = response.text or ""

        # Strip thinking tags (if any)
        if self.settings['features'].get('clean_thinking_tags', True):
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            raw = re.sub(r'<\|.*?\|>',         '', raw, flags=re.DOTALL)

        # Check finish reason
        finish_reason = None
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                finish_reason = str(candidate.finish_reason).upper()

        if finish_reason in ("MAX_TOKENS", "LENGTH"):
            raise TruncatedResponseError("Response was truncated because it reached the token limit.")

        return raw.strip()

    def _build_multi_turn_contents(
        self,
        chunk: str,
        context_snippet: str,
        glossary_str: str,
        characters_str: str,
        source_lang: str,
        idx: int,
        chunks: list[str],
        translated_chunks: dict[int, str],
        cpt: float,
    ) -> list[dict]:
        system_prompt = self._build_system_prompt(source_lang, glossary_str, characters_str)
        system_tokens = estimate_tokens(system_prompt, cpt)

        inject_in_system = self.settings['features'].get('inject_glossary_in_system_prompt', True)
        if inject_in_system:
            current_user_msg = self._build_user_message(chunk, context_snippet, "", "")
        else:
            current_user_msg = self._build_user_message(chunk, context_snippet, glossary_str, characters_str)

        current_tokens = estimate_tokens(current_user_msg, cpt)

        budget = self.settings['translation'].get('history_token_budget', 4000)
        max_history_chunks = self.settings['translation'].get('history_chapters', 2)

        history_pairs = []
        for prev_idx in range(idx - 1, -1, -1):
            if len(history_pairs) >= max_history_chunks:
                break
            if prev_idx in translated_chunks:
                history_pairs.append((chunks[prev_idx], translated_chunks[prev_idx]))

        history_pairs.reverse()

        valid_pairs = []
        current_history_tokens = 0

        for src, trans in reversed(history_pairs):
            user_text = f"[VĂN BẢN CẦN DỊCH]\n{src}"
            model_text = f"<translation>\n{trans}\n</translation>"
            pair_tokens = estimate_tokens(user_text, cpt) + estimate_tokens(model_text, cpt)
            if current_history_tokens + pair_tokens <= budget:
                valid_pairs.append((user_text, model_text))
                current_history_tokens += pair_tokens
            else:
                break

        valid_pairs.reverse()

        contents = []
        for user_text, model_text in valid_pairs:
            contents.append({"role": "user", "parts": [{"text": user_text}]})
            contents.append({"role": "model", "parts": [{"text": model_text}]})

        contents.append({"role": "user", "parts": [{"text": current_user_msg}]})
        return contents

    async def _translate_chunk(
        self,
        chunk: str,
        summary: str,
        glossary_str: str,
        characters_str: str,
        source_lang: str,
        contents: list = None,
        retry_instruction: str = None,
        temperature: float = None,
    ) -> str:
        inject_in_system = self.settings['features'].get('inject_glossary_in_system_prompt', True)
        if inject_in_system:
            system = self._build_system_prompt(source_lang, glossary_str, characters_str)
        else:
            system = self._build_system_prompt(source_lang)

        if retry_instruction:
            system += f"\n\n[LƯU Ý QUAN TRỌNG - THỬ LẠI]\n{retry_instruction}"

        if contents:
            user = contents
            if retry_instruction:
                import copy
                user = copy.deepcopy(contents)
                user[-1]["parts"][0]["text"] += f"\n\n[LƯU Ý QUAN TRỌNG - THỬ LẠI]\n{retry_instruction}"
        else:
            if inject_in_system:
                user   = self._build_user_message(chunk, summary, "", "")
            else:
                user   = self._build_user_message(chunk, summary, glossary_str, characters_str)
        return await self._call_genai(system, user, temperature=temperature)

    async def _sleep_with_backoff(self, retry_count: int):
        delay = self.initial_delay * (self.exponential_base ** (retry_count - 1))
        delay = min(delay, self.max_delay)
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        await asyncio.sleep(delay)

    def _split_chunk_in_half(self, chunk: str, source_lang: str) -> list[str]:
        # Split by paragraph first
        paragraphs = chunk.split('\n\n')
        if len(paragraphs) >= 2:
            mid = len(paragraphs) // 2
            part1 = '\n\n'.join(paragraphs[:mid])
            part2 = '\n\n'.join(paragraphs[mid:])
            return [part1, part2]
        
        # If only one paragraph, split by sentence
        endings = re.compile(r'(?<=[.!?。！？…])\s*')
        sentences = [s for s in endings.split(chunk) if s.strip()]
        if len(sentences) >= 2:
            mid = len(sentences) // 2
            part1 = ' '.join(sentences[:mid])
            part2 = ' '.join(sentences[mid:])
            return [part1, part2]
            
        # Fallback: split by character count
        mid = len(chunk) // 2
        return [chunk[:mid], chunk[mid:]]

    async def _split_and_translate_chunk(
        self,
        chunk: str,
        context_snippet: str,
        glossary_str: str,
        characters_str: str,
        source_lang: str,
        chunk_idx: int,
        max_depth: int,
        contents: list = None,
        chunks: list[str] = None,
        translated_chunks: dict[int, str] = None,
    ) -> str:
        parts = self._split_chunk_in_half(chunk, source_lang)
        
        inject_in_system = self.settings['features'].get('inject_glossary_in_system_prompt', True)
        
        contents_1 = None
        if contents:
            import copy
            contents_1 = copy.deepcopy(contents)
            user_msg_1 = self._build_user_message(parts[0], context_snippet, "", "") if inject_in_system else self._build_user_message(parts[0], context_snippet, glossary_str, characters_str)
            contents_1[-1]["parts"][0]["text"] = user_msg_1

        # Translate part 1
        t1 = await self._translate_chunk_with_retry(
            parts[0], context_snippet, glossary_str, characters_str, source_lang, chunk_idx, max_depth=max_depth, contents=contents_1, chunks=chunks, translated_chunks=translated_chunks
        )
        
        # The context snippet for part 2 is the end of part 1's source text
        context_snippet_2 = parts[0][-1000:]
        
        contents_2 = None
        if contents:
            import copy
            contents_2 = copy.deepcopy(contents)
            user_msg_2 = self._build_user_message(parts[1], context_snippet_2, "", "") if inject_in_system else self._build_user_message(parts[1], context_snippet_2, glossary_str, characters_str)
            contents_2[-1]["parts"][0]["text"] = user_msg_2
        
        # Translate part 2
        t2 = await self._translate_chunk_with_retry(
            parts[1], context_snippet_2, glossary_str, characters_str, source_lang, chunk_idx, max_depth=max_depth, contents=contents_2, chunks=chunks, translated_chunks=translated_chunks
        )
        
        return t1 + '\n\n' + t2

    async def _translate_chunk_with_retry(
        self,
        chunk: str,
        context_snippet: str,
        glossary_str: str,
        characters_str: str,
        source_lang: str,
        chunk_idx: int,
        max_depth: int = 3,
        contents: list = None,
        chunks: list[str] = None,
        translated_chunks: dict[int, str] = None,
    ) -> str:
        retries = 0
        current_chunk = chunk
        retry_instruction = None
        base_temp = self.model_opts.get('temperature', 0.25)
        
        detect_dup = self.settings['features'].get('detect_duplicate_translation', True)
        dup_thresh = self.settings['translation'].get('duplicate_threshold', 0.85)
        lazy_thresh = self.settings['translation'].get('lazy_threshold', 0.75)
        lookback = self.settings['translation'].get('similarity_lookback', 2)

        while retries <= self.max_retries:
            try:
                temp = base_temp
                if retries > 0:
                    temp = min(1.0, base_temp + 0.15 * retries)

                raw_response = await self._run_api_task(
                    self._translate_chunk,
                    current_chunk,
                    context_snippet,
                    glossary_str,
                    characters_str,
                    source_lang,
                    contents,
                    retry_instruction,
                    temperature=temp,
                )
                
                # Format check
                has_start = re.search(r'<translation>', raw_response, re.IGNORECASE)
                has_end = re.search(r'</translation>', raw_response, re.IGNORECASE)
                if not (has_start and has_end):
                    raise InvalidFormatError("Response format is invalid (missing <translation> tags).")
                
                parsed_translation = self._parse_xml_output_only(raw_response)
                
                # Similarity checks
                if detect_dup and parsed_translation:
                    # 1. Lazy translation check (similarity between original chunk and translation)
                    if source_lang.lower() != self.target_lang.lower():
                        lazy_sim = calculate_tfidf_cosine_similarity(current_chunk, parsed_translation)
                        if lazy_sim > lazy_thresh:
                            raise LazyTranslationError(
                                f"Lazy translation detected: similarity {lazy_sim:.2f} exceeds threshold {lazy_thresh:.2f}"
                            )

                    # 2. Duplicate translation check (similarity between current translation and previous N translations)
                    if translated_chunks and chunk_idx is not None:
                        for prev_idx in range(chunk_idx - 1, max(-1, chunk_idx - lookback - 1), -1):
                            if prev_idx in translated_chunks:
                                prev_trans = translated_chunks[prev_idx]
                                dup_sim = calculate_tfidf_cosine_similarity(prev_trans, parsed_translation)
                                if dup_sim > dup_thresh:
                                    raise DuplicateTranslationError(
                                        f"Duplicate translation detected with chunk {prev_idx+1}: "
                                        f"similarity {dup_sim:.2f} exceeds threshold {dup_thresh:.2f}"
                                    )

                return parsed_translation
                
            except TruncatedResponseError as e:
                # Length truncation! Split and translate
                print(f"      ⚠️  Chunk {chunk_idx+1} truncated. Splitting into 2 smaller chunks...")
                if max_depth > 0:
                    return await self._split_and_translate_chunk(
                        current_chunk, context_snippet, glossary_str, characters_str, source_lang, chunk_idx, max_depth - 1, contents, chunks, translated_chunks
                    )
                else:
                    raise TranslationError("Max splitting depth reached, cannot translate chunk.")
                    
            except InvalidFormatError as e:
                retries += 1
                if retries > self.max_retries:
                    raise
                print(f"      ⚠️  Chunk {chunk_idx+1} format error: {e}. Retrying ({retries}/{self.max_retries})...")
                retry_instruction = "LƯU Ý: Phản hồi của bạn thiếu thẻ <translation>...</translation>. Vui lòng trả về bản dịch được bọc chính xác trong thẻ này."
                await self._sleep_with_backoff(retries)
                
            except LazyTranslationError as e:
                retries += 1
                if retries > self.max_retries:
                    raise
                print(f"      ⚠️  Chunk {chunk_idx+1} lazy translation check failed: {e}. Retrying ({retries}/{self.max_retries}) with temperature higher...")
                target_lang_label = get_source_lang_label(self.target_lang)
                retry_instruction = f"LƯU Ý: Bản dịch vừa rồi quá giống văn bản gốc (chưa được dịch). Hãy dịch hoàn chỉnh sang {target_lang_label}, không sao chép lại văn bản gốc."
                await self._sleep_with_backoff(retries)

            except DuplicateTranslationError as e:
                retries += 1
                if retries > self.max_retries:
                    raise
                print(f"      ⚠️  Chunk {chunk_idx+1} duplicate translation check failed: {e}. Retrying ({retries}/{self.max_retries}) with temperature higher...")
                retry_instruction = "LƯU Ý: Bản dịch vừa rồi bị lặp lại hoặc quá giống phân đoạn trước. Vui lòng dịch đúng nội dung mới của phân đoạn này, không lặp lại câu chữ cũ."
                await self._sleep_with_backoff(retries)

            except GeminiAPIError as e:
                is_retryable = e.status_code in (429, 408, 500, 502, 503, 504) or e.status_code is None
                if not is_retryable:
                    raise
                
                retries += 1
                if retries > self.max_retries:
                    raise
                
                print(f"      ⚠️  Chunk {chunk_idx+1} API error (status: {e.status_code}): {e}. Retrying ({retries}/{self.max_retries})...")
                await self._sleep_with_backoff(retries)

    async def _pre_extract_glossary_and_characters(self, text: str, source_lang: str) -> dict:
        """Extract baseline glossary and characters from a text sample."""
        # Use first 15000 characters as a representative sample
        sample = text[:15000]
        lang_labels = {'en': 'English', 'zh': 'Tiếng Trung (Chinese)',
                       'ja': 'Tiếng Nhật (Japanese)', 'ko': 'Tiếng Hàn (Korean)'}
        user = self.prompts['pre_extract_prompt'].replace('{source_lang}', lang_labels.get(source_lang, source_lang)).replace('{text}', sample)
        try:
            raw = await self._call_genai("", user, temperature=0.1)
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

    async def _write_text_file(self, path: Path, text: str):
        """Write text to file in a non-blocking way using the executor."""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: path.write_text(text, encoding='utf-8')
        )

    async def _save_checkpoint_async(self, stem: str, data: dict):
        """Saves checkpoint in executor."""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.checkpoint_mgr.save,
            stem,
            data
        )

    async def _delete_checkpoint_async(self, stem: str):
        """Deletes checkpoint in executor."""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.checkpoint_mgr.delete,
            stem
        )

    async def _update_glossary_terms_async(self, terms: dict) -> int:
        """Updates glossary terms in executor."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.glossary_mgr.update_terms,
            terms
        )

    async def _update_glossary_characters_async(self, chars: dict) -> int:
        """Updates glossary characters in executor."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.glossary_mgr.update_characters,
            chars
        )

    # ------------------------------------------------------------------
    # Core: translate one file
    # ------------------------------------------------------------------
    async def translate_file(self, input_path: Path, resume: bool = False) -> Path:
        loop = asyncio.get_event_loop()
        # Non-blocking file read
        text = await loop.run_in_executor(self.executor, input_path.read_text, 'utf-8')

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
            # Non-blocking checkpoint load
            cp = await loop.run_in_executor(self.executor, self.checkpoint_mgr.load, input_path.stem)
            if cp:
                if 'translated_chunks' in cp:
                    translated_chunks = {int(k): v for k, v in cp['translated_chunks'].items()}
                    detected_lang = cp.get('source_lang', detected_lang)
                    if cp.get('glossary_terms'):
                        await self._update_glossary_terms_async(cp['glossary_terms'])
                    if cp.get('characters'):
                        await self._update_glossary_characters_async(cp['characters'])
                    print(f"   ▶️  Resuming progress: {len(translated_chunks)}/{total} chunks completed")
                elif 'translated_parts' in cp:
                    for idx, part in enumerate(cp['translated_parts']):
                        translated_chunks[idx] = part
                    detected_lang = cp.get('source_lang', detected_lang)
                    if cp.get('glossary_terms'):
                        await self._update_glossary_terms_async(cp['glossary_terms'])
                    if cp.get('characters'):
                        await self._update_glossary_characters_async(cp['characters'])
                    print(f"   ▶️  Resuming progress (list format): {len(translated_chunks)}/{total} chunks")

        # ── Pre-extract glossary & characters ────────────────────────
        if (not resume or not self.glossary_mgr.char_count()) and len(text) > 0:
            print("   🔍 Pre-extracting baseline characters and glossary terms ... ", end='', flush=True)
            pre_data = await self._run_api_task(
                self._pre_extract_glossary_and_characters,
                text,
                detected_lang,
            )
            if pre_data:
                terms_added = await self._update_glossary_terms_async(pre_data.get('terms', {}))
                chars_added = await self._update_glossary_characters_async(pre_data.get('characters', {}))
                print(f"✅ (Found {chars_added} characters, {terms_added} terms)")
            else:
                print("⚠️ (No data found or extraction failed)")

        # ── Translate chunks in parallel ─────────────────────────────
        chunks_to_translate = [i for i in range(total) if i not in translated_chunks]
        failed_chunks = {}

        if chunks_to_translate:
            use_rolling_history = self.settings['features'].get('rolling_history', True)
            cpt = stats.get('chars_per_token', 3.5)
            
            if use_rolling_history:
                limit_desc = "sequential with rolling history"
            else:
                limit_desc = f"global max: {self.max_concurrent_requests}" if self.use_global_semaphore else f"workers: {self.max_workers}"
                
            print(f"   ⚡ Translating {len(chunks_to_translate)} chunks ({limit_desc}) ...")
            
            local_semaphore = asyncio.Semaphore(self.max_workers) if (not self.use_global_semaphore and not use_rolling_history) else None
            filter_relevant = self.settings['features'].get('relevance_filtering', True)
            
            async def _translate_single_chunk(idx: int):
                async def _do_translation():
                    t_start = time.time()
                    chunk = chunks[idx]

                    # Context: last 1000 characters of previous original text
                    if idx > 0:
                        prev_chunk = chunks[idx - 1]
                        context_snippet = prev_chunk[-1000:]
                    else:
                        context_snippet = "(Đây là phần đầu của tác phẩm, không có bối cảnh trước)"

                    # Dynamic glossary formatting per chunk
                    g_str = self.glossary_mgr.format_terms_for_prompt(chunk if filter_relevant else None)
                    c_str = self.glossary_mgr.format_characters_for_prompt(chunk if filter_relevant else None)

                    # Build multi-turn contents for rolling history
                    contents = None
                    if use_rolling_history:
                        contents = self._build_multi_turn_contents(
                            chunk,
                            context_snippet,
                            g_str,
                            c_str,
                            detected_lang,
                            idx,
                            chunks,
                            translated_chunks,
                            cpt,
                        )

                    try:
                        translation = await self._translate_chunk_with_retry(
                            chunk,
                            context_snippet,
                            g_str,
                            c_str,
                            detected_lang,
                            idx,
                            contents=contents,
                            chunks=chunks,
                            translated_chunks=translated_chunks,
                        )
                        translated_chunks[idx] = translation

                        # Thread-safe save checkpoint (async)
                        await self._save_checkpoint_async(input_path.stem, {
                            'total_chunks': total,
                            'translated_chunks': {str(k): v for k, v in translated_chunks.items()},
                            'source_lang': detected_lang,
                            'glossary_terms': self.glossary_mgr.get_all(),
                            'characters': self.glossary_mgr.get_characters(),
                        })

                        elapsed = time.time() - t_start
                        print(f"      ✅ Chunk {idx+1:03d}/{total} completed in {elapsed:.1f}s")
                    except Exception as e:
                        print(f"      ❌ Chunk {idx+1:03d}/{total} failed after all retries: {e}")
                        failed_chunks[idx] = str(e)
                        translated_chunks[idx] = f"[LỖI DỊCH CHUNK {idx+1}: {e}]"

                if local_semaphore:
                    async with local_semaphore:
                        await _do_translation()
                else:
                    await _do_translation()

            # Execute tasks
            if use_rolling_history:
                for idx in chunks_to_translate:
                    await _translate_single_chunk(idx)
            else:
                tasks = [_translate_single_chunk(idx) for idx in chunks_to_translate]
                await asyncio.gather(*tasks)

        # ── Assemble & save output ───────────────────────────────────
        final_parts = [translated_chunks[i] for i in range(total)]
        final_text  = '\n\n'.join(final_parts)
        lang_suffix = f"_{self.target_lang}"
        output_path = self.output_dir / f"{input_path.stem}{lang_suffix}{input_path.suffix}"
        await self._write_text_file(output_path, final_text)

        # Glossary report
        report_path = self.output_dir / f"{self.project}_glossary_report.md"
        await self._write_text_file(report_path, self.glossary_mgr.export_report())

        # Clean checkpoint on success (only if no chunks failed)
        if failed_chunks:
            print(f"\n   ⚠️  WARNING: File completed with {len(failed_chunks)} failed chunks:")
            for k, err in failed_chunks.items():
                print(f"      - Chunk {k+1}: {err}")
        else:
            await self._delete_checkpoint_async(input_path.stem)

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

            if self.use_global_semaphore:
                print(f"\n📁 Found {len(files)} files | global concurrency limit: {self.max_concurrent_requests}")
                local_semaphore = None
            else:
                print(f"\n📁 Found {len(files)} files | {self.max_workers} parallel workers")
                local_semaphore = asyncio.Semaphore(self.max_workers)

            async def _translate_guarded(f: Path):
                if local_semaphore:
                    async with local_semaphore:
                        return await self.translate_file(f, resume=resume)
                else:
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