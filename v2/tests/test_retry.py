import unittest
import asyncio
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TranslationPipeline, GeminiAPIError, InvalidFormatError, TruncatedResponseError
from google.genai.errors import APIError


class TestRetryAndTruncation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_settings = {
            'features': {
                'auto_glossary': False,
                'auto_summary': False,
                'clean_thinking_tags': False,
                'use_async_client': False,
            },
            'model': {
                'name': 'dummy-model',
                'temperature': 0.25,
                'top_p': 0.9,
            },
            'paths': {
                'checkpoint_dir': 'test_checkpoints',
                'data_dir': 'test_data',
                'output_dir': 'test_output',
            },
            'translation': {
                'chunk_size': 1000,
                'genre': 'default',
                'max_workers': 2,
                'target_language': 'vi',
            },
            'concurrency': {
                'use_global_semaphore': True,
                'max_concurrent_requests': 2
            },
            'retry': {
                'max_retries': 3,
                'initial_delay': 0.01,  # fast tests
                'exponential_base': 2.0,
                'max_delay': 0.1,
                'jitter': False
            }
        }

    def mock_load_yaml(self, path):
        if 'settings.yaml' in path:
            return self.mock_settings
        elif 'genres.yaml' in path:
            return {'genres': {'default': {'label': 'Default'}}, 'languages': {'vi': {}}}
        elif 'system_prompts.yaml' in path:
            return {
                'system_base': 'Bạn là dịch giả...',
                'context_section': 'Bối cảnh: {summary}',
                'glossary_section': 'Glossary: {terms}',
                'characters_section': 'Characters: {characters}',
                'source_rules_en': 'English rules',
            }
        return {}

    async def test_retry_on_429(self):
        with patch.object(TranslationPipeline, '_load_yaml', side_effect=self.mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client') as mock_client:
                    pipeline = TranslationPipeline(project='test_proj')
                    
                    call_count = 0
                    def mock_generate_content(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        if call_count <= 2:
                            raise APIError(code=429, response_json={}, response=None)
                        
                        mock_resp = MagicMock()
                        mock_resp.text = "<translation>Thành công</translation>"
                        mock_resp.candidates = []
                        return mock_resp
                    
                    pipeline.client.models.generate_content = mock_generate_content
                    
                    result = await pipeline._translate_chunk_with_retry(
                        chunk="text",
                        context_snippet="context",
                        glossary_str="",
                        characters_str="",
                        source_lang="en",
                        chunk_idx=0
                    )
                    
                    self.assertEqual(result, "Thành công")
                    self.assertEqual(call_count, 3) # 2 fails + 1 success

    async def test_retry_exhausted(self):
        with patch.object(TranslationPipeline, '_load_yaml', side_effect=self.mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client') as mock_client:
                    pipeline = TranslationPipeline(project='test_proj')
                    
                    def mock_generate_content(*args, **kwargs):
                        raise APIError(code=429, response_json={}, response=None)
                    
                    pipeline.client.models.generate_content = mock_generate_content
                    
                    with self.assertRaises(GeminiAPIError):
                        await pipeline._translate_chunk_with_retry(
                            chunk="text",
                            context_snippet="context",
                            glossary_str="",
                            characters_str="",
                            source_lang="en",
                            chunk_idx=0
                        )

    async def test_missing_xml_tag_retry(self):
        with patch.object(TranslationPipeline, '_load_yaml', side_effect=self.mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client') as mock_client:
                    pipeline = TranslationPipeline(project='test_proj')
                    
                    call_count = 0
                    def mock_generate_content(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        mock_resp = MagicMock()
                        mock_resp.candidates = []
                        if call_count == 1:
                            mock_resp.text = "Bản dịch thiếu tag"
                        else:
                            mock_resp.text = "<translation>Bản dịch có tag</translation>"
                        return mock_resp
                    
                    pipeline.client.models.generate_content = mock_generate_content
                    
                    result = await pipeline._translate_chunk_with_retry(
                        chunk="text",
                        context_snippet="context",
                        glossary_str="",
                        characters_str="",
                        source_lang="en",
                        chunk_idx=0
                    )
                    
                    self.assertEqual(result, "Bản dịch có tag")
                    self.assertEqual(call_count, 2)

    async def test_truncation_splits_chunk(self):
        with patch.object(TranslationPipeline, '_load_yaml', side_effect=self.mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client') as mock_client:
                    pipeline = TranslationPipeline(project='test_proj')
                    
                    call_count = 0
                    def mock_generate_content(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        mock_resp = MagicMock()
                        if call_count == 1:
                            mock_cand = MagicMock()
                            mock_cand.finish_reason = "MAX_TOKENS"
                            mock_resp.candidates = [mock_cand]
                            mock_resp.text = "Cắt xén..."
                        else:
                            mock_resp.candidates = []
                            mock_resp.text = f"<translation>Phần {call_count-1}</translation>"
                        return mock_resp
                    
                    pipeline.client.models.generate_content = mock_generate_content
                    
                    chunk_text = "Paragraph 1\n\nParagraph 2"
                    result = await pipeline._translate_chunk_with_retry(
                        chunk=chunk_text,
                        context_snippet="context",
                        glossary_str="",
                        characters_str="",
                        source_lang="en",
                        chunk_idx=0,
                        max_depth=1
                    )
                    
                    self.assertEqual(result, "Phần 1\n\nPhần 2")
                    self.assertEqual(call_count, 3)


if __name__ == '__main__':
    unittest.main()
