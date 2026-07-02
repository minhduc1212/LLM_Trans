import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import asyncio

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TranslationPipeline


class TestSimilarityChecks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_settings = {
            'features': {
                'auto_glossary': False,
                'auto_summary': False,
                'clean_thinking_tags': False,
                'use_async_client': True,
                'detect_duplicate_translation': True,
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
                'duplicate_threshold': 0.85,
                'lazy_threshold': 0.75,
                'similarity_lookback': 2,
            },
            'concurrency': {
                'use_global_semaphore': True,
                'max_concurrent_requests': 2
            },
            'retry': {
                'max_retries': 3,
                'initial_delay': 0.01,
                'exponential_base': 2.0,
                'max_delay': 0.1,
                'jitter': False,
            }
        }

    def mock_load_yaml(self, path):
        if 'settings.yaml' in path:
            return self.mock_settings
        elif 'genres.yaml' in path:
            return {'genres': {'default': {'label': 'Default'}}, 'languages': {'vi': {'instruction': 'Translate to Vietnamese'}}}
        elif 'system_prompts.yaml' in path:
            return {
                'system_base': 'Base',
                'context_section': '{summary}',
                'glossary_section': '{terms}',
                'characters_section': '{characters}',
                'source_rules_en': 'English rules',
            }
        return {}

    async def test_lazy_translation_retry(self):
        with patch.object(TranslationPipeline, '_load_yaml', side_effect=self.mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client'):
                    pipeline = TranslationPipeline(project='test_proj')
                    
                    calls = []
                    async def mock_translate_chunk(chunk, summary, glossary, chars, lang, contents=None, retry_instruction=None, temperature=None):
                        calls.append((retry_instruction, temperature))
                        if len(calls) == 1:
                            return "<translation>The spaceship landed safely on the new planet.</translation>"
                        return "<translation>Tàu vũ trụ đã hạ cánh an toàn trên hành tinh mới.</translation>"
                        
                    pipeline._translate_chunk = mock_translate_chunk
                    
                    chunk = "The spaceship landed safely on the new planet."
                    result = await pipeline._translate_chunk_with_retry(
                        chunk=chunk,
                        context_snippet="",
                        glossary_str="",
                        characters_str="",
                        source_lang="en",
                        chunk_idx=0
                    )
                    
                    self.assertEqual(result, "Tàu vũ trụ đã hạ cánh an toàn trên hành tinh mới.")
                    self.assertEqual(len(calls), 2)
                    self.assertIsNone(calls[0][0])
                    self.assertEqual(calls[0][1], 0.25)
                    self.assertIsNotNone(calls[1][0])
                    self.assertIn("văn bản gốc", calls[1][0])
                    self.assertAlmostEqual(calls[1][1], 0.40)

    async def test_duplicate_translation_retry(self):
        with patch.object(TranslationPipeline, '_load_yaml', side_effect=self.mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client'):
                    pipeline = TranslationPipeline(project='test_proj')
                    
                    translated_chunks = {
                        0: "Anh ta đứng nhìn xa xăm."
                    }
                    
                    calls = []
                    async def mock_translate_chunk(chunk, summary, glossary, chars, lang, contents=None, retry_instruction=None, temperature=None):
                        calls.append((retry_instruction, temperature))
                        if len(calls) == 1:
                            return "<translation>Anh ta đứng nhìn xa xăm.</translation>"
                        return "<translation>Cố nhân đã khuất bóng từ lâu.</translation>"
                        
                    pipeline._translate_chunk = mock_translate_chunk
                    
                    chunk = "He watched the distant horizon."
                    result = await pipeline._translate_chunk_with_retry(
                        chunk=chunk,
                        context_snippet="",
                        glossary_str="",
                        characters_str="",
                        source_lang="en",
                        chunk_idx=1,
                        translated_chunks=translated_chunks
                    )
                    
                    self.assertEqual(result, "Cố nhân đã khuất bóng từ lâu.")
                    self.assertEqual(len(calls), 2)
                    self.assertIsNone(calls[0][0])
                    self.assertEqual(calls[0][1], 0.25)
                    self.assertIsNotNone(calls[1][0])
                    self.assertIn("lặp lại", calls[1][0])
                    self.assertAlmostEqual(calls[1][1], 0.40)


if __name__ == '__main__':
    unittest.main()
