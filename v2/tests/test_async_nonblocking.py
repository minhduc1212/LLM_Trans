import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import sys
import asyncio

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TranslationPipeline


class TestAsyncNonBlocking(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_settings = {
            'features': {
                'auto_glossary': False,
                'auto_summary': False,
                'clean_thinking_tags': False,
                'use_async_client': True,
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
            }
        }

    def mock_load_yaml(self, path):
        if 'settings.yaml' in path:
            return self.mock_settings
        elif 'genres.yaml' in path:
            return {'genres': {'default': {'label': 'Default'}}, 'languages': {'vi': {}}}
        elif 'system_prompts.yaml' in path:
            return {
                'system_base': 'Base',
                'context_section': '{summary}',
                'glossary_section': '{terms}',
                'characters_section': '{characters}',
                'source_rules_en': 'English rules',
            }
        return {}

    async def test_async_native_client_call(self):
        with patch.object(TranslationPipeline, '_load_yaml', side_effect=self.mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client') as mock_client_cls:
                    mock_client = mock_client_cls.return_value
                    mock_response = MagicMock()
                    mock_response.text = "<translation>Async Success</translation>"
                    mock_response.candidates = []
                    
                    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
                    
                    pipeline = TranslationPipeline(project='test_proj')
                    result = await pipeline._call_genai("system instructions", "user message")
                    
                    self.assertEqual(result, "<translation>Async Success</translation>")
                    mock_client.aio.models.generate_content.assert_awaited_once()

    async def test_nonblocking_file_writes(self):
        with patch.object(TranslationPipeline, '_load_yaml', side_effect=self.mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client'):
                    pipeline = TranslationPipeline(project='test_proj')
                    
                    loop = asyncio.get_event_loop()
                    with patch.object(loop, 'run_in_executor', wraps=loop.run_in_executor) as spy_executor:
                        test_path = Path("test_write.txt")
                        try:
                            await pipeline._write_text_file(test_path, "Hello world")
                            spy_executor.assert_called()
                            self.assertEqual(test_path.read_text(encoding='utf-8'), "Hello world")
                        finally:
                            if test_path.exists():
                                test_path.unlink()


if __name__ == '__main__':
    unittest.main()
