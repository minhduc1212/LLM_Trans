import unittest
import asyncio
import threading
import time
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TranslationPipeline


class TestConcurrency(unittest.IsolatedAsyncioTestCase):
    async def test_global_semaphore_limit(self):
        # We want to test that the number of concurrent API calls doesn't exceed max_concurrent_requests.
        max_concurrent = 3
        
        # Mock settings dict
        mock_settings = {
            'features': {
                'auto_glossary': False,
                'auto_summary': False,
                'clean_thinking_tags': False,
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
                'max_concurrent_requests': max_concurrent
            }
        }
        
        # Patch yaml loading to return mock configs
        def mock_load_yaml(path):
            if 'settings.yaml' in path:
                return mock_settings
            elif 'genres.yaml' in path:
                return {'genres': {'default': {'label': 'Default'}}, 'languages': {'vi': {}}}
            elif 'system_prompts.yaml' in path:
                return {}
            return {}

        with patch.object(TranslationPipeline, '_load_yaml', side_effect=mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client') as mock_client:
                    pipeline = TranslationPipeline(project='test_proj')
                    
                    # Track concurrency
                    lock = threading.Lock()
                    active_calls = 0
                    max_active_observed = 0
                    
                    async def mock_call_async(*args, **kwargs):
                        nonlocal active_calls, max_active_observed
                        with lock:
                            active_calls += 1
                            if active_calls > max_active_observed:
                                max_active_observed = active_calls
                        
                        await asyncio.sleep(0.05)
                        
                        with lock:
                            active_calls -= 1
                        return "<translation>Bản dịch giả lập</translation>"
                    
                    pipeline._call_genai = mock_call_async
                    async def mock_pre_extract(*args, **kwargs):
                        return {}
                    pipeline._pre_extract_glossary_and_characters = mock_pre_extract
                    
                    # Run 10 parallel translation tasks
                    tasks = [pipeline._run_api_task(pipeline._call_genai, "sys", "user") for _ in range(10)]
                    await asyncio.gather(*tasks)
                    
                    print(f"\n[TEST INFO] Max active concurrent requests observed: {max_active_observed}")
                    self.assertTrue(max_active_observed <= max_concurrent, 
                                    f"Observed {max_active_observed} active calls, which is > {max_concurrent}")
                    self.assertTrue(max_active_observed > 0)


if __name__ == '__main__':
    unittest.main()
