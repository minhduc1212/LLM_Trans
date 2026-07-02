import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TranslationPipeline


class TestRollingHistory(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_settings = {
            'features': {
                'auto_glossary': False,
                'auto_summary': False,
                'clean_thinking_tags': False,
                'relevance_filtering': False,
                'inject_glossary_in_system_prompt': True,
                'rolling_history': True,
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
                'history_chapters': 2,
                'history_token_budget': 100,  # small budget to test truncation
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

    def test_build_multi_turn_contents(self):
        with patch.object(TranslationPipeline, '_load_yaml', side_effect=self.mock_load_yaml):
            with patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'}):
                with patch('google.genai.Client'):
                    pipeline = TranslationPipeline(project='test_proj')
                    
                    chunks = [
                        "Original Chunk 0 text",
                        "Original Chunk 1 text",
                        "Original Chunk 2 text",
                    ]
                    translated_chunks = {
                        0: "Bản dịch Chunk 0",
                        1: "Bản dịch Chunk 1",
                    }
                    
                    cpt = 4.0  # chars per token
                    
                    # 1. Test building history for chunk 2 (N=2, budget=1000)
                    pipeline.settings['translation']['history_token_budget'] = 1000
                    contents = pipeline._build_multi_turn_contents(
                        chunk=chunks[2],
                        context_snippet="context",
                        glossary_str="",
                        characters_str="",
                        source_lang="en",
                        idx=2,
                        chunks=chunks,
                        translated_chunks=translated_chunks,
                        cpt=cpt
                    )
                    
                    # The contents should be user (chunk 0) -> model (trans 0) -> user (chunk 1) -> model (trans 1) -> user (chunk 2)
                    self.assertEqual(len(contents), 5)
                    self.assertEqual(contents[0]["role"], "user")
                    self.assertIn("Original Chunk 0 text", contents[0]["parts"][0]["text"])
                    self.assertEqual(contents[1]["role"], "model")
                    self.assertIn("Bản dịch Chunk 0", contents[1]["parts"][0]["text"])
                    self.assertEqual(contents[2]["role"], "user")
                    self.assertIn("Original Chunk 1 text", contents[2]["parts"][0]["text"])
                    self.assertEqual(contents[3]["role"], "model")
                    self.assertIn("Bản dịch Chunk 1", contents[3]["parts"][0]["text"])
                    
                    # We set the budget to allow exactly one pair (approx 20 tokens) but not both
                    pipeline.settings['translation']['history_token_budget'] = 25
                    contents_truncated = pipeline._build_multi_turn_contents(
                        chunk=chunks[2],
                        context_snippet="context",
                        glossary_str="",
                        characters_str="",
                        source_lang="en",
                        idx=2,
                        chunks=chunks,
                        translated_chunks=translated_chunks,
                        cpt=cpt
                    )
                    
                    # Should only contain chunk 1 history + current chunk 2 (total 3 turns)
                    self.assertEqual(len(contents_truncated), 3)
                    self.assertEqual(contents_truncated[0]["role"], "user")
                    self.assertIn("Original Chunk 1 text", contents_truncated[0]["parts"][0]["text"])
                    self.assertEqual(contents_truncated[1]["role"], "model")
                    self.assertIn("Bản dịch Chunk 1", contents_truncated[1]["parts"][0]["text"])
                    self.assertEqual(contents_truncated[2]["role"], "user")
                    self.assertIn("Original Chunk 2 text", contents_truncated[2]["parts"][0]["text"])


if __name__ == '__main__':
    unittest.main()
