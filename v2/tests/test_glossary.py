import unittest
import asyncio
import shutil
from pathlib import Path
import sys

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.glossary_manager import GlossaryManager


class TestGlossary(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_dir = Path("test_glossary_data")
        self.test_dir.mkdir(exist_ok=True)
        self.project = "test_project"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_relevance_filtering(self):
        mgr = GlossaryManager(self.project, base_dir=str(self.test_dir))
        
        mgr.update_terms({
            "Ariadne": "A-ri-át-nê",
            "torpor": "ngủ đông",
            "warp drive": "động cơ bẻ cong không gian"
        })
        
        mgr.update_characters({
            "Ariadne": "Ariadne | nhân vật chính | cô",
            "Zephyr": "Zephyr | thuyền trưởng | anh"
        })
        
        chunk_text = "Ariadne woke up from torpor after a long journey."
        
        formatted_terms = mgr.format_terms_for_prompt(chunk_text)
        formatted_chars = mgr.format_characters_for_prompt(chunk_text)
        
        self.assertIn("Ariadne", formatted_terms)
        self.assertIn("torpor", formatted_terms)
        self.assertNotIn("warp drive", formatted_terms)
        
        self.assertIn("Ariadne", formatted_chars)
        self.assertNotIn("Zephyr", formatted_chars)

    def test_character_merge_field_by_field(self):
        mgr = GlossaryManager(self.project, base_dir=str(self.test_dir))
        
        mgr.update_characters({
            "Ariadne": "Ariadne | nhân vật chính | cô"
        })
        
        mgr.update_characters({
            "Ariadne": "Ariadne | phi hành gia nữ | tôi"
        })
        
        chars = mgr.get_characters()
        self.assertEqual(chars["Ariadne"], "Ariadne | nhân vật chính, phi hành gia nữ | cô / tôi")

    async def test_concurrent_updates_no_data_loss(self):
        mgr = GlossaryManager(self.project, base_dir=str(self.test_dir))
        loop = asyncio.get_event_loop()
        
        async def run_update(worker_id: int):
            await loop.run_in_executor(
                None,
                mgr.update_terms,
                {f"term_{worker_id}": f"translation_{worker_id}"}
            )

        tasks = [run_update(i) for i in range(20)]
        await asyncio.gather(*tasks)
        
        reloaded_mgr = GlossaryManager(self.project, base_dir=str(self.test_dir))
        all_terms = reloaded_mgr.get_all()
        
        self.assertEqual(len(all_terms), 20)
        for i in range(20):
            self.assertIn(f"term_{i}", all_terms)
            self.assertEqual(all_terms[f"term_{i}"], f"translation_{i}")


if __name__ == '__main__':
    unittest.main()
