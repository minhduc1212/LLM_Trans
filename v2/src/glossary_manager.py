"""
glossary_manager.py - Persistent glossary for terminology and character tracking.

Stored in: data/{project}_glossary.json

Structure:
{
  "_meta": { project, genre, last_updated },
  "terms": { "torpor": "ngủ đông", "somaforming": "biến đổi cơ thể (somaforming)" },
  "characters": { "Ariadne": "nhân vật chính, xưng 'tôi'" }
}

Features:
  - Load on startup (sync across runs)
  - Update incrementally after each chunk
  - Save after every update
  - Format terms for prompt injection (with relevance filtering to save tokens)
"""

import json
import threading
from datetime import datetime
from pathlib import Path


class GlossaryManager:
    def __init__(self, project: str, genre: str = "", base_dir: str = "data"):
        self.path = Path(base_dir) / f"{project}_glossary.json"
        self.project = project
        self.genre = genre
        self.lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "_meta": {
                "project": self.project,
                "genre": self.genre,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
            },
            "terms": {},
            "characters": {},
        }

    def save(self) -> None:
        self._data["_meta"]["last_updated"] = datetime.now().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(self.path)  # replace() works on both Windows & Linux (overwrites existing)

    def _merge_character_info(self, existing_info: str, new_info: str) -> str:
        """
        Merge character info structured as 'translation | role | form of address'.
        """
        exist_parts = [p.strip() for p in existing_info.split('|')]
        new_parts = [p.strip() for p in new_info.split('|')]
        
        while len(exist_parts) < 3:
            exist_parts.append("")
        while len(new_parts) < 3:
            new_parts.append("")
            
        merged_parts = []
        # 1. Name translation
        if exist_parts[0]:
            merged_parts.append(exist_parts[0])
        else:
            merged_parts.append(new_parts[0])
            
        # 2. Role
        if exist_parts[1] and new_parts[1]:
            if new_parts[1] not in exist_parts[1]:
                merged_parts.append(f"{exist_parts[1]}, {new_parts[1]}")
            else:
                merged_parts.append(exist_parts[1])
        elif exist_parts[1]:
            merged_parts.append(exist_parts[1])
        else:
            merged_parts.append(new_parts[1])
            
        # 3. Form of address
        if exist_parts[2] and new_parts[2]:
            if new_parts[2] not in exist_parts[2]:
                merged_parts.append(f"{exist_parts[2]} / {new_parts[2]}")
            else:
                merged_parts.append(exist_parts[2])
        elif exist_parts[2]:
            merged_parts.append(exist_parts[2])
        else:
            merged_parts.append(new_parts[2])
            
        return " | ".join(merged_parts)

    def update_terms(self, new_terms: dict) -> int:
        """
        Merge new terms into glossary. Returns number of NEW terms added.
        Existing terms are NOT overwritten (first translation wins).
        """
        with self.lock:
            self._data = self._load()
            added = 0
            for term, translation in new_terms.items():
                if term and translation:
                    if term in self._data["terms"]:
                        existing = self._data["terms"][term]
                        if existing != translation:
                            print(f"      ⚠️  [Glossary Conflict] Term '{term}': existing='{existing}', proposed='{translation}'. Keeping existing.")
                    else:
                        self._data["terms"][term] = translation
                        added += 1
            if added:
                self.save()
            return added

    def update_characters(self, new_chars: dict) -> int:
        """
        Merge character info. Characters are merged field-by-field.
        """
        with self.lock:
            self._data = self._load()
            added = 0
            for name, info in new_chars.items():
                if name and info:
                    if name in self._data["characters"]:
                        existing = self._data["characters"][name]
                        merged = self._merge_character_info(existing, info)
                        if merged != existing:
                            self._data["characters"][name] = merged
                            added += 1
                    else:
                        self._data["characters"][name] = info
                        added += 1
            if added:
                self.save()
            return added

    # Alias used in pipeline
    def update(self, new_terms: dict) -> int:
        return self.update_terms(new_terms)

    def get_all(self) -> dict:
        with self.lock:
            return self._data["terms"].copy()

    def get_characters(self) -> dict:
        with self.lock:
            return self._data["characters"].copy()

    def format_terms_for_prompt(self, chunk_text: str = None) -> str:
        """Format terms for injection into prompt."""
        with self.lock:
            terms = self._data["terms"]
            if not terms:
                return ""
            
            if chunk_text is not None:
                chunk_lower = chunk_text.lower()
                filtered = {k: v for k, v in terms.items() if k.lower() in chunk_lower}
                if not filtered:
                    return ""
                return '\n'.join(f"- {k} → {v}" for k, v in filtered.items())
                
            return '\n'.join(f"- {k} → {v}" for k, v in terms.items())

    def format_characters_for_prompt(self, chunk_text: str = None) -> str:
        """Format character list for prompt injection."""
        with self.lock:
            chars = self._data["characters"]
            if not chars:
                return ""
            
            if chunk_text is not None:
                chunk_lower = chunk_text.lower()
                filtered = {k: v for k, v in chars.items() if k.lower() in chunk_lower}
                if not filtered:
                    return ""
                return '\n'.join(f"- {k}: {v}" for k, v in filtered.items())
                
            return '\n'.join(f"- {k}: {v}" for k, v in chars.items())

    def has_term(self, term: str) -> bool:
        with self.lock:
            return term in self._data["terms"]

    def get_term(self, term: str) -> str | None:
        with self.lock:
            return self._data["terms"].get(term)

    def term_count(self) -> int:
        with self.lock:
            return len(self._data["terms"])

    def char_count(self) -> int:
        with self.lock:
            return len(self._data["characters"])

    def export_report(self) -> str:
        """Export glossary as a human-readable report."""
        with self.lock:
            lines = [
                f"# Glossary Report - {self.project}",
                f"Genre: {self.genre}",
                f"Last updated: {self._data['_meta'].get('last_updated', 'N/A')}",
                "",
                f"## Thuật ngữ ({len(self._data['terms'])} terms)",
            ]
            for term, translation in self._data["terms"].items():
                lines.append(f"- **{term}** → {translation}")

            lines += ["", f"## Nhân vật ({len(self._data['characters'])} characters)"]
            for name, info in self._data["characters"].items():
                lines.append(f"- **{name}**: {info}")

            return '\n'.join(lines)