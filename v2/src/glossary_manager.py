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
  - Format terms for prompt injection (limited to last N entries to save tokens)
"""

import json
from datetime import datetime
from pathlib import Path


MAX_TERMS_IN_PROMPT = 40    # Show only latest N terms in prompt (token budget)
MAX_CHARS_IN_PROMPT = 10   # Show all characters (usually small)


class GlossaryManager:
    def __init__(self, project: str, genre: str = "", base_dir: str = "data"):
        self.path = Path(base_dir) / f"{project}_glossary.json"
        self.project = project
        self.genre = genre
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

    def update_terms(self, new_terms: dict) -> int:
        """
        Merge new terms into glossary. Returns number of NEW terms added.
        Existing terms are NOT overwritten (first translation wins).
        """
        added = 0
        for term, translation in new_terms.items():
            if term and translation and term not in self._data["terms"]:
                self._data["terms"][term] = translation
                added += 1
        if added:
            self.save()
        return added

    def update_characters(self, new_chars: dict) -> int:
        """
        Merge character info. Characters CAN be updated (context may change).
        """
        added = 0
        for name, info in new_chars.items():
            if name and info:
                self._data["characters"][name] = info
                added += 1
        if added:
            self.save()
        return added

    # Alias used in pipeline
    def update(self, new_terms: dict) -> int:
        return self.update_terms(new_terms)

    def get_all(self) -> dict:
        return self._data["terms"]

    def get_characters(self) -> dict:
        return self._data["characters"]

    def format_terms_for_prompt(self) -> str:
        """Format terms for injection into prompt. Returns empty string if no terms."""
        terms = self._data["terms"]
        if not terms:
            return ""
        # Take the most recent N terms (ordered dict in Python 3.7+)
        recent = dict(list(terms.items())[-MAX_TERMS_IN_PROMPT:])
        return '\n'.join(f"- {k} → {v}" for k, v in recent.items())

    def format_characters_for_prompt(self) -> str:
        """Format character list for prompt injection."""
        chars = self._data["characters"]
        if not chars:
            return ""
        return '\n'.join(f"- {k}: {v}" for k, v in chars.items())

    def has_term(self, term: str) -> bool:
        return term in self._data["terms"]

    def get_term(self, term: str) -> str | None:
        return self._data["terms"].get(term)

    def term_count(self) -> int:
        return len(self._data["terms"])

    def char_count(self) -> int:
        return len(self._data["characters"])

    def export_report(self) -> str:
        """Export glossary as a human-readable report."""
        lines = [
            f"# Glossary Report - {self.project}",
            f"Genre: {self.genre}",
            f"Last updated: {self._data['_meta'].get('last_updated', 'N/A')}",
            "",
            f"## Thuật ngữ ({self.term_count()} terms)",
        ]
        for term, translation in self._data["terms"].items():
            lines.append(f"- **{term}** → {translation}")

        lines += ["", f"## Nhân vật ({self.char_count()} characters)"]
        for name, info in self._data["characters"].items():
            lines.append(f"- **{name}**: {info}")

        return '\n'.join(lines)