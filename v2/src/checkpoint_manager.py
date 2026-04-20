"""
checkpoint_manager.py - Save/load/delete translation progress.

Checkpoint format (JSON):
{
  "file_stem": "chapter_01",
  "chunk_index": 5,           ← last successfully translated chunk index
  "total_chunks": 20,
  "translated_parts": [...],  ← all translated text so far
  "summary": "...",           ← context summary for next chunk
  "glossary": {...},          ← glossary state at checkpoint
  "saved_at": "ISO timestamp"
}

Files are saved to: checkpoints/{project}/{file_stem}.json
Automatically deleted when translation completes successfully.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path


class CheckpointManager:
    def __init__(self, project: str, base_dir: str = "checkpoints"):
        self.project_dir = Path(base_dir) / project
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, file_stem: str) -> Path:
        return self.project_dir / f"{file_stem}.json"

    def save(self, file_stem: str, data: dict) -> None:
        """Save checkpoint after a chunk is translated."""
        checkpoint = {
            "file_stem": file_stem,
            "saved_at": datetime.now().isoformat(),
            **data,
        }
        path = self._path(file_stem)
        # Write atomically: write to temp file then rename
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(path)  # replace() works on both Windows & Linux (overwrites existing)

    def load(self, file_stem: str) -> dict | None:
        """Load checkpoint. Returns None if not found."""
        path = self._path(file_stem)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            return data
        except (json.JSONDecodeError, IOError):
            return None

    def delete(self, file_stem: str) -> None:
        """Delete checkpoint after successful completion."""
        path = self._path(file_stem)
        if path.exists():
            path.unlink()

    def delete_project(self) -> None:
        """Delete entire project checkpoint directory."""
        if self.project_dir.exists():
            shutil.rmtree(self.project_dir)

    def list_checkpoints(self) -> list[str]:
        """List all files with active checkpoints."""
        return [p.stem for p in self.project_dir.glob('*.json')]

    def get_progress(self, file_stem: str) -> tuple[int, int] | None:
        """Returns (chunk_index, total_chunks) or None if no checkpoint."""
        data = self.load(file_stem)
        if data:
            return data.get('chunk_index', 0), data.get('total_chunks', 0)
        return None