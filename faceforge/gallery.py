"""Small, local, opt-in identity gallery. Embeddings are biometric data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
from .core import cosine


class FaceGallery:
    def __init__(self, path: str | Path = "private/gallery.json"):
        self.path = Path(path)
        self.entries: dict[str, list[float]] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Malformed gallery")
            self.entries = data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
        if os.name == "posix":
            self.path.chmod(0o600)

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("Enter a name between 1 and 100 characters")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError("Expected a finite 512-d embedding")
        self.entries[name] = (vector / max(np.linalg.norm(vector), 1e-10)).tolist()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._save()

    def delete(self, name: str) -> bool:
        if name not in self.entries:
            return False
        del self.entries[name]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._save()
        return True

    def search(self, embedding: np.ndarray, threshold: float = 0.68) -> tuple[str, float]:
        if not self.entries:
            return "No enrolled identities", 0.0
        scores = [(name, cosine(embedding, np.asarray(vector))) for name, vector in self.entries.items()]
        name, score = max(scores, key=lambda item: item[1])
        return (name if score >= threshold else "Unknown"), float(score)
