"""Swappable file storage for trained model artifacts.

Ships with a local-disk implementation, which is enough for a single
backend instance backed by a persistent volume (Docker Compose, Render
disk, Fly volume). Swap in an S3-compatible implementation behind the
same interface if the backend needs to scale horizontally.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings

settings = get_settings()


class FileStorage(Protocol):
    def save_bytes(self, key: str, data: bytes) -> str: ...
    def read_bytes(self, key: str) -> bytes: ...
    def new_key(self, prefix: str, suffix: str = ".pkl") -> str: ...


class LocalFileStorage:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or settings.storage_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def new_key(self, prefix: str, suffix: str = ".pkl") -> str:
        return f"{prefix}/{uuid.uuid4().hex}{suffix}"

    def _path_for(self, key: str) -> Path:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_bytes(self, key: str, data: bytes) -> str:
        path = self._path_for(key)
        path.write_bytes(data)
        return str(path)

    def read_bytes(self, key: str) -> bytes:
        return self._path_for(key).read_bytes()


def get_storage() -> FileStorage:
    return LocalFileStorage()
