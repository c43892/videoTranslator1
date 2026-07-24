"""Storage boundary used by cloud jobs."""

from abc import ABC, abstractmethod
from pathlib import Path


class ObjectStorage(ABC):
    @abstractmethod
    def download(self, uri: str, destination: Path) -> Path:
        """Download an object to a local staging path."""

    @abstractmethod
    def upload(self, source: Path, uri: str) -> str:
        """Upload a local result and return its canonical URI."""
