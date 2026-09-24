"""Durable home-download queue and reverse-connected worker presence."""
from dataclasses import dataclass
from typing import ClassVar

from .models import Document


@dataclass
class DownloadWorker(Document):
    COLLECTION: ClassVar[str] = "download_workers"
    worker_id: str = ""
    last_seen: int = 0


@dataclass
class DownloadTask(Document):
    COLLECTION: ClassVar[str] = "download_tasks"
    task_id: str = ""
    conversation_id: str = ""
    url: str = ""
    status: str = "queued"
    created_at: int = 0
    next_check_at: int = 0
    unavailable_retries: int = 0
    worker_id: str = ""
    lease_token: str = ""
    lease_until: int = 0
    deadline: int = 0
    object_key: str = ""
    size_bytes: int = 0
    error: str = ""
