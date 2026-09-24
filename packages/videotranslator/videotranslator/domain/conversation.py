"""A reviewable input draft. Interpreting text never authorizes execution."""
from dataclasses import dataclass, field
from typing import ClassVar

from .models import Document


@dataclass
class Conversation(Document):
    COLLECTION: ClassVar[str] = "conversations"
    conversation_id: str = ""
    owner_user_id: str = ""
    revision: int = 0
    locale: str = "en"
    explicit_locale: str = ""
    source_kind: str = ""
    youtube_url: str = ""
    filename: str = ""
    size_bytes: int = 0
    target_language: str = ""
    messages: list[dict] = field(default_factory=list)
    round_start: int = 0
    status: str = "draft"
    upload_id: str = ""
    job_id: str = ""
    lease_until: int = 0
    download_task_id: str = ""
    error: str = ""
    interpreter_mode: str = "guided"
    duration_ms: int = 0
    duration_probe_raw: str = ""
    media_type: str = "video"
    quoted_cents: int = 0
    pricing_version: str = ""

    @property
    def ready(self) -> bool:
        return bool(self.target_language and (
            self.source_kind == "youtube" and self.youtube_url
            or self.source_kind == "upload" and self.filename and self.size_bytes > 0
        ))
