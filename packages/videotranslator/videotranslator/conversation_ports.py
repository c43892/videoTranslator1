"""Replaceable conversation understanding and video acquisition boundaries."""
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict


class Interpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detected_locale: str = ""
    explicit_locale: str = ""
    source_kind: str = ""
    youtube_url: str = ""
    target_language: str = ""
    intent: Literal["product", "off_topic", "complaint"] = "product"
    # Optional clarification, never instructions to the execution layer.
    reply: str = ""
    mode: str = "guided"


class ConversationInterpreter(Protocol):
    def interpret(self, text: str, context: dict) -> Interpretation: ...


class VideoSourceImporter(Protocol):
    def download(self, url: str, destination: Path, max_bytes: int) -> Path: ...
