from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol, Callable

SCHEMA_VERSION = 1

@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker_id: str

@dataclass
class Segment:
    id: str
    speaker_id: str
    start: float
    end: float
    source_text: str
    translation: str = ''
    original_audio: str = ''
    speaker_reference: str = ''
    emotion_reference: str = ''
    synthesized_audio: str = ''
    aligned_audio: str = ''
    flags: list[str] = field(default_factory=list)
    render_status: str = 'pending'
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

@dataclass
class Stems:
    dialogue: str
    music: str
    effects: str

class Storage(Protocol):
    def path(self, key: str) -> Path: ...
    def exists(self, key: str) -> bool: ...
    def read_json(self, key: str) -> dict: ...
    def write_json(self, key: str, value: dict) -> None: ...

class Separator(Protocol):
    def separate(self, audio: str, prefix: str, checkpoint: Callable[[dict], None],
                 remote: dict | None, check_cancel: Callable[[], None]) -> Stems: ...

class Transcriber(Protocol):
    def transcribe(self, audio: str) -> dict: ...

class Translator(Protocol):
    def translate(self, segments: list[Segment], language: str, terminology: str) -> dict[str, str]: ...

class Synthesizer(Protocol):
    def tokenize(self, texts: list[str]) -> list[int]: ...
    def synthesize(self, segment: Segment, output: str) -> dict: ...

class NeedsReview(Exception):
    pass

class Cancelled(Exception):
    pass

class ProviderError(Exception):
    pass
