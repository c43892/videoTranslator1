from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    database_url: str = 'sqlite:///./local.db'
    redis_url: str = 'redis://redis:6379/0'
    storage_root: str = '/data'
    mvsep_api_key: str = ''
    openai_api_key: str = ''
    elevenlabs_api_key: str = ''
    transcription_provider: Literal['whisper', 'scribe'] = 'whisper'
    elevenlabs_base_url: str = 'https://api.elevenlabs.io/v1'
    scribe_model: str = 'scribe_v2'
    deepseek_api_key: str = ''
    mvsep_base_url: str = 'https://de2.mvsep.com/api'
    separation_provider: Literal['demucs', 'mvsep'] = 'demucs'
    demucs_url: str = 'http://tts:8001'
    demucs_model: Literal['htdemucs', 'htdemucs_ft'] = 'htdemucs'
    demucs_segment_seconds: float = Field(default=5, ge=1, le=7.8)
    whisper_model: str = 'whisper-1'
    openai_base_url: str = 'https://api.openai.com/v1'
    diarization_model: str = 'gpt-4o-transcribe-diarize'
    deepseek_base_url: str = 'https://api.deepseek.com'
    deepseek_model: str = 'deepseek-flash'
    tts_url: str = 'http://tts:8001'
    tts_tokenize_timeout_seconds: float = Field(default=120, ge=1, le=1800)
    tts_max_text_tokens: int = 120
    speaker_reference_min_seconds: float = 2
    speaker_reference_max_seconds: float = 12
    emotion_reference_max_seconds: float = 15
    max_speedup: float = 1.25
    cookie_secure: bool = False
    allow_registration: bool = True
    max_upload_mb: int = 2048
    max_video_seconds: int = 7200
    youtube_max_height: int = 1080
    youtube_download_timeout_seconds: int = 1800

    def missing_keys(self):
        required = ['ELEVENLABS_API_KEY' if self.transcription_provider == 'scribe' else 'OPENAI_API_KEY', 'DEEPSEEK_API_KEY']
        if self.separation_provider == 'mvsep':
            required.insert(0, 'MVSEP_API_KEY')
        return [name for name in required
                if not getattr(self, name.lower()).strip()]

    def separation_label(self):
        return f'Demucs {self.demucs_model} · 本地 GPU' if self.separation_provider == 'demucs' else 'MVSep DnR v3'

    def transcription_model(self):
        return self.scribe_model if self.transcription_provider == 'scribe' else self.whisper_model

    def pipeline_config(self):
        return {k: getattr(self, k) for k in (
            'separation_provider', 'demucs_model', 'demucs_segment_seconds',
            'transcription_provider', 'scribe_model', 'elevenlabs_base_url',
            'mvsep_base_url', 'openai_base_url', 'whisper_model', 'diarization_model', 'deepseek_base_url', 'deepseek_model',
            'tts_max_text_tokens', 'speaker_reference_min_seconds', 'speaker_reference_max_seconds',
            'emotion_reference_max_seconds', 'max_speedup')}

@lru_cache
def settings():
    return Settings()
