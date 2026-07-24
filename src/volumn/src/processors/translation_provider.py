"""Translation provider abstraction for the media translation pipeline."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class TranslationProvider(ABC):
    """Stable interface used by the pipeline regardless of API vendor."""

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate an SRT document while preserving its timestamps."""


class OpenAICompatibleTranslationProvider(TranslationProvider):
    """Adapter for providers exposing an OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        target_language: str,
        source_language: str = "auto",
        base_url: Optional[str] = None,
        chunk_size: int = 15,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        from .translator import GPT4Translator

        self.client = GPT4Translator(
            api_key=api_key,
            target_language=target_language,
            source_language=source_language,
            model=model,
            base_url=base_url,
            chunk_size=chunk_size,
            extra_body=extra_body,
        )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.client.process(input_data)


def create_translation_provider(
    provider: str,
    *,
    target_language: str,
    source_language: Optional[str] = None,
    model: Optional[str] = None,
) -> TranslationProvider:
    """Create a configured translation provider from environment credentials."""
    provider_name = provider.lower().strip()

    if provider_name == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek translation")
        return OpenAICompatibleTranslationProvider(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            target_language=target_language,
            source_language=source_language or "auto",
            extra_body={"thinking": {"type": "disabled"}},
        )

    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI translation")
        return OpenAICompatibleTranslationProvider(
            api_key=api_key,
            model=model or os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-5-mini"),
            target_language=target_language,
            source_language=source_language or "auto",
        )

    raise ValueError(f"Unsupported translation provider: {provider}")
