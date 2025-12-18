"""
Translation processor for Step 4: Subtitle Translation
Translates SRT subtitles using OpenAI ChatGPT API.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI

try:
    from .base import ProcessorResult
    from ..utils.srt_handler import SRTEntry, load_srt, save_srt
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from processors.base import ProcessorResult
    from utils.srt_handler import SRTEntry, load_srt, save_srt


logger = logging.getLogger(__name__)


@dataclass
class TranslationResult(ProcessorResult):
    """Result of translation operation"""
    json_output_path: Optional[Path] = None
    srt_output_path: Optional[Path] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    translated_count: int = 0
    skipped_count: int = 0


class Translator:
    """
    OpenAI ChatGPT-based subtitle translator.
    
    Translates SRT subtitle content from source language to target language
    while preserving timestamps and handling non-dialogue events.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini",
        temperature: float = 0.3,
        max_retries: int = 3,
        timeout: int = 120,
        chunk_size: int = 15
    ):
        """
        Initialize translator.
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Model to use for translation (default: gpt-5-mini)
            temperature: Temperature for API calls (lower = more consistent)
            max_retries: Maximum number of retry attempts for failed API calls
            timeout: Timeout for API calls in seconds (default: 120)
            chunk_size: Number of entries per batch (default: 15)
        """
        # Hardcoded API key for local dev as requested
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY env var.")

        
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.chunk_size = chunk_size
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        
        logger.info(f"Initialized Translator with model: {model}")
    
    def is_pure_event(self, text: str) -> bool:
        """
        Check if text is a pure non-dialogue event.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is pure event (e.g., [[ laughing ]]), False otherwise
        """
        text = text.strip()
        return bool(re.match(r'^\[\[.*\]\]$', text))
    
    def remove_events(self, text: str) -> str:
        """
        Remove all event markers from text.
        
        Args:
            text: Text containing potential event markers
            
        Returns:
            Text with all [[ event ]] markers removed
        """
        # Remove all [[ event ]] patterns
        cleaned = re.sub(r'\[\[.*?\]\]', '', text)
        # Clean up whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    def build_translation_prompt(
        self,
        text: str,
        source_language: str,
        target_language: str,
        terminology: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Build translation prompt for ChatGPT.
        
        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language
            terminology: Optional terminology reference dict
            
        Returns:
            Formatted prompt string
        """
        if terminology:
            # Enhanced prompt with terminology
            terminology_str = json.dumps(terminology, ensure_ascii=False, indent=2)
            prompt = f"""You are a professional translator. Translate the following text from {source_language} to {target_language}.

Requirements:
1. Maintain the natural tone and style of the original text
2. Preserve any formatting or special characters
3. STRICTLY follow the terminology reference provided below for specific terms
4. Provide only the translated text without explanations

Terminology Reference:
{terminology_str}

Text to translate: {text}"""
        else:
            # Base prompt without terminology
            prompt = f"""You are a professional translator. Translate the following text from {source_language} to {target_language}.

Requirements:
1. Maintain the natural tone and style of the original text
2. Preserve any formatting or special characters
3. Provide only the translated text without explanations

Text to translate: {text}"""
        
        return prompt
    
    def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
        terminology: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Translate a single text string using ChatGPT.
        
        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language
            terminology: Optional terminology reference
            
        Returns:
            Translated text
            
        Raises:
            Exception: If translation fails after max retries
        """
        prompt = self.build_translation_prompt(
            text, source_language, target_language, terminology
        )
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Translation attempt {attempt + 1}/{self.max_retries}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature
                )
                
                translated = response.choices[0].message.content.strip()
                
                if not translated:
                    raise ValueError("Empty translation received from API")
                
                logger.debug(f"Translation successful: '{text[:50]}...' -> '{translated[:50]}...'")
                return translated
                
            except Exception as e:
                last_error = e
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retry (exponential backoff)
                    import time
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries failed
        raise Exception(f"Translation failed after {self.max_retries} attempts: {last_error}")
    
    def translate_srt_entry(
        self,
        entry: SRTEntry,
        source_language: str,
        target_language: str,
        terminology: Optional[Dict[str, str]] = None
    ) -> Tuple[str, bool]:
        """
        Translate a single SRT entry.
        
        Args:
            entry: SRT entry to translate
            source_language: Source language
            target_language: Target language
            terminology: Optional terminology reference
            
        Returns:
            Tuple of (translated_text, was_translated)
            - was_translated is False if text was skipped (pure event or error)
        """
        original_text = entry.text.strip()
        
        # Check if pure event
        if self.is_pure_event(original_text):
            logger.debug(f"Skipping pure event: {original_text}")
            return original_text, False
        
        # Check for mixed content (events + dialogue)
        if '[[' in original_text and ']]' in original_text:
            # Remove events and get dialogue part
            cleaned_text = self.remove_events(original_text)
            
            if not cleaned_text:
                # After removing events, nothing left
                logger.debug(f"Empty after event removal, keeping original: {original_text}")
                return original_text, False
            
            # Translate the cleaned text
            logger.debug(f"Mixed content detected. Translating: '{cleaned_text}'")
            try:
                translated = self.translate_text(
                    cleaned_text, source_language, target_language, terminology
                )
                return translated, True
            except Exception as e:
                logger.error(f"Translation failed for '{cleaned_text}': {e}")
                return original_text, False
        
        # Pure dialogue - translate as-is
        try:
            translated = self.translate_text(
                original_text, source_language, target_language, terminology
            )
            return translated, True
        except Exception as e:
            logger.error(f"Translation failed for '{original_text}': {e}")
            return original_text, False
    
    def translate_batch(
        self,
        entries: List[SRTEntry],
        source_language: str,
        target_language: str,
        terminology: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Translate all SRT entries in a single batch API call.
        
        Args:
            entries: List of SRT entries to translate
            source_language: Source language
            target_language: Target language
            terminology: Optional terminology reference
            
        Returns:
            List of translated texts (same order as input)
        """
        # Build batch input - JSON array format for LLM
        batch_items = []
        for entry in entries:
            batch_items.append({
                "seq": entry.index,
                "text": entry.text
            })
        
        batch_json = json.dumps(batch_items, ensure_ascii=False, indent=2)
        
        # Build prompt for batch translation
        if terminology:
            terminology_str = json.dumps(terminology, ensure_ascii=False, indent=2)
            prompt = f"""You are a professional translator. Translate the following subtitle entries from {source_language} to {target_language}.

Requirements:
1. Maintain the natural tone and style of the original text
2. Preserve dialogue coherence and context across entries
3. STRICTLY follow the terminology reference provided below for specific terms
4. For entries that are pure events (text matching pattern [[ event ]]), keep them unchanged
5. For mixed content containing [[ event ]] markers, remove the event markers and translate only the dialogue part
6. Return a JSON array with the same structure, replacing "text" with "translation"

Terminology Reference:
{terminology_str}

Input JSON:
{batch_json}

Output only the JSON array with translations. Do not include explanations or markdown code blocks."""
        else:
            prompt = f"""You are a professional translator. Translate the following subtitle entries from {source_language} to {target_language}.

Requirements:
1. Maintain the natural tone and style of the original text
2. Preserve dialogue coherence and context across entries
3. For entries that are pure events (text matching pattern [[ event ]]), keep them unchanged
4. For mixed content containing [[ event ]] markers, remove the event markers and translate only the dialogue part
5. Return a JSON array with the same structure, replacing "text" with "translation"

Input JSON:
{batch_json}

Output only the JSON array with translations. Do not include explanations or markdown code blocks."""
        
        # Call API with retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Batch translation attempt {attempt + 1}/{self.max_retries}")
                
                # Build API call parameters
                api_params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a professional translator. Output only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
                
                # Only add temperature if model supports it (gpt-5-mini doesn't support custom temperature)
                if not self.model.startswith("gpt-5"):
                    api_params["temperature"] = self.temperature
                
                response = self.client.chat.completions.create(**api_params)
                
                response_text = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    lines = response_text.split('\n')
                    response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                
                # Parse JSON response
                translated_items = json.loads(response_text)
                
                if not isinstance(translated_items, list) or len(translated_items) != len(entries):
                    raise ValueError(f"Expected {len(entries)} translations, got {len(translated_items) if isinstance(translated_items, list) else 'non-list'}")
                
                # Extract translations in order
                translations = [item.get("translation", item.get("text", "")) for item in translated_items]
                
                logger.info(f"Batch translation successful: {len(translations)} entries translated")
                return translations
                
            except Exception as e:
                last_error = e
                logger.warning(f"Batch translation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries failed
        raise Exception(f"Batch translation failed after {self.max_retries} attempts: {last_error}")
    
    def translate_srt(
        self,
        srt_path: Path,
        source_language: str,
        target_language: str,
        output_dir: Optional[Path] = None,
        terminology: Optional[Dict[str, str]] = None
    ) -> TranslationResult:
        """
        Translate an SRT file in a single batch.
        
        Args:
            srt_path: Path to input SRT file (SRT_original.srt)
            source_language: Source language name or code
            target_language: Target language name or code
            output_dir: Output directory (default: same as input)
            terminology: Optional terminology reference dictionary
            
        Returns:
            TranslationResult with paths to output files
        """
        logger.info(f"Starting translation: {srt_path}")
        logger.info(f"Source: {source_language} -> Target: {target_language}")
        
        srt_path = Path(srt_path)
        if not srt_path.exists():
            return TranslationResult(
                success=False,
                error_message=f"Input SRT file not found: {srt_path}"
            )
        
        # Set output directory
        if output_dir is None:
            output_dir = srt_path.parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output files
        json_output = output_dir / "translation_output.json"
        srt_output = output_dir / "SRT_translated.srt"
        
        try:
            # Load SRT entries (already merged in Step 3)
            entries = load_srt(srt_path)
            logger.info(f"Loaded {len(entries)} entries from SRT file")
            
            if not entries:
                return TranslationResult(
                    success=False,
                    error_message="No entries found in SRT file"
                )
            
            # Split into chunks and translate
            logger.info(f"Translating {len(entries)} entries in chunks of {self.chunk_size}...")
            translations = []
            
            for i in range(0, len(entries), self.chunk_size):
                chunk = entries[i:i + self.chunk_size]
                chunk_num = (i // self.chunk_size) + 1
                total_chunks = (len(entries) + self.chunk_size - 1) // self.chunk_size
                
                logger.info(f"Translating chunk {chunk_num}/{total_chunks} ({len(chunk)} entries)...")
                chunk_translations = self.translate_batch(
                    chunk, source_language, target_language, terminology
                )
                translations.extend(chunk_translations)
                logger.info(f"Chunk {chunk_num}/{total_chunks} complete")
            
            # Build output data
            translation_data = []
            translated_entries = []
            translated_count = 0
            skipped_count = 0
            
            for i, (entry, translated_text) in enumerate(zip(entries, translations)):
                # Check if translation is same as original (likely skipped event)
                if self.is_pure_event(entry.text) and translated_text == entry.text:
                    skipped_count += 1
                else:
                    translated_count += 1
                
                # Build JSON output entry
                json_entry = {
                    "seq": entry.index,
                    "start": entry.start_time,
                    "end": entry.end_time,
                    "text": entry.text,
                    "translation": translated_text
                }
                translation_data.append(json_entry)
                
                # Build SRT entry for translated file
                translated_entry = SRTEntry(
                    index=entry.index,
                    start_time=entry.start_time,
                    end_time=entry.end_time,
                    text=translated_text
                )
                translated_entries.append(translated_entry)
            
            # Save JSON output
            logger.info(f"Saving JSON output to: {json_output}")
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(translation_data, f, ensure_ascii=False, indent=2)
            
            # Save SRT output
            logger.info(f"Saving SRT output to: {srt_output}")
            save_srt(translated_entries, srt_output)
            
            logger.info(f"Translation complete: {translated_count} translated, {skipped_count} skipped")
            
            return TranslationResult(
                success=True,
                json_output_path=json_output,
                srt_output_path=srt_output,
                source_language=source_language,
                target_language=target_language,
                translated_count=translated_count,
                skipped_count=skipped_count,
                metadata={
                    "total_entries": len(entries),
                    "model": self.model,
                    "temperature": self.temperature
                }
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}", exc_info=True)
            return TranslationResult(
                success=False,
                error_message=str(e)
            )


class GPT4Translator:
    """
    Wrapper class for GPT-4 based translation with simplified process() interface.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        target_language: str = "English",
        source_language: str = "auto",
        model: str = "gpt-5-mini",
        chunk_size: int = 15
    ):
        """
        Initialize GPT4Translator.
        
        Args:
            api_key: OpenAI API key
            target_language: Target language for translation
            source_language: Source language (default: auto-detect)
            model: OpenAI model to use (default: gpt-5-mini)
            chunk_size: Number of entries per batch (default: 15)
        """
        self.target_language = target_language
        self.source_language = source_language
        self.translator = Translator(api_key=api_key, model=model, chunk_size=chunk_size)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process translation with simplified interface.
        
        Args:
            input_data: {
                'srt_file': path to SRT_original.srt,
                'output_dir': path for output files,
                'target_language': (optional) override target language,
                'source_language': (optional) override source language,
                'terminology_file': (optional) path to terminology JSON file
            }
            
        Returns:
            {
                'srt_translated': path to SRT_translated.srt,
                'json_output': path to translation_output.json,
                'translated_count': number of translated entries,
                'skipped_count': number of skipped entries
            }
        """
        srt_file = input_data['srt_file']
        output_dir = input_data['output_dir']
        target_lang = input_data.get('target_language', self.target_language)
        source_lang = input_data.get('source_language', self.source_language)
        terminology_file = input_data.get('terminology_file')
        
        # Load terminology if provided
        terminology = None
        if terminology_file and Path(terminology_file).exists():
            logger.info(f"Loading terminology from: {terminology_file}")
            try:
                with open(terminology_file, 'r', encoding='utf-8') as f:
                    terminology = json.load(f)
                logger.info(f"Loaded {len(terminology)} terminology entries")
            except Exception as e:
                logger.warning(f"Failed to load terminology file: {e}")
        
        # Translate
        result = self.translator.translate_srt(
            srt_path=Path(srt_file),
            source_language=source_lang,
            target_language=target_lang,
            output_dir=Path(output_dir),
            terminology=terminology
        )
        
        if not result.success:
            raise RuntimeError(f"Translation failed: {result.error_message}")
        
        return {
            'srt_translated': str(result.srt_output_path),
            'json_output': str(result.json_output_path),
            'translated_count': result.translated_count,
            'skipped_count': result.skipped_count
        }
