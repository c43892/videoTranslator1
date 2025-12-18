"""
OpenAI API Transcriber implementation for Step 3: Audio Transcription
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import subprocess
import shutil


try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import Transcriber, TranscriptionResult, LANGUAGE_CODES
from .transcriber import WhisperTranscriber # Import for static helpers if needed

try:
    from ..utils.srt_handler import parse_srt, merge_close_subtitles, save_srt
except (ImportError, ValueError):
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.srt_handler import parse_srt, merge_close_subtitles, save_srt


logger = logging.getLogger(__name__)




class OpenAITranscriber(Transcriber):
    """
    OpenAI API-based audio transcription implementation.
    Uses OpenAI's Whisper API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "whisper-1",
    ):
        """
        Initialize OpenAI transcriber.
        
        Args:
            api_key: OpenAI API key (defaults to env OPENAI_API_KEY)
            model_name: Model name (default: whisper-1)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is not installed. Install with: pip install openai"
            )
        
        # Hardcoded API key for local dev as requested
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass in init.")
            
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"OpenAITranscriber initialized with model '{model_name}'")
        
        # We can reuse the non-dialogue detection logic from WhisperTranscriber if needed,
        # but OpenAI API returns text/SRT directly. We'll parse the SRT response.

    def validate_input(self, audio_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate input audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            return False, f"Audio file not found: {audio_path}"
        
        if not audio_path.is_file():
            return False, f"Path is not a file: {audio_path}"
        
        # Check file size (at least 1KB)
        if audio_path.stat().st_size < 1024:
            return False, f"Audio file is too small (< 1KB): {audio_path}"
        
        # Check file extension
        valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm'}
        if audio_path.suffix.lower() not in valid_extensions:
            logger.warning(
                f"Audio file extension '{audio_path.suffix}' may not be supported. "
                f"Supported: {valid_extensions}"
            )
        
        return True, None

    def transcribe(
        self,
        audio_path: Path,
        output_path: Path,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio to SRT format using OpenAI API.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to output SRT file
            language: Language code (None for auto-detection)
            
        Returns:
            TranscriptionResult with transcription metadata
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        
        if not audio_path.exists():
            return TranscriptionResult(success=False, error_message=f"File not found: {audio_path}")

        # Normalize language code
        if language:
            original_lang = language
            language = LANGUAGE_CODES.get(language.lower().strip(), language)
            if language != original_lang:
                logger.info(f"Normalized language code: '{original_lang}' -> '{language}'")

        try:
            # Check file size and compress if necessary
            upload_path = self._compress_audio_if_needed(audio_path)
            
            logger.info(f"Uploading '{upload_path.name}' to OpenAI Whisper API...")
            
            # Prepare file for upload
            with open(upload_path, "rb") as audio_file:
                # Call OpenAI API
                # We request 'srt' format directly to save processing
                response = self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=audio_file,
                    response_format="srt",
                    language=language,
                    prompt=kwargs.get('initial_prompt', "Use proper punctuation. Do not divide in the middle of a sentence.")
                )
            
            srt_content = response
            
            # Clean up compressed file if we created one
            if upload_path != audio_path:
                try:
                    os.remove(upload_path)
                    logger.info(f"Removed temporary compressed file: {upload_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {upload_path}: {e}")

            # OpenAI API returns raw SRT string when response_format="srt"
            # We still want to parse and clean it using our utilities
            
            logger.info("DEBUG: Parsing SRT from API response")
            srt_entries = parse_srt(srt_content)
            original_count = len(srt_entries)
            logger.info(f"Original subtitle count: {original_count}")
            
            # Merge close subtitles
            srt_entries = merge_close_subtitles(srt_entries, min_gap_ms=200, max_merged_duration=10.0)
            merged_count = len(srt_entries)
            
            if merged_count < original_count:
                logger.info(f"Merged close subtitles: {original_count} -> {merged_count} entries")
            
            # Save merged SRT file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_srt(srt_entries, output_path)
            
            logger.info(f"SRT file saved to '{output_path}'")
            
            # We don't get detailed segments metadata from "srt" response format easily 
            # without re-parsing, but that's fine for now.
            
            return TranscriptionResult(
                success=True,
                srt_path=output_path,
                detected_language=language or "auto", # We don't get detected language in SRT mode easily
                segment_count=len(srt_entries),
                duration=0.0, # Unknown without parsing audio or srt
                model_used=self.model_name,
                metadata={'segments': len(srt_entries)}
            )
            
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}", exc_info=True)
            
            # Additional debug info for API errors
            error_details = str(e)
            if hasattr(e, 'response'):
                try:
                    logger.error(f"API Response Status: {e.response.status_code}")
                    logger.error(f"API Response Body: {e.response.text}")
                    error_details += f"\nResponse: {e.response.text}"
                except:
                    pass
            
            print(f"\n❌ OpenAI API Error: {error_details}")
            
            return TranscriptionResult(
                success=False,
                error_message=str(e)
            )

    def get_supported_models(self) -> List[str]:
        return ['whisper-1']

    def _compress_audio_if_needed(self, audio_path: Path) -> Path:
        """
        Check if audio file exceeds limit (24MB) and compress if needed.
        
        Args:
            audio_path: Path to original audio file
            
        Returns:
            Path to audio file to use (original or compressed)
        """
        # OpenAI limit is 25MB, we use 24MB as safety threshold
        LIMIT_BYTES = 24 * 1024 * 1024
        
        file_size = audio_path.stat().st_size
        if file_size <= LIMIT_BYTES:
            return audio_path
            
        logger.info(f"Audio file size ({file_size/1024/1024:.2f}MB) exceeds 24MB limit. Compressing...")
        
        # Determine ffmpeg path (assume it's in PATH)
        ffmpeg_cmd = "ffmpeg"
        
        # Try 32k bitrate first
        output_path_32k = audio_path.parent / f"compressed_32k_{audio_path.name}.mp3"
        if self._run_compression(audio_path, output_path_32k, "32k"):
            new_size = output_path_32k.stat().st_size
            if new_size <= LIMIT_BYTES:
                logger.info(f"Compressed to 32k: {new_size/1024/1024:.2f}MB")
                return output_path_32k
            else:
                logger.info(f"32k compression still too large ({new_size/1024/1024:.2f}MB). Trying 16k...")
                # Remove intermediate file
                try:
                    os.remove(output_path_32k)
                except:
                    pass
        
        # Try 16k bitrate
        output_path_16k = audio_path.parent / f"compressed_16k_{audio_path.name}.mp3"
        if self._run_compression(audio_path, output_path_16k, "16k"):
            new_size = output_path_16k.stat().st_size
            if new_size <= LIMIT_BYTES:
                logger.info(f"Compressed to 16k: {new_size/1024/1024:.2f}MB")
                return output_path_16k
            else:
                logger.warning(f"16k compression still exceeds limit ({new_size/1024/1024:.2f}MB). Using it anyway and hoping for the best.")
                return output_path_16k
                
        # If compression failed, return original and let it fail at API level
        logger.error("Compression failed. Using original file.")
        return audio_path

    def _run_compression(self, input_path: Path, output_path: Path, bitrate: str) -> bool:
        """Run ffmpeg compression."""
        try:
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-b:a", bitrate,
                "-y",
                str(output_path)
            ]
            
            # Suppress output unless error
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True  # Raise CalledProcessError on non-zero exit
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg compression failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error running FFmpeg: {e}")
            return False

