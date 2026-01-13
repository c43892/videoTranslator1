"""
Meganova API Transcriber implementation for Step 3: Audio Transcription
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess

from .base import Transcriber, TranscriptionResult, LANGUAGE_CODES

try:
    from ..utils.srt_handler import parse_srt, merge_close_subtitles, save_srt
except (ImportError, ValueError):
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.srt_handler import parse_srt, merge_close_subtitles, save_srt


logger = logging.getLogger(__name__)


class MeganovaTranscriber(Transcriber):
    """
    Meganova API-based audio transcription implementation.
    Uses Meganova's hosted Faster-Whisper service.
    """
    
    API_URL = "https://api.meganova.ai/v1/audio/transcriptions"
    DEFAULT_MODEL = "Systran/faster-whisper-large-v3"
    
    def __init__(
        self,
        api_key: Optional[str] = "sk-0ZdE_aafPY_oRWdoFNEejQ", # Default key provided by user
        model_name: str = "Systran/faster-whisper-large-v3",
    ):
        """
        Initialize Meganova transcriber.
        
        Args:
            api_key: Meganova API key
            model_name: Model name (default: Systran/faster-whisper-large-v3)
        """
        self.api_key = api_key or os.getenv('MEGANOVA_API_KEY')
        
        if not self.api_key:
            raise ValueError("Meganova API key is required.")
            
        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info(f"MeganovaTranscriber initialized with model '{self.model_name}'")

    def validate_input(self, audio_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate input audio file.
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            return False, f"Audio file not found: {audio_path}"
        
        if not audio_path.is_file():
            return False, f"Path is not a file: {audio_path}"
        
        if audio_path.stat().st_size < 1024:
            return False, f"Audio file is too small (< 1KB): {audio_path}"
        
        return True, None

    def transcribe(
        self,
        audio_path: Path,
        output_path: Path,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio using Meganova API.
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        
        if not audio_path.exists():
            return TranscriptionResult(success=False, error_message=f"File not found: {audio_path}")

        # Normalize language code if provided
        if language:
            language_code = LANGUAGE_CODES.get(language.lower().strip(), language)
        else:
            language_code = None

        try:
            # Check file size and compress if necessary (25MB limit is common for these APIs)
            upload_path = self._compress_audio_if_needed(audio_path)
            
            logger.info(f"Uploading '{upload_path.name}' to Meganova API...")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model_name,
                "response_format": "json" # We'll parse the custom JSON response
            }
            
            if language_code:
                data["language"] = language_code

            # Meganova API call
            with open(upload_path, "rb") as audio_file:
                files = {
                    "file": (upload_path.name, audio_file, "audio/mpeg")
                }
                
                response = requests.post(
                    self.API_URL,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=300 # 5 minutes timeout for large files
                )
            
            # Clean up compressed file if we created one
            if upload_path != audio_path:
                try:
                    os.remove(upload_path)
                    logger.info(f"Removed temporary compressed file: {upload_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {upload_path}: {e}")

            if response.status_code != 200:
                raise Exception(f"API Error ({response.status_code}): {response.text}")

            # Parse response
            # Meganova/Whisper usually returns segments in JSON
            result_json = response.json()
            
            if "text" not in result_json and "segments" not in result_json:
                 raise Exception(f"Unexpected API response format: {result_json.keys()}")

            # Convert to SRT entries
            from ..utils.srt_handler import SRTEntry
            srt_entries = []
            
            # If segments are provided directly
            if "segments" in result_json:
                for idx, seg in enumerate(result_json["segments"]):
                    # Handle time format (seconds)
                    start_time = float(seg.get("start", 0))
                    end_time = float(seg.get("end", 0))
                    text = seg.get("text", "").strip()
                    
                    if text:
                        srt_entries.append(SRTEntry(
                            index=idx + 1,
                            start_time=self._format_time(start_time),
                            end_time=self._format_time(end_time),
                            text=text
                        ))
            else:
                # Fallback: if only text is returned (unlikely with response_format=json usually)
                # We might need to split by lines or sentences, but Whisper API usually gives segments
                logger.warning("No segments found in response, only full text. Creating single SRT entry.")
                srt_entries.append(SRTEntry(
                    index=1,
                    start_time="00:00:00,000",
                    end_time="00:00:10,000", # Dummy duration
                    text=result_json.get("text", "")
                ))

            logger.info(f"Parsed {len(srt_entries)} segments from API response")
            
            # Save SRT file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_srt(srt_entries, output_path)
            logger.info(f"SRT file saved to '{output_path}'")

            return TranscriptionResult(
                success=True,
                srt_path=output_path,
                detected_language=result_json.get("language", language or "auto"),
                segment_count=len(srt_entries),
                duration=result_json.get("duration", 0.0),
                model_used=self.model_name,
                metadata={'segments': len(srt_entries)}
            )

        except Exception as e:
            logger.error(f"Meganova transcription failed: {e}", exc_info=True)
            return TranscriptionResult(success=False, error_message=str(e))

    def _format_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        if seconds is None:
            return "00:00:00,000"
        
        millis = int((seconds * 1000) % 1000)
        seconds = int(seconds)
        minutes = (seconds // 60) % 60
        hours = seconds // 3600
        seconds = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    def _compress_audio_if_needed(self, audio_path: Path) -> Path:
        """
        Check if audio file exceeds limit (24MB) and compress if needed.
        (Copied from OpenAITranscriber logic)
        """
        LIMIT_BYTES = 24 * 1024 * 1024
        
        file_size = audio_path.stat().st_size
        if file_size <= LIMIT_BYTES:
            return audio_path
            
        logger.info(f"Audio file size ({file_size/1024/1024:.2f}MB) exceeds 24MB limit. Compressing...")
        
        output_path = audio_path.parent / f"compressed_{audio_path.name}.mp3"
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-b:a", "32k",
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return output_path
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return audio_path
