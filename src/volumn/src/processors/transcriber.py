"""
Transcriber implementations for Step 3: Audio Transcription
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

from .base import Transcriber, TranscriptionResult, LANGUAGE_CODES

try:
    from ..utils.srt_handler import parse_srt, merge_close_subtitles, save_srt
except (ImportError, ValueError):
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.srt_handler import parse_srt, merge_close_subtitles, save_srt


logger = logging.getLogger(__name__)


class WhisperTranscriber(Transcriber):
    """
    Whisper-based audio transcription implementation.
    Uses OpenAI's Whisper model for local transcription with language detection
    and non-dialogue event detection.
    """
    
    # Non-dialogue event patterns and markers
    NON_DIALOGUE_MARKERS = {
        'laugh': '[[ laughing ]]',
        'laughter': '[[ laughing ]]',
        'music': '[[ music ]]',
        'applause': '[[ applause ]]',
        'cry': '[[ crying ]]',
        'crying': '[[ crying ]]',
        'sob': '[[ crying ]]',
        'breathing': '[[ breathing ]]',
        'breath': '[[ breathing ]]',
        'panting': '[[ panting ]]',
        'pant': '[[ panting ]]',
        'sigh': '[[ sighing ]]',
        'sighing': '[[ sighing ]]',
        'cough': '[[ coughing ]]',
        'coughing': '[[ coughing ]]',
        'scream': '[[ screaming ]]',
        'screaming': '[[ screaming ]]',
        'gasp': '[[ gasping ]]',
        'gasping': '[[ gasping ]]',
        'inaudible': '[[ inaudible ]]',
        'unintelligible': '[[ inaudible ]]',
        'silence': '[[ silence ]]',
    }
    
    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        device: Optional[str] = None,
        detect_non_dialogue: bool = True,
        confidence_threshold: float = 0.7,
        max_chars_per_line: int = 80,      # Reduced from 1000 to 80 for shorter segments
        max_lines_per_subtitle: int = 2,   # Increased from 1 to 2 for flexibility
        min_duration: float = 0.5,
        max_duration: float = 7.0          # Reduced from 120.0 to 7.0 seconds
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to use ("cpu", "cuda", or None for auto)
            detect_non_dialogue: Enable non-dialogue event detection
            confidence_threshold: Minimum confidence for valid speech (0.0-1.0)
            max_chars_per_line: Maximum characters per subtitle line
            max_lines_per_subtitle: Maximum lines per subtitle entry
            min_duration: Minimum subtitle duration in seconds
            max_duration: Maximum subtitle duration in seconds
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper is not installed. Install with: pip install openai-whisper"
            )
        
        self.model_name = model_name
        self.device = device
        self.detect_non_dialogue = detect_non_dialogue
        self.confidence_threshold = confidence_threshold
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_subtitle = max_lines_per_subtitle
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        self.model = None
        logger.info(f"WhisperTranscriber initialized with model '{model_name}'")
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is None:
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on device: {self.model.device}")
    
    def transcribe(
        self,
        audio_path: Path,
        output_path: Path,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio to SRT format.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to output SRT file
            language: Language code (None for auto-detection)
            **kwargs: Additional Whisper parameters
            
        Returns:
            TranscriptionResult with transcription metadata
        """
        # Validate input
        is_valid, error_msg = self.validate_input(audio_path)
        if not is_valid:
            return TranscriptionResult(
                success=False,
                error_message=error_msg
            )
        
        try:
            # Load model if not already loaded
            self._load_model()
            
            # Normalize language code for consistency
            if language:
                original_lang = language
                language = LANGUAGE_CODES.get(language.lower().strip(), language)
                if language != original_lang:
                    logger.info(f"Normalized language code: '{original_lang}' -> '{language}'")

            # Prepare transcription options
            transcribe_options = {
                'verbose': kwargs.get('verbose', False),
                'word_timestamps': kwargs.get('word_timestamps', True),
                # Prompt can help with better formatting and sentence boundaries
                'initial_prompt': kwargs.get('initial_prompt', 
                    'Use proper punctuation. Do not divide in the middle of a sentence. '
                    'Create complete sentences with proper endings. '
                    'Start a new segment when the speaker changes.'),
                # Condition on previous text for better context
                'condition_on_previous_text': kwargs.get('condition_on_previous_text', True),
                # Compression ratio threshold - reject segments with poor quality
                'compression_ratio_threshold': kwargs.get('compression_ratio_threshold', 2.4),
                'logprob_threshold': kwargs.get('logprob_threshold', -1.0),
                'no_speech_threshold': kwargs.get('no_speech_threshold', 0.6),
            }
            
            if language:
                transcribe_options['language'] = language
            
            logger.info(f"Starting transcription of '{audio_path}'...")
            
            # Transcribe audio
            logger.info(f"DEBUG: About to call model.transcribe with options: {transcribe_options}")
            try:
                result = self.model.transcribe(str(audio_path), **transcribe_options)
                logger.info(f"DEBUG: Transcribe returned successfully")
            except Exception as e:
                logger.error(f"DEBUG: Transcribe failed with error: {e}", exc_info=True)
                raise
            
            # Extract metadata
            logger.info(f"DEBUG: Extracting metadata from result")
            detected_language = result.get('language', 'unknown')
            segments = result.get('segments', [])
            
            logger.info(f"Transcription complete. Detected language: {detected_language}")
            logger.info(f"Total segments: {len(segments)}")
            
            # Convert to SRT format
            logger.info(f"DEBUG: Converting {len(segments)} segments to SRT")
            try:
                srt_content = self._convert_to_srt(segments)
                logger.info(f"DEBUG: SRT conversion successful, content length: {len(srt_content)}")
            except Exception as e:
                logger.error(f"DEBUG: SRT conversion failed: {e}", exc_info=True)
                raise
            
            # Parse SRT to entries, merge close subtitles, then save
            logger.info(f"DEBUG: Parsing SRT for subtitle merging")
            srt_entries = parse_srt(srt_content)
            original_count = len(srt_entries)
            logger.info(f"Original subtitle count: {original_count}")
            
            # Merge close subtitles to improve downstream processing
            srt_entries = merge_close_subtitles(srt_entries, min_gap_ms=200, max_merged_duration=10.0)
            merged_count = len(srt_entries)
            
            if merged_count < original_count:
                logger.info(f"Merged close subtitles: {original_count} -> {merged_count} entries")
                print(f"ℹ️  Merged {original_count - merged_count} close subtitle pairs ({original_count} → {merged_count} segments)")
            
            # Save merged SRT file
            logger.info(f"DEBUG: Preparing to save SRT to '{output_path}'")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"DEBUG: Output directory created/verified")
            
            save_srt(srt_entries, output_path)
            
            logger.info(f"SRT file saved to '{output_path}'")
            
            # Calculate duration
            duration = segments[-1]['end'] if segments else 0.0
            
            # Release GPU memory after Whisper transcription
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU memory released after transcription")
            except ImportError:
                pass  # PyTorch not available, skip GPU cleanup
            except Exception as e:
                logger.warning(f"Failed to release GPU memory: {str(e)}")
            
            return TranscriptionResult(
                success=True,
                srt_path=output_path,
                detected_language=detected_language,
                language_confidence=None,  # Whisper doesn't provide this directly
                segment_count=len(segments),
                duration=duration,
                model_used=self.model_name,
                metadata={
                    'text': result.get('text', ''),
                    'segments': len(segments)
                }
            )
            
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Release GPU memory even on error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU memory released after transcription error")
            except:
                pass
            
            return TranscriptionResult(
                success=False,
                error_message=error_msg
            )
    
    def _convert_to_srt(self, segments: List[Dict[str, Any]]) -> str:
        """
        Convert Whisper segments to SRT format.
        
        Args:
            segments: List of Whisper segment dictionaries
            
        Returns:
            SRT formatted string
        """
        # Merge segments into complete sentences
        merged_segments = self._merge_segments_by_sentence(segments)
        
        srt_entries = []
        
        for idx, segment in enumerate(merged_segments, 1):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            
            # Check for non-dialogue events
            if self.detect_non_dialogue:
                text = self._process_non_dialogue(text, segment)
            
            # Format timestamps
            start_ts = self._format_timestamp(start_time)
            end_ts = self._format_timestamp(end_time)
            
            # Create SRT entry
            srt_entry = f"{idx}\n{start_ts} --> {end_ts}\n{text}\n"
            srt_entries.append(srt_entry)
        
        return '\n'.join(srt_entries)
    
    def _merge_segments_by_sentence(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge Whisper segments into complete sentences based on punctuation.
        
        Args:
            segments: List of Whisper segment dictionaries
            
        Returns:
            List of merged segments with complete sentences
        """
        if not segments:
            return []
        
        merged = []
        current_segment = None
        
        for segment in segments:
            text = segment['text'].strip()
            
            if not text:
                continue
            
            # Start a new segment if needed
            if current_segment is None:
                current_segment = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': text,
                    'words': segment.get('words', [])
                }
            else:
                # Check if we should merge or start new segment
                duration = current_segment['end'] - current_segment['start']
                new_duration = segment['end'] - current_segment['start']
                current_text = current_segment['text']
                combined_text = current_text + ' ' + text
                
                # Enhanced splitting conditions:
                # 1. Current segment ends with strong punctuation (period, exclamation, question mark)
                ends_with_strong_punct = current_text.rstrip().endswith(('.', '!', '?', '。', '！', '？'))
                
                # 2. Current segment ends with medium punctuation (semicolon, colon) AND is reasonably long
                ends_with_medium_punct = (
                    current_text.rstrip().endswith((':', ';', '；', '：')) and 
                    len(current_text) > 30
                )
                
                # 3. Current segment ends with comma AND is getting long
                ends_with_comma = (
                    current_text.rstrip().endswith((',', '、', '，')) and 
                    len(current_text) > 50
                )
                
                # 4. Combined duration exceeds max (typically 7 seconds)
                duration_exceeded = new_duration >= self.max_duration
                
                # 5. Combined text exceeds max character limit
                length_exceeded = len(combined_text) > self.max_chars_per_line * self.max_lines_per_subtitle
                
                # 6. Current segment duration already reasonable and has natural break
                natural_break = (
                    duration >= 3.0 and 
                    (ends_with_medium_punct or ends_with_comma)
                )
                
                should_split = (
                    ends_with_strong_punct or
                    ends_with_medium_punct or
                    ends_with_comma or
                    duration_exceeded or
                    length_exceeded or
                    natural_break
                )
                
                if should_split:
                    # Save current segment and start new one
                    merged.append(current_segment)
                    current_segment = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text,
                        'words': segment.get('words', [])
                    }
                else:
                    # Merge with current segment
                    current_segment['end'] = segment['end']
                    current_segment['text'] = combined_text
                    if 'words' in segment:
                        current_segment['words'].extend(segment.get('words', []))
        
        # Don't forget the last segment
        if current_segment is not None:
            merged.append(current_segment)
        
        # Post-process: split segments that are still too long
        final_segments = []
        for segment in merged:
            seg_duration = segment['end'] - segment['start']
            seg_length = len(segment['text'])
            
            # If segment is too long (duration OR length), split it further
            if seg_duration > self.max_duration or seg_length > self.max_chars_per_line * self.max_lines_per_subtitle:
                final_segments.extend(self._split_long_segment(segment))
            else:
                final_segments.append(segment)
        
        return final_segments
    
    def _split_long_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a long segment into smaller segments by punctuation boundaries.
        
        Args:
            segment: Segment to split
            
        Returns:
            List of smaller segments
        """
        text = segment['text']
        max_length = self.max_chars_per_line * self.max_lines_per_subtitle
        
        # Try multiple splitting strategies in order of preference
        
        # Strategy 1: Split by strong punctuation (period, exclamation, question mark)
        strong_pattern = r'([.!?。！？]+[\s"\'\u201c\u201d]*)'
        parts = re.split(strong_pattern, text)
        
        # If that doesn't help, try medium punctuation (semicolon, colon)
        if len(parts) <= 2 and len(text) > max_length:
            medium_pattern = r'([;:；：]+[\s"\'\u201c\u201d]*)'
            parts = re.split(medium_pattern, text)
        
        # If still too long, try splitting by comma
        if len(parts) <= 2 and len(text) > max_length:
            comma_pattern = r'([,、，]+[\s"\'\u201c\u201d]*)'
            parts = re.split(comma_pattern, text)
        
        # If still no good splits, split by word count
        if len(parts) <= 2 and len(text) > max_length:
            return self._split_by_word_count(segment, max_length)
        
        segments = []
        current_text = ''
        start_time = segment['start']
        duration = segment['end'] - segment['start']
        text_length = len(text)
        
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            punct = parts[i + 1] if i + 1 < len(parts) else ''
            full_sentence = (sentence + punct).strip()
            
            if not full_sentence:
                continue
            
            # If adding this sentence exceeds max length or max duration, save current and start new
            combined = (current_text + ' ' + full_sentence).strip()
            estimated_duration = (len(combined) / text_length) * duration if text_length > 0 else 0
            
            if current_text and (len(combined) > max_length or estimated_duration > self.max_duration):
                # Calculate end time based on text proportion
                text_ratio = len(current_text) / text_length if text_length > 0 else 0
                end_time = start_time + (duration * text_ratio)
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': current_text,
                    'words': []
                })
                
                start_time = end_time
                current_text = full_sentence
            else:
                current_text = combined
        
        # Add the last segment
        if current_text:
            segments.append({
                'start': start_time,
                'end': segment['end'],
                'text': current_text,
                'words': []
            })
        
        return segments if segments else [segment]
    
    def _split_by_word_count(self, segment: Dict[str, Any], max_length: int) -> List[Dict[str, Any]]:
        """
        Split segment by word count when no punctuation boundaries are available.
        
        Args:
            segment: Segment to split
            max_length: Maximum character length
            
        Returns:
            List of smaller segments
        """
        text = segment['text']
        words = text.split()
        
        segments = []
        current_words = []
        start_time = segment['start']
        duration = segment['end'] - segment['start']
        total_words = len(words)
        
        for i, word in enumerate(words):
            current_words.append(word)
            current_text = ' '.join(current_words)
            
            # Split if we exceed max length or if we're at a reasonable break point
            if len(current_text) >= max_length or (
                len(current_text) > max_length * 0.7 and i < len(words) - 1
            ):
                # Calculate time based on word proportion
                word_ratio = (i + 1) / total_words if total_words > 0 else 0
                end_time = start_time + (duration * word_ratio)
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': current_text,
                    'words': []
                })
                
                start_time = end_time
                current_words = []
        
        # Add remaining words
        if current_words:
            segments.append({
                'start': start_time,
                'end': segment['end'],
                'text': ' '.join(current_words),
                'words': []
            })
        
        return segments if segments else [segment]
    
    def _process_non_dialogue(self, text: str, segment: Dict[str, Any]) -> str:
        """
        Detect and mark non-dialogue events.
        
        Args:
            text: Transcribed text
            segment: Whisper segment data
            
        Returns:
            Processed text with non-dialogue markers
        """
        # Check if text is empty or very short
        if not text or len(text.strip()) < 2:
            return '[[ silence ]]'
        
        # Check for Whisper's built-in markers (like [MUSIC], [LAUGHTER])
        whisper_marker_pattern = r'\[([A-Z]+)\]'
        matches = re.findall(whisper_marker_pattern, text)
        
        if matches:
            # Convert Whisper markers to our format
            marker = matches[0].lower()
            if marker in self.NON_DIALOGUE_MARKERS:
                return self.NON_DIALOGUE_MARKERS[marker]
            else:
                return f'[[ {marker.lower()} ]]'
        
        # Check for known non-dialogue keywords in text
        text_lower = text.lower()
        for keyword, marker in self.NON_DIALOGUE_MARKERS.items():
            if keyword in text_lower and len(text) < 20:
                # If text is short and contains keyword, likely non-dialogue
                return marker
        
        # Check confidence/probability if available
        if 'no_speech_prob' in segment:
            no_speech_prob = segment['no_speech_prob']
            if no_speech_prob > (1 - self.confidence_threshold):
                return '[[ inaudible ]]'
        
        # Check average log probability
        if 'avg_logprob' in segment:
            avg_logprob = segment['avg_logprob']
            # Very low probability might indicate non-speech
            if avg_logprob < -1.0:
                # Still return the text but might want to flag it
                pass
        
        return text
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format timestamp in SRT format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
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
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported Whisper models.
        
        Returns:
            List of model names
        """
        return ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'detect_non_dialogue': self.detect_non_dialogue,
            'confidence_threshold': self.confidence_threshold,
            'loaded': self.model is not None
        }
