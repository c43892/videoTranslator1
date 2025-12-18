"""Base classes for video processors"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field



LANGUAGE_CODES = {
    "afrikaans": "af",
    "arabic": "ar",
    "armenian": "hy",
    "azerbaijani": "az",
    "belarusian": "be",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese": "zh",
    "mandarin": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "german": "de",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "kannada": "kn",
    "kazakh": "kk",
    "korean": "ko",
    "latvian": "lv",
    "lithuanian": "lt",
    "macedonian": "mk",
    "malay": "ms",
    "marathi": "mr",
    "maori": "mi",
    "nepali": "ne",
    "norwegian": "no",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tamil": "ta",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "vietnamese": "vi",
    "welsh": "cy",
}

@dataclass
class ProcessorResult:
    """Base result class for processor operations"""
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeparationResult(ProcessorResult):
    """Result of video/audio separation operation"""
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None


@dataclass
class TranscriptionResult(ProcessorResult):
    """Result of transcription operation"""
    srt_path: Optional[Path] = None
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    segment_count: int = 0
    duration: Optional[float] = None
    model_used: Optional[str] = None


class VideoAudioSeparator(ABC):
    """Abstract base class for video/audio separation implementations"""
    
    @abstractmethod
    def separate(
        self,
        input_video_path: Path,
        output_dir: Optional[Path] = None,
        video_stream_index: int = 0,
        audio_stream_index: int = 0,
        **kwargs
    ) -> SeparationResult:
        """
        Separate video and audio tracks from input video file.
        
        Args:
            input_video_path: Path to input video file
            output_dir: Directory for output files (default: same as input)
            video_stream_index: Index of video stream to extract (default: 0)
            audio_stream_index: Index of audio stream to extract (default: 0)
            **kwargs: Implementation-specific parameters
            
        Returns:
            SeparationResult containing paths to separated files and metadata
            
        Raises:
            FileNotFoundError: If input video doesn't exist
            ValueError: If video format is unsupported or file is corrupted
            RuntimeError: If separation process fails
        """
        pass
    
    @abstractmethod
    def get_stream_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get information about streams in the video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing stream information:
            - video_streams: List of video stream metadata
            - audio_streams: List of audio stream metadata
            - duration: Total duration in seconds
            - format: Container format
        """
        pass
    
    @abstractmethod
    def validate_input(self, video_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate that input video file is suitable for processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class Transcriber(ABC):
    """Abstract base class for audio transcription implementations"""
    
    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        output_path: Path,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio to SRT format with timestamps.
        
        Args:
            audio_path: Path to input audio file (vocals)
            output_path: Path to output SRT file
            language: Language code (None for auto-detection)
            **kwargs: Implementation-specific parameters
            
        Returns:
            TranscriptionResult containing SRT path, detected language, and metadata
            
        Raises:
            FileNotFoundError: If input audio doesn't exist
            ValueError: If audio format is unsupported
            RuntimeError: If transcription process fails
        """
        pass
    
    @abstractmethod
    def validate_input(self, audio_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate that input audio file is suitable for transcription.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
