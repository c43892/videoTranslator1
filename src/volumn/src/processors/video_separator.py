"""FFmpeg-based video/audio separator implementation"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .base import VideoAudioSeparator, SeparationResult

logger = logging.getLogger(__name__)


class FFmpegVideoAudioSeparator(VideoAudioSeparator):
    """FFmpeg-based implementation of video/audio separation"""
    
    def __init__(self):
        """Initialize the FFmpeg separator"""
        self._check_ffmpeg_installed()
    
    def _check_ffmpeg_installed(self) -> None:
        """Check if FFmpeg is installed and accessible"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg is not properly installed")
            logger.info("FFmpeg is available")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg is not installed. Please install FFmpeg to use this separator."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg check timed out")
    
    def validate_input(self, video_path: Path) -> tuple[bool, Optional[str]]:
        """Validate input video file"""
        if not video_path.exists():
            return False, f"Video file not found: {video_path}"
        
        if not video_path.is_file():
            return False, f"Path is not a file: {video_path}"
        
        if video_path.stat().st_size == 0:
            return False, f"Video file is empty: {video_path}"
        
        # Check if file can be read by FFmpeg
        try:
            info = self.get_stream_info(video_path)
            
            if not info.get('video_streams'):
                return False, "No video stream found in file"
            
            if not info.get('audio_streams'):
                return False, "No audio stream found in file"
            
            return True, None
            
        except Exception as e:
            return False, f"Failed to read video file: {str(e)}"
    
    def get_stream_info(self, video_path: Path) -> Dict[str, Any]:
        """Get detailed stream information using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            # Extract relevant information
            video_streams = [
                s for s in data.get('streams', [])
                if s.get('codec_type') == 'video'
            ]
            
            audio_streams = [
                s for s in data.get('streams', [])
                if s.get('codec_type') == 'audio'
            ]
            
            format_info = data.get('format', {})
            
            return {
                'video_streams': video_streams,
                'audio_streams': audio_streams,
                'duration': float(format_info.get('duration', 0)),
                'format': format_info.get('format_name', 'unknown'),
                'size': int(format_info.get('size', 0)),
                'bit_rate': int(format_info.get('bit_rate', 0))
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffprobe timed out while reading video file")
        except json.JSONDecodeError:
            raise RuntimeError("Failed to parse ffprobe output")
        except Exception as e:
            raise RuntimeError(f"Error getting stream info: {str(e)}")
    
    def separate(
        self,
        input_video_path: Path,
        output_dir: Optional[Path] = None,
        video_stream_index: int = 0,
        audio_stream_index: int = 0,
        **kwargs
    ) -> SeparationResult:
        """
        Separate video and audio tracks using FFmpeg
        
        Additional kwargs:
            video_filename: Custom video output filename (default: video_only.mp4)
            audio_filename: Custom audio output filename (default: audio_full.wav)
            overwrite: Overwrite existing files (default: False)
        """
        try:
            # Validate input
            is_valid, error = self.validate_input(input_video_path)
            if not is_valid:
                return SeparationResult(
                    success=False,
                    error_message=error
                )
            
            # Set output directory
            if output_dir is None:
                output_dir = input_video_path.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get output filenames
            video_filename = kwargs.get('video_filename', 'video_only.mp4')
            audio_filename = kwargs.get('audio_filename', 'audio_full.wav')
            overwrite = kwargs.get('overwrite', False)
            
            video_output_path = output_dir / video_filename
            audio_output_path = output_dir / audio_filename
            
            # Check if files exist
            if not overwrite:
                if video_output_path.exists():
                    return SeparationResult(
                        success=False,
                        error_message=f"Video output file already exists: {video_output_path}"
                    )
                if audio_output_path.exists():
                    return SeparationResult(
                        success=False,
                        error_message=f"Audio output file already exists: {audio_output_path}"
                    )
            
            # Get stream info for metadata
            stream_info = self.get_stream_info(input_video_path)
            
            logger.info(f"Separating video: {input_video_path}")
            logger.info(f"Duration: {stream_info['duration']}s")
            logger.info(f"Format: {stream_info['format']}")
            
            # Extract video without audio (stream copy for speed)
            logger.info(f"Extracting video to: {video_output_path}")
            video_result = self._extract_video(
                input_video_path,
                video_output_path,
                video_stream_index,
                overwrite
            )
            
            if not video_result:
                return SeparationResult(
                    success=False,
                    error_message="Failed to extract video stream"
                )
            
            # Extract audio to WAV (for consistency in downstream processing)
            logger.info(f"Extracting audio to: {audio_output_path}")
            audio_result = self._extract_audio(
                input_video_path,
                audio_output_path,
                audio_stream_index,
                overwrite
            )
            
            if not audio_result:
                return SeparationResult(
                    success=False,
                    error_message="Failed to extract audio stream"
                )
            
            # Verify outputs
            if not video_output_path.exists() or video_output_path.stat().st_size == 0:
                return SeparationResult(
                    success=False,
                    error_message="Video output file is empty or missing"
                )
            
            if not audio_output_path.exists() or audio_output_path.stat().st_size == 0:
                return SeparationResult(
                    success=False,
                    error_message="Audio output file is empty or missing"
                )
            
            logger.info("Separation completed successfully")
            
            return SeparationResult(
                success=True,
                video_path=video_output_path,
                audio_path=audio_output_path,
                metadata={
                    'input_file': str(input_video_path),
                    'duration': stream_info['duration'],
                    'format': stream_info['format'],
                    'video_streams_count': len(stream_info['video_streams']),
                    'audio_streams_count': len(stream_info['audio_streams'])
                }
            )
            
        except Exception as e:
            logger.error(f"Separation failed: {str(e)}", exc_info=True)
            return SeparationResult(
                success=False,
                error_message=f"Separation failed: {str(e)}"
            )
    
    def _extract_video(
        self,
        input_path: Path,
        output_path: Path,
        stream_index: int,
        overwrite: bool
    ) -> bool:
        """Extract video stream without audio"""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-map', f'0:v:{stream_index}',  # Select video stream
                '-c:v', 'copy',  # Copy video codec (no re-encoding)
                '-an',  # No audio
            ]
            
            if overwrite:
                cmd.append('-y')
            else:
                cmd.append('-n')
            
            cmd.append(str(output_path))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg video extraction error: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Video extraction timed out")
            return False
        except Exception as e:
            logger.error(f"Video extraction failed: {str(e)}")
            return False
    
    def _extract_audio(
        self,
        input_path: Path,
        output_path: Path,
        stream_index: int,
        overwrite: bool
    ) -> bool:
        """Extract audio stream to WAV format"""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-map', f'0:a:{stream_index}',  # Select audio stream
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # Convert to WAV (16-bit PCM)
                '-ar', '44100',  # Sample rate 44.1kHz
                '-ac', '2',  # Stereo (2 channels)
            ]
            
            if overwrite:
                cmd.append('-y')
            else:
                cmd.append('-n')
            
            cmd.append(str(output_path))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg audio extraction error: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            return False
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            return False
