"""Audio clipping processor for extracting clips based on SRT timestamps"""

import re
import subprocess
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .base import ProcessorResult


@dataclass
class AudioClip:
    """Information about a single audio clip"""
    clip_number: int
    filename: str
    start_time: str  # SRT format: HH:MM:SS,mmm
    end_time: str
    start_seconds: float
    end_seconds: float
    duration: float
    text: str
    output_path: Path


@dataclass
class ClippingResult(ProcessorResult):
    """Result of audio clipping operation"""
    clips: List[AudioClip] = field(default_factory=list)
    total_clips: int = 0
    source_audio: Optional[Path] = None
    source_srt: Optional[Path] = None
    output_dir: Optional[Path] = None


class AudioClipper(ABC):
    """Abstract base class for audio clipping implementations"""
    
    @abstractmethod
    def clip_audio(
        self,
        vocal_audio_path: Path,
        srt_path: Path,
        output_dir: Optional[Path] = None,
        output_format: str = "mp3",
        bitrate: str = "192k",
        **kwargs
    ) -> ClippingResult:
        """
        Extract audio clips based on SRT timestamps.
        
        Args:
            vocal_audio_path: Path to vocal audio file (from Step 2)
            srt_path: Path to original SRT file (from Step 3)
            output_dir: Directory for output clips (default: ./output/step5/)
            output_format: Output audio format (default: mp3)
            bitrate: Audio bitrate for encoding (default: 192k)
            **kwargs: Implementation-specific parameters
            
        Returns:
            ClippingResult containing list of clips and metadata
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If SRT format is invalid or timestamps are malformed
            RuntimeError: If clipping process fails
        """
        pass
    
    @abstractmethod
    def parse_srt(self, srt_path: Path) -> List[Dict[str, Any]]:
        """
        Parse SRT file and extract subtitle entries.
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            List of dictionaries containing subtitle information
        """
        pass
    
    @abstractmethod
    def srt_time_to_seconds(self, srt_time: str) -> float:
        """
        Convert SRT timestamp to seconds.
        
        Args:
            srt_time: Timestamp in SRT format (HH:MM:SS,mmm)
            
        Returns:
            Time in seconds (float)
        """
        pass


class FFmpegAudioClipper(AudioClipper):
    """FFmpeg-based audio clipping implementation"""
    
    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        default_bitrate: str = "192k",
        default_format: str = "mp3"
    ):
        """
        Initialize FFmpeg audio clipper.
        
        Args:
            ffmpeg_path: Path to ffmpeg executable (default: "ffmpeg" from PATH)
            default_bitrate: Default audio bitrate (default: 192k)
            default_format: Default output format (default: mp3)
        """
        self.ffmpeg_path = ffmpeg_path
        self.default_bitrate = default_bitrate
        self.default_format = default_format
        self._validate_ffmpeg()
    
    def _validate_ffmpeg(self):
        """Verify FFmpeg is available and functional"""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg validation failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"FFmpeg not found at '{self.ffmpeg_path}'. "
                "Please install FFmpeg or provide correct path."
            )
        except Exception as e:
            raise RuntimeError(f"FFmpeg validation error: {str(e)}")
    
    def clip_audio(
        self,
        vocal_audio_path: Path,
        srt_path: Path,
        output_dir: Optional[Path] = None,
        output_format: str = "mp3",
        bitrate: str = "192k",
        **kwargs
    ) -> ClippingResult:
        """Implementation of audio clipping using FFmpeg"""
        
        # Validate inputs
        if not vocal_audio_path.exists():
            return ClippingResult(
                success=False,
                error_message=f"Vocal audio file not found: {vocal_audio_path}"
            )
        
        if not srt_path.exists():
            return ClippingResult(
                success=False,
                error_message=f"SRT file not found: {srt_path}"
            )
        
        # Set output directory
        if output_dir is None:
            output_dir = Path("output/step5")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse SRT file
        try:
            srt_entries = self.parse_srt(srt_path)
        except Exception as e:
            return ClippingResult(
                success=False,
                error_message=f"Failed to parse SRT file: {str(e)}"
            )
        
        if not srt_entries:
            return ClippingResult(
                success=False,
                error_message="SRT file is empty or contains no valid entries"
            )
        
        # Extract clips
        clips = []
        failed_clips = []
        total_entries = len(srt_entries)
        
        print(f"Extracting {total_entries} audio clips...")
        
        for idx, entry in enumerate(srt_entries, 1):
            clip_number = entry['number']
            start_time = entry['start']
            end_time = entry['end']
            text = entry['text']
            
            # Show progress every 10 clips or at the end
            if idx % 10 == 0 or idx == total_entries:
                print(f"Progress: {idx}/{total_entries} clips ({idx*100//total_entries}%)")
            
            try:
                start_seconds = self.srt_time_to_seconds(start_time)
                end_seconds = self.srt_time_to_seconds(end_time)
                duration = end_seconds - start_seconds
                
                # Validate duration
                if duration <= 0:
                    print(f"Warning: Clip {clip_number} has zero or negative duration, skipping")
                    failed_clips.append(clip_number)
                    continue
                
                # Generate output filename
                filename = f"original_clip_{clip_number}.{output_format}"
                output_path = output_dir / filename
                
                # Extract clip using FFmpeg
                success = self._extract_clip(
                    input_audio=vocal_audio_path,
                    output_path=output_path,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    output_format=output_format,
                    bitrate=bitrate
                )
                
                if success:
                    clip = AudioClip(
                        clip_number=clip_number,
                        filename=filename,
                        start_time=start_time,
                        end_time=end_time,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        duration=duration,
                        text=text,
                        output_path=output_path
                    )
                    clips.append(clip)
                else:
                    failed_clips.append(clip_number)
                    
            except Exception as e:
                print(f"Error processing clip {clip_number}: {str(e)}")
                failed_clips.append(clip_number)
        
        # Save metadata
        metadata = {
            'source_audio': str(vocal_audio_path),
            'source_srt': str(srt_path),
            'total_clips': len(clips),
            'failed_clips': failed_clips,
            'clips': [
                {
                    'clip_number': clip.clip_number,
                    'filename': clip.filename,
                    'start_time': clip.start_time,
                    'end_time': clip.end_time,
                    'start_seconds': clip.start_seconds,
                    'end_seconds': clip.end_seconds,
                    'duration': clip.duration,
                    'text': clip.text
                }
                for clip in clips
            ]
        }
        
        # Save metadata file
        metadata_path = output_dir / "clipping_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\nAudio clipping complete:")
        print(f"  Successfully extracted: {len(clips)} clips")
        if failed_clips:
            print(f"  Failed clips: {len(failed_clips)}")
        
        # Determine success
        success = len(clips) > 0
        error_message = None
        if not success:
            error_message = "No clips were successfully extracted"
        elif failed_clips:
            error_message = f"Partial success: {len(failed_clips)} clips failed"
        
        return ClippingResult(
            success=success,
            error_message=error_message,
            clips=clips,
            total_clips=len(clips),
            source_audio=vocal_audio_path,
            source_srt=srt_path,
            output_dir=output_dir,
            metadata=metadata
        )
    
    def _extract_clip(
        self,
        input_audio: Path,
        output_path: Path,
        start_seconds: float,
        end_seconds: float,
        output_format: str,
        bitrate: str
    ) -> bool:
        """
        Extract a single audio clip using FFmpeg.
        
        Args:
            input_audio: Path to source audio file
            output_path: Path for output clip
            start_seconds: Start time in seconds
            end_seconds: End time in seconds
            output_format: Output audio format
            bitrate: Audio bitrate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                "-i", str(input_audio),
                "-ss", str(start_seconds),
                "-to", str(end_seconds),
                "-c:a", "libmp3lame" if output_format == "mp3" else "copy",
                "-b:a", bitrate,
                "-avoid_negative_ts", "make_zero",
                "-y",  # Overwrite output file
                str(output_path)
            ]
            
            # Execute FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"FFmpeg error for {output_path.name}: {result.stderr}")
                return False
            
            # Verify output file was created
            if not output_path.exists() or output_path.stat().st_size == 0:
                print(f"Output file not created or empty: {output_path}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"FFmpeg timeout for {output_path.name}")
            return False
        except Exception as e:
            print(f"Exception during clip extraction: {str(e)}")
            return False
    
    def parse_srt(self, srt_path: Path) -> List[Dict[str, Any]]:
        """
        Parse SRT file and extract subtitle entries.
        
        Args:
            srt_path: Path to SRT file
            
        Returns:
            List of dictionaries containing subtitle information
        """
        entries = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(srt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
        
        # Split into blocks (entries separated by blank lines)
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            if not block.strip():
                continue
            
            lines = block.strip().split('\n')
            
            if len(lines) < 3:
                continue
            
            try:
                # Parse entry number
                number = int(lines[0].strip())
                
                # Parse timestamp line
                timestamp_line = lines[1].strip()
                timestamp_match = re.match(
                    r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
                    timestamp_line
                )
                
                if not timestamp_match:
                    print(f"Warning: Invalid timestamp format in entry {number}")
                    continue
                
                start_time = timestamp_match.group(1)
                end_time = timestamp_match.group(2)
                
                # Parse text (remaining lines)
                text = '\n'.join(lines[2:]).strip()
                
                entries.append({
                    'number': number,
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing SRT block: {str(e)}")
                continue
        
        return entries
    
    def srt_time_to_seconds(self, srt_time: str) -> float:
        """
        Convert SRT timestamp to seconds.
        
        Args:
            srt_time: Timestamp in SRT format (HH:MM:SS,mmm)
            
        Returns:
            Time in seconds (float)
            
        Example:
            '00:01:23,456' -> 83.456
        """
        # Parse HH:MM:SS,mmm format
        match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', srt_time)
        
        if not match:
            raise ValueError(f"Invalid SRT timestamp format: {srt_time}")
        
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        milliseconds = int(match.group(4))
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        
        return total_seconds
