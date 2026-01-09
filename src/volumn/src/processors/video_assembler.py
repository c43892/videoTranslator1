"""
Step 8: Final Video Assembly
Combines video track with translated audio track to create the final output video.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


logger = logging.getLogger(__name__)


def get_video_info(file_path: str) -> Dict[str, Any]:
    """
    Get video file information using ffprobe.
    
    Args:
        file_path: Path to video file
        
    Returns:
        Dictionary with video info (duration, resolution, fps, codec, bitrate)
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=duration,width,height,r_frame_rate,codec_name,bit_rate',
        '-show_entries', 'format=duration',
        '-of', 'json',
        file_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    # Extract video stream info
    video_stream = data.get('streams', [{}])[0]
    format_info = data.get('format', {})
    
    # Parse frame rate (usually in format like "30/1")
    fps_str = video_stream.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)
    
    # Get duration (prefer stream duration, fallback to format duration)
    duration = float(video_stream.get('duration', format_info.get('duration', 0)))
    
    return {
        'duration': duration,
        'width': int(video_stream.get('width', 0)),
        'height': int(video_stream.get('height', 0)),
        'fps': round(fps, 3),
        'codec': video_stream.get('codec_name', 'unknown'),
        'bitrate': video_stream.get('bit_rate', 'unknown'),
        'resolution': f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}"
    }


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Get audio file information using ffprobe.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio info (duration, sample_rate, channels, bitrate)
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=duration,sample_rate,channels,bit_rate',
        '-show_entries', 'format=duration',
        '-of', 'json',
        file_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    # Extract audio stream info
    audio_stream = data.get('streams', [{}])[0]
    format_info = data.get('format', {})
    
    # Get duration (prefer stream duration, fallback to format duration)
    duration = float(audio_stream.get('duration', format_info.get('duration', 0)))
    
    return {
        'duration': duration,
        'sample_rate': int(audio_stream.get('sample_rate', 44100)),
        'channels': int(audio_stream.get('channels', 2)),
        'bitrate': audio_stream.get('bit_rate', 'unknown')
    }


@dataclass
class AssemblyMetadata:
    """Metadata for video assembly operation"""
    video_source: str
    audio_source: str
    output_file: str
    video_properties: Dict[str, Any]
    audio_properties: Dict[str, Any]
    duration_match: bool
    duration_difference: float
    adjustments: list
    warnings: list
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class VideoAssembler:
    """
    Video assembler processor for Step 8.
    Combines video track with translated audio track using FFmpeg.
    """
    
    def __init__(
        self,
        output_format: str = "mp4",
        video_codec: str = "copy",
        audio_codec: str = "aac",
        audio_bitrate: str = "192k",
        video_bitrate: Optional[str] = None,
        embed_subtitles: bool = True,
        duration_tolerance: float = 0.1,
        auto_adjust_duration: bool = True
    ):
        """
        Initialize VideoAssembler.
        
        Args:
            output_format: Output video format (mp4, mkv, avi, mov)
            video_codec: Video codec (copy=no re-encode, h264, h265)
            audio_codec: Audio codec (aac, mp3, opus)
            audio_bitrate: Audio bitrate (e.g., "192k")
            video_bitrate: Video bitrate (None=auto, or e.g., "5000k")
            duration_tolerance: Acceptable duration difference in seconds
            auto_adjust_duration: Automatically pad/trim audio to match video
        """
        self.output_format = output_format
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.video_bitrate = video_bitrate
        self.video_bitrate = video_bitrate
        self.embed_subtitles = embed_subtitles
        self.duration_tolerance = duration_tolerance
        self.auto_adjust_duration = auto_adjust_duration
        self.warnings = []
        self.adjustments = []
        
        logger.info(f"VideoAssembler initialized: format={output_format}, video_codec={video_codec}, audio_codec={audio_codec}")
    
    def _get_language_code(self, language: str) -> str:
        """
        Convert language name to ISO 639-2 code.
        
        Args:
            language: Language name (e.g., "English", "Chinese")
            
        Returns:
            ISO 639-2 language code (e.g., "eng", "chi")
        """
        # Common language mappings
        language_codes = {
            'english': 'eng',
            'chinese': 'chi',
            'mandarin': 'chi',
            'japanese': 'jpn',
            'korean': 'kor',
            'spanish': 'spa',
            'french': 'fra',
            'german': 'ger',
            'italian': 'ita',
            'portuguese': 'por',
            'russian': 'rus',
            'arabic': 'ara'
        }
        
        lang_lower = language.lower()
        return language_codes.get(lang_lower, 'und')  # 'und' = undefined
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for video assembly.
        
        Args:
            input_data: {
                'video_file': path to video_only.[format],
                'audio_file': path to audio_translated_full.mp3,
                'output_dir': path for output files,
                'output_format': (optional) override output format,
                'video_bitrate': (optional) override video bitrate,
                'output_format': (optional) override output format,
                'video_bitrate': (optional) override video bitrate,
                'audio_bitrate': (optional) override audio bitrate,
                'srt_file': (optional) path to SRT file to embed
            }
            
        Returns:
            {
                'video_translated': path to output video,
                'metadata_file': path to metadata JSON,
                'metadata': assembly metadata dict
            }
        """
        logger.info("Starting video assembly process...")
        
        # Extract input parameters
        video_file = Path(input_data['video_file'])
        audio_file = Path(input_data['audio_file'])
        srt_file = Path(input_data.get('srt_file', '')) if input_data.get('srt_file') else None
        output_dir = Path(input_data['output_dir'])
        
        # Override settings if provided
        output_format = input_data.get('output_format', self.output_format)
        audio_bitrate = input_data.get('audio_bitrate', self.audio_bitrate)
        video_bitrate = input_data.get('video_bitrate', self.video_bitrate)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset warnings and adjustments
        self.warnings = []
        self.adjustments = []
        
        # Validate input files
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Get file information
        logger.info(f"Analyzing video file: {video_file}")
        video_info = get_video_info(str(video_file))
        logger.info(f"Video: {video_info['resolution']}, {video_info['fps']}fps, {video_info['duration']:.2f}s, codec={video_info['codec']}")
        
        logger.info(f"Analyzing audio file: {audio_file}")
        audio_info = get_audio_info(str(audio_file))
        logger.info(f"Audio: {audio_info['duration']:.2f}s, {audio_info['sample_rate']}Hz, {audio_info['channels']} channels")
        
        # Check duration match
        video_duration = video_info['duration']
        audio_duration = audio_info['duration']
        duration_diff = abs(video_duration - audio_duration)
        duration_match = duration_diff <= self.duration_tolerance
        
        if not duration_match:
            warning = f"Duration mismatch: video={video_duration:.3f}s, audio={audio_duration:.3f}s, diff={duration_diff:.3f}s"
            logger.warning(warning)
            self.warnings.append(warning)
        
        # Prepare audio file (adjust duration if needed)
        final_audio_file = audio_file
        
        if not duration_match and self.auto_adjust_duration:
            logger.info("Adjusting audio duration to match video...")
            adjusted_audio = output_dir / f"audio_adjusted.mp3"
            self.adjust_audio_duration(
                str(audio_file),
                video_duration,
                str(adjusted_audio),
                audio_duration
            )
            final_audio_file = adjusted_audio
        
        # Combine video and audio
        output_file = output_dir / f"video_translated.{output_format}"
        logger.info(f"Combining video and audio into: {output_file}")
        
        # Get subtitle language from input_data
        subtitle_language = input_data.get('subtitle_language', 'English')
        
        self.combine_video_audio(
            str(video_file),
            str(final_audio_file),
            str(output_file),
            video_bitrate=video_bitrate,
            audio_bitrate=audio_bitrate,
            srt_path=str(srt_file) if srt_file and self.embed_subtitles else None,
            subtitle_language=subtitle_language if srt_file and self.embed_subtitles else None
        )
        
        # Generate metadata
        metadata = AssemblyMetadata(
            video_source=str(video_file),
            audio_source=str(audio_file),
            output_file=str(output_file),
            video_properties=video_info,
            audio_properties=audio_info,
            duration_match=duration_match,
            duration_difference=round(duration_diff, 3),
            adjustments=self.adjustments,
            warnings=self.warnings,
            timestamp=datetime.now().isoformat()
        )
        
        # Save metadata
        metadata_file = output_dir / "assembly_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Assembly complete! Output: {output_file}")
        logger.info(f"Metadata saved to: {metadata_file}")
        
        return {
            'video_translated': str(output_file),
            'metadata_file': str(metadata_file),
            'metadata': metadata.to_dict()
        }
    
    def adjust_audio_duration(
        self,
        audio_path: str,
        target_duration: float,
        output_path: str,
        current_duration: float
    ):
        """
        Adjust audio duration to match target by padding or trimming.
        
        Args:
            audio_path: Input audio file
            target_duration: Target duration in seconds
            output_path: Output audio file
            current_duration: Current audio duration
        """
        if current_duration < target_duration:
            # Pad with silence
            padding = target_duration - current_duration
            logger.info(f"Padding audio with {padding:.3f}s of silence")
            
            cmd = [
                'ffmpeg',
                '-i', audio_path,
                '-af', f'apad=whole_dur={target_duration}',
                '-y',
                output_path
            ]
            
            self.adjustments.append(f"Padded audio with {padding:.3f}s silence")
            
        else:
            # Trim audio
            logger.info(f"Trimming audio to {target_duration:.3f}s")
            
            cmd = [
                'ffmpeg',
                '-i', audio_path,
                '-t', str(target_duration),
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            trim_amount = current_duration - target_duration
            self.adjustments.append(f"Trimmed {trim_amount:.3f}s from audio")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug("Audio duration adjusted successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to adjust audio duration: {e.stderr}")
            raise RuntimeError(f"Audio duration adjustment failed: {e.stderr}")
    
    def combine_video_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        video_bitrate: Optional[str] = None,
        audio_bitrate: str = "192k",
        srt_path: Optional[str] = None,
        subtitle_language: Optional[str] = None
    ):
        """
        Combine video and audio tracks using FFmpeg.
        
        Args:
            video_path: Input video file
            audio_path: Input audio file
            output_path: Output video file
            video_bitrate: Video bitrate (None=auto/copy)
            audio_bitrate: Audio bitrate
            srt_path: Optional path to SRT file to embed
        """
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path
        ]
        
        # Add subtitle input if provided
        if srt_path:
            cmd.extend(['-i', srt_path])
            
        cmd.extend([
            '-map', '0:v:0',  # Map video from first input
            '-map', '1:a:0',  # Map audio from second input
        ])
        
        # Map subtitle if provided
        if srt_path:
            cmd.extend(['-map', '2:s:0'])
            
        cmd.extend([
            '-c:v', self.video_codec,  # Video codec (copy or h264/h265)
        ])
        
        # Add video bitrate if specified and not using copy
        if video_bitrate and self.video_codec != 'copy':
            cmd.extend(['-b:v', video_bitrate])
        
        # Add audio encoding
        cmd.extend([
            '-c:a', self.audio_codec,
            '-b:a', audio_bitrate,
        ])
        
        # Add subtitle encoding if provided
        if srt_path:
            # mov_text is the subtitle format for MP4 container
            if self.output_format == 'mp4':
                cmd.extend(['-c:s', 'mov_text'])
            else:
                # Default to copy for other containers (mkv supports srt directly)
                cmd.extend(['-c:s', 'copy'])
            
            # Add subtitle language metadata if provided
            if subtitle_language:
                # Convert language name to ISO 639-2 code if needed
                lang_code = self._get_language_code(subtitle_language)
                cmd.extend(['-metadata:s:s:0', f'language={lang_code}'])
        
        cmd.extend([
            '-shortest',  # End output at shortest input
            '-y',  # Overwrite output
            output_path
        ])
        
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Video and audio combined successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg combination failed: {e.stderr}")
            raise RuntimeError(f"Failed to combine video and audio: {e.stderr}")


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration
    input_data = {
        'video_file': 'test/output/test02/step1/video_only.mp4',
        'audio_file': 'test/output/test02/step7/audio_translated_full.mp3',
        'output_dir': 'test/output/test02/step8/'
    }
    
    assembler = VideoAssembler(
        output_format='mp4',
        video_codec='copy',
        audio_codec='aac',
        audio_bitrate='192k'
    )
    
    result = assembler.process(input_data)
    print(f"\nOutput file: {result['video_translated']}")
    print(f"Metadata file: {result['metadata_file']}")


if __name__ == "__main__":
    main()
