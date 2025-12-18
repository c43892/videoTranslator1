"""
Step 7: Audio Track Stitching
Combines translated vocal clips with background audio to create complete translated audio.
Uses FFmpeg only - no external Python audio libraries required.
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Handle both relative and absolute imports
try:
    from ..utils.srt_handler import load_srt, SRTEntry
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.srt_handler import load_srt, SRTEntry


logger = logging.getLogger(__name__)


def get_audio_duration(file_path: str) -> float:
    """
    Get audio duration in seconds using ffprobe.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Get audio file information using ffprobe.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dictionary with audio info (duration, sample_rate, channels)
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'stream=duration,sample_rate,channels',
        '-of', 'json',
        file_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    stream = data['streams'][0]
    return {
        'duration': float(stream.get('duration', 0)),
        'sample_rate': int(stream.get('sample_rate', 44100)),
        'channels': int(stream.get('channels', 2))
    }


@dataclass
class ClipMetadata:
    """Metadata for a single processed clip"""
    clip_number: int
    source_file: str
    timestamp_start: str
    timestamp_end: str
    duration_original: float
    duration_translated: float
    time_stretch_applied: bool
    stretch_factor: Optional[float] = None
    gap_added_ms: Optional[int] = None
    final_duration: float = 0.0
    fade_in_ms: int = 50
    fade_out_ms: int = 50
    processed_file: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class StitchingMetadata:
    """Complete metadata for stitching operation"""
    no_vocals_file: str
    srt_reference: str
    total_clips: int
    output_file: str
    clips: List[Dict[str, Any]]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class AudioStitcher:
    """
    Audio track stitching processor for Step 7.
    Combines translated vocal clips with background audio using FFmpeg only.
    """
    
    def __init__(
        self,
        fade_in_ms: int = 50,
        fade_out_ms: int = 50,
        duration_tolerance_ms: int = 50,
        time_stretch_tool: str = "rubberband",
        export_bitrate: str = "192k"
    ):
        """
        Initialize AudioStitcher.
        
        Args:
            fade_in_ms: Fade-in duration in milliseconds
            fade_out_ms: Fade-out duration in milliseconds
            duration_tolerance_ms: Tolerance for duration matching
            time_stretch_tool: Tool for time-stretching ("rubberband" or "ffmpeg")
            export_bitrate: MP3 export bitrate
        """
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms
        self.duration_tolerance_ms = duration_tolerance_ms
        self.time_stretch_tool = time_stretch_tool
        self.export_bitrate = export_bitrate
        self.warnings: List[str] = []
        self.temp_dir = None
        
        logger.info(f"AudioStitcher initialized with {time_stretch_tool} for time-stretching")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for audio stitching.
        
        Args:
            input_data: {
                'translated_clips_dir': path to directory with translated_clip_*.mp3,
                'no_vocals_file': path to background audio,
                'srt_file': path to SRT_original.srt,
                'srt_file': path to SRT_original.srt,
                'output_dir': path for output files,
                'non_speech_file': (optional) path to non-speech vocals
            }
            
        Returns:
            {
                'audio_translated_full': path to output MP3,
                'metadata_file': path to metadata JSON,
                'metadata': stitching metadata dict
            }
        """
        logger.info("Starting audio stitching process...")
        
        # Extract input parameters
        translated_clips_dir = Path(input_data['translated_clips_dir'])
        no_vocals_file = Path(input_data['no_vocals_file'])
        srt_file = Path(input_data['srt_file'])
        output_dir = Path(input_data['output_dir'])
        non_speech_file = None
        if 'non_speech_file' in input_data and input_data['non_speech_file']:
            non_speech_file = Path(input_data['non_speech_file'])
            if not non_speech_file.exists():
                logger.warning(f"Non-speech file not found: {non_speech_file}")
                non_speech_file = None
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory for processing
        self.temp_dir = tempfile.mkdtemp(prefix="audio_stitch_")
        logger.debug(f"Created temp directory: {self.temp_dir}")
        
        # Reset warnings
        self.warnings = []
        
        try:
            # Get background audio info
            logger.info(f"Getting background audio info: {no_vocals_file}")
            bg_info = get_audio_info(str(no_vocals_file))
            logger.info(f"Background audio: {bg_info['duration']:.2f}s, {bg_info['sample_rate']}Hz, {bg_info['channels']} channels")
            
            if non_speech_file:
                logger.info(f"Using non-speech vocals track: {non_speech_file}")
            
            # Parse SRT timestamps
            logger.info(f"Parsing SRT file: {srt_file}")
            srt_entries = load_srt(str(srt_file))
            logger.info(f"Parsed {len(srt_entries)} SRT entries")
            
            # Process all clips
            logger.info(f"Processing {len(srt_entries)} clips...")
            print(f"Processing {len(srt_entries)} audio clips for stitching...")
            clips_metadata: List[ClipMetadata] = []
            processed_clip_files: List[Tuple[str, float]] = []  # (file_path, start_time_seconds)
            
            for i, entry in enumerate(srt_entries, start=1):
                clip_file = translated_clips_dir / f"translated_clip_{i}.mp3"
                
                # Show progress every 10 clips or at the end
                if i % 10 == 0 or i == len(srt_entries):
                    print(f"Progress: {i}/{len(srt_entries)} clips processed ({i*100//len(srt_entries)}%)")
                
                if not clip_file.exists():
                    warning = f"Clip file not found: {clip_file}"
                    logger.warning(warning)
                    self.warnings.append(warning)
                    continue
                
                # Get next entry for cross-fade detection
                next_entry = srt_entries[i] if i < len(srt_entries) else None
                
                # Process the clip
                processed_file, metadata = self.process_single_clip(
                    str(clip_file),
                    entry.duration,
                    entry,
                    i,
                    next_entry
                )
                
                # Store for mixing
                start_time_seconds = entry.start_seconds
                processed_clip_files.append((processed_file, start_time_seconds))
                clips_metadata.append(metadata)
                
                logger.debug(f"Processed clip {i}/{len(srt_entries)}: {clip_file.name}")
            
            # Mix all clips onto background using FFmpeg
            logger.info("Mixing clips onto background track with FFmpeg...")
            print(f"\nMixing {len(clips_metadata)} clips with background audio...")
            output_file = output_dir / "audio_translated_full.mp3"
            self.mix_clips_with_ffmpeg(
                str(no_vocals_file),
                processed_clip_files,
                str(output_file),
                non_speech_file=str(non_speech_file) if non_speech_file else None
            )
            print(f"✓ Audio stitching complete!")
            
            # Generate and save metadata
            metadata = StitchingMetadata(
                no_vocals_file=str(no_vocals_file),
                srt_reference=str(srt_file),
                total_clips=len(clips_metadata),
                output_file=str(output_file),
                clips=[clip.to_dict() for clip in clips_metadata],
                warnings=self.warnings
            )
            
            metadata_file = output_dir / "stitching_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Stitching complete! Output: {output_file}")
            logger.info(f"Metadata saved to: {metadata_file}")
            
            return {
                'audio_translated_full': str(output_file),
                'metadata_file': str(metadata_file),
                'metadata': metadata.to_dict()
            }
            
        finally:
            # Clean up temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                try:
                    shutil.rmtree(self.temp_dir)
                    logger.debug(f"Cleaned up temp directory: {self.temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")
    
    def process_single_clip(
        self,
        clip_file: str,
        original_duration: float,
        srt_entry: SRTEntry,
        clip_number: int,
        next_srt_entry: Optional[SRTEntry] = None
    ) -> Tuple[str, ClipMetadata]:
        """
        Process a single translated clip.
        
        Args:
            clip_file: Path to translated clip MP3
            original_duration: Expected duration from SRT (seconds)
            srt_entry: SRT entry for this clip
            clip_number: Sequential clip number
            next_srt_entry: Next SRT entry (if available) for cross-fade detection
            
        Returns:
            Tuple of (processed_file_path, metadata)
        """
        # Get the translated clip duration
        translated_duration = get_audio_duration(clip_file)
        
        # Calculate fade parameters based on gap to next clip
        # If gap is small (<200ms), reduce or eliminate fades to prevent overlap artifacts
        fade_in_ms = self.fade_in_ms
        fade_out_ms = self.fade_out_ms
        
        if next_srt_entry is not None:
            gap_to_next_ms = (next_srt_entry.start_seconds - srt_entry.end_seconds) * 1000
            
            if gap_to_next_ms < 200:  # Small gap detected
                # For very tight gaps, use minimal or no fade-out to avoid overlap
                if gap_to_next_ms < 50:
                    # Almost no gap - no fade-out to prevent overlap pop
                    fade_out_ms = 0
                    logger.debug(f"Clip {clip_number}: Gap to next clip is {gap_to_next_ms:.1f}ms - disabling fade-out for cross-fade")
                else:
                    # Small gap - reduce fade-out to half the gap
                    fade_out_ms = min(fade_out_ms, int(gap_to_next_ms / 2))
                    logger.debug(f"Clip {clip_number}: Gap to next clip is {gap_to_next_ms:.1f}ms - reducing fade-out to {fade_out_ms}ms")
        
        # Initialize metadata
        metadata = ClipMetadata(
            clip_number=clip_number,
            source_file=os.path.basename(clip_file),
            timestamp_start=srt_entry.start_time,
            timestamp_end=srt_entry.end_time,
            duration_original=original_duration,
            duration_translated=translated_duration,
            time_stretch_applied=False,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms
        )
        
        # Check duration difference
        duration_diff_ms = abs((translated_duration - original_duration) * 1000)
        
        # Output file for processed clip
        processed_file = os.path.join(self.temp_dir, f"processed_clip_{clip_number}.wav")
        
        if duration_diff_ms <= self.duration_tolerance_ms:
            # Durations match - just apply fades
            logger.debug(f"Clip {clip_number}: Duration match ({translated_duration:.3f}s)")
            metadata.final_duration = translated_duration
            self.apply_fades_with_ffmpeg(clip_file, processed_file, translated_duration, fade_in_ms, fade_out_ms)
        
        elif translated_duration > original_duration:
            # Clip is LONGER - apply time-stretching
            stretch_factor = original_duration / translated_duration
            logger.info(f"Clip {clip_number}: Stretching by factor {stretch_factor:.3f} ({translated_duration:.3f}s -> {original_duration:.3f}s)")
            
            # First time-stretch, then apply fades
            temp_stretched = os.path.join(self.temp_dir, f"stretched_{clip_number}.wav")
            self.time_stretch_audio(clip_file, temp_stretched, stretch_factor)
            self.apply_fades_with_ffmpeg(temp_stretched, processed_file, original_duration, fade_in_ms, fade_out_ms)
            
            metadata.time_stretch_applied = True
            metadata.stretch_factor = stretch_factor
            metadata.final_duration = original_duration
            
            # Add warning if extreme compression
            if stretch_factor < 0.7:
                warning = f"Clip {clip_number} required significant time-stretching by factor {stretch_factor:.3f}"
                logger.warning(warning)
                self.warnings.append(warning)
        
        else:
            # Clip is SHORTER - add gap/silence
            gap_ms = int((original_duration - translated_duration) * 1000)
            logger.debug(f"Clip {clip_number}: Adding {gap_ms}ms gap ({translated_duration:.3f}s -> {original_duration:.3f}s)")
            
            # Apply fades, then add silence
            temp_faded = os.path.join(self.temp_dir, f"faded_{clip_number}.wav")
            self.apply_fades_with_ffmpeg(clip_file, temp_faded, translated_duration, fade_in_ms, fade_out_ms)
            self.add_silence_with_ffmpeg(temp_faded, processed_file, gap_ms)
            
            metadata.gap_added_ms = gap_ms
            metadata.final_duration = original_duration
        
        metadata.processed_file = processed_file
        return processed_file, metadata
    
    def apply_fades_with_ffmpeg(
        self,
        input_file: str,
        output_file: str,
        duration: float,
        fade_in_ms: Optional[int] = None,
        fade_out_ms: Optional[int] = None
    ):
        """
        Apply fade-in and fade-out using FFmpeg.
        Uses exponential curves for smoother transitions and better pop prevention.
        
        Args:
            input_file: Input audio file
            output_file: Output audio file
            duration: Total duration in seconds
            fade_in_ms: Custom fade-in duration (uses self.fade_in_ms if None)
            fade_out_ms: Custom fade-out duration (uses self.fade_out_ms if None)
        """
        # Use provided values or fall back to instance defaults
        fade_in_ms = fade_in_ms if fade_in_ms is not None else self.fade_in_ms
        fade_out_ms = fade_out_ms if fade_out_ms is not None else self.fade_out_ms
        
        fade_in_sec = fade_in_ms / 1000.0
        fade_out_sec = fade_out_ms / 1000.0
        fade_out_start = max(0, duration - fade_out_sec)
        
        # Build filter: afade=in + afade=out with exponential curves
        # Using 'esin' (exponential sine) curve for smoother audio transitions
        filters = []
        if fade_in_ms > 0:
            filters.append(f"afade=t=in:st=0:d={fade_in_sec}:curve=esin")
        if fade_out_ms > 0:
            filters.append(f"afade=t=out:st={fade_out_start}:d={fade_out_sec}:curve=esin")
        
        filter_chain = ",".join(filters) if filters else "anull"
        
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-af', filter_chain,
            '-y',
            output_file
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
    
    def add_silence_with_ffmpeg(
        self,
        input_file: str,
        output_file: str,
        silence_ms: int
    ):
        """
        Add silence to the end of audio file using FFmpeg.
        
        Args:
            input_file: Input audio file
            output_file: Output audio file
            silence_ms: Silence duration in milliseconds
        """
        silence_sec = silence_ms / 1000.0
        
        # Use apad filter to add silence
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-af', f"apad=pad_dur={silence_sec}",
            '-y',
            output_file
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
    
    def time_stretch_audio(
        self,
        input_file: str,
        output_file: str,
        factor: float
    ):
        """
        Apply time-stretching with pitch preservation.
        
        Args:
            input_file: Input audio file
            output_file: Output audio file
            factor: Stretch factor (< 1.0 = compress, > 1.0 = expand)
        """
        if self.time_stretch_tool == "rubberband":
            self._time_stretch_rubberband(input_file, output_file, factor)
        elif self.time_stretch_tool == "ffmpeg":
            self._time_stretch_ffmpeg(input_file, output_file, factor)
        else:
            raise ValueError(f"Unknown time-stretch tool: {self.time_stretch_tool}")
    
    def _time_stretch_rubberband(
        self,
        input_file: str,
        output_file: str,
        factor: float
    ):
        """
        Time-stretch using rubberband CLI.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            factor: Stretch factor
        """
        try:
            # Run rubberband
            # -t factor: time stretch factor
            # -p: preserve pitch
            cmd = [
                "rubberband",
                "-t", str(factor),
                "-p",
                input_file,
                output_file
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Rubberband failed: {e.stderr}")
            logger.warning("Falling back to FFmpeg time-stretching")
            self._time_stretch_ffmpeg(input_file, output_file, factor)
            
        except FileNotFoundError:
            logger.error("Rubberband not found. Falling back to FFmpeg method.")
            self._time_stretch_ffmpeg(input_file, output_file, factor)
    
    def _time_stretch_ffmpeg(
        self,
        input_file: str,
        output_file: str,
        factor: float
    ):
        """
        Time-stretch using FFmpeg atempo filter.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            factor: Stretch factor
        """
        # Build atempo filter chain
        # atempo only accepts 0.5-2.0, so chain multiple filters if needed
        atempo_filters = []
        remaining_factor = factor
        
        while remaining_factor < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining_factor /= 0.5
        
        while remaining_factor > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining_factor /= 2.0
        
        atempo_filters.append(f"atempo={remaining_factor:.6f}")
        
        filter_chain = ",".join(atempo_filters)
        
        # Run FFmpeg
        cmd = [
            "ffmpeg",
            "-i", input_file,
            "-filter:a", filter_chain,
            "-y",
            output_file
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg time-stretching failed: {e.stderr}")
            raise
    
    def mix_clips_with_ffmpeg(
        self,
        background_file: str,
        clips: List[Tuple[str, float]],
        output_file: str,
        non_speech_file: Optional[str] = None
    ):
        """
        Mix all clips onto background track using FFmpeg filter_complex.
        
        Args:
            background_file: Background audio file path
            clips: List of (clip_file_path, start_time_seconds) tuples
            output_file: Output file path
            non_speech_file: Optional path to non-speech vocal track
        """
        if not clips and not non_speech_file:
            # No clips to mix, just copy background
            logger.warning("No clips to mix, copying background as-is")
            cmd = [
                'ffmpeg',
                '-i', background_file,
                '-c:a', 'libmp3lame',
                '-b:a', self.export_bitrate,
                '-y',
                output_file
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return
        
        # Optimize for command line length (avoid WinError 206)
        # 1. Run in temp directory
        # 2. Use relative paths for inputs where possible
        # 3. Use -filter_complex_script for the massive filter graph
        
        # Ensure we have a temp dir (should be created in process, but fallback if direct call)
        work_dir = self.temp_dir if self.temp_dir and os.path.exists(self.temp_dir) else os.path.dirname(output_file)
        
        # Prepare inputs using absolute paths for external files
        input_args = ['-i', os.path.abspath(background_file)]
        
        clip_start_index = 1
        vocals_base_stream = None
        
        if non_speech_file:
            input_args.extend(['-i', os.path.abspath(non_speech_file)])
            clip_start_index = 2
            vocals_base_stream = "[1:a]"
            
        # For clips, they are likely in the temp dir. Use relative paths if possible to save chars.
        # If not in temp dir, use absolute.
        cleaned_clips = []
        for clip_file, start_ch in clips:
            try:
                # Try to make relative to work_dir
                rel_path = os.path.relpath(clip_file, work_dir)
                # If relative path is shorter, use it
                if len(rel_path) < len(clip_file):
                    input_args.extend(['-i', rel_path])
                else:
                    input_args.extend(['-i', clip_file])
            except ValueError:
                # Path mismatch (different drives on Windows), use absolute
                input_args.extend(['-i', clip_file])
            
            cleaned_clips.append((start_ch)) # We only need the start time now
            
        
        # Build filter graph
        filter_parts = []
        
        current_vocal_stream = None
        if non_speech_file:
             current_vocal_stream = "[1:a]"
             
        for i, start_time in enumerate(cleaned_clips, start=clip_start_index):
             delay_ms = int(start_time * 1000)
             delayed_stream = f"[delayed{i}]"
             filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}{delayed_stream}")
             
             if current_vocal_stream is None:
                 current_vocal_stream = delayed_stream
             else:
                 duration_mode = "first" if non_speech_file else "longest"
                 mixed_vocal = f"[vocal_mix_{i}]"
                 filter_parts.append(f"{current_vocal_stream}{delayed_stream}amix=inputs=2:duration={duration_mode}:dropout_transition=2:normalize=0{mixed_vocal}")
                 current_vocal_stream = mixed_vocal

        # --- Stage 2: Final Mix (Background + Vocal Track) ---
        if current_vocal_stream:
             # Mix background [0:a] with full vocal track
             filter_parts.append(f"[0:a]{current_vocal_stream}amix=inputs=2:duration=first:dropout_transition=2:normalize=0[out]")
        else:
             filter_parts.append(f"[0:a]anull[out]")
        
        filter_complex = ";".join(filter_parts)
        
        # Write filter graph to script file
        filter_script_path = os.path.join(work_dir, "filter_script.txt")
        with open(filter_script_path, "w", encoding="utf-8") as f:
            f.write(filter_complex)
        
        # Build command
        cmd = [
            'ffmpeg',
            *input_args,
            '-filter_complex_script', 'filter_script.txt', # Relative to work_dir
            '-map', '[out]',
            '-c:a', 'libmp3lame',
            '-b:a', self.export_bitrate,
            '-y',
            os.path.abspath(output_file) # Ensure output is absolute
        ]
        
        # Log command length for debugging
        cmd_str = " ".join(cmd)
        logger.debug(f"FFmpeg command length: {len(cmd_str)} chars")
        logger.debug(f"FFmpeg filter script size: {len(filter_complex)} chars")
        
        try:
            # Run in work_dir so relative paths work
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=work_dir)
            logger.info(f"Successfully mixed {len(clips)} clips onto background")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg mixing failed: {e.stderr}")
            # If script failed, maybe dump the script content to log for debug?
            # logger.error(f"Filter script content: {filter_complex}")
            raise


def main():
    """Example usage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example configuration
    input_data = {
        'translated_clips_dir': 'test/output/test02/step6/',
        'no_vocals_file': 'test/output/test02/step2/no_vocals.wav',
        'srt_file': 'test/output/test02/step3/SRT_original.srt',
        'output_dir': 'test/output/test02/step7/'
    }
    
    stitcher = AudioStitcher(
        fade_in_ms=50,
        fade_out_ms=50,
        time_stretch_tool="ffmpeg",
        export_bitrate="192k"
    )
    
    result = stitcher.process(input_data)
    print(f"\nOutput file: {result['audio_translated_full']}")
    print(f"Metadata file: {result['metadata_file']}")


if __name__ == "__main__":
    main()
