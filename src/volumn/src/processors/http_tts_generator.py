"""
HTTP-based TTS Generator for Step 6
Uses external Gradio IndexTTS2 service instead of local model
"""

import logging
import shutil
import subprocess
from pathlib import Path
import re
import sys

try:
    from gradio_client import Client, handle_file
    GRADIO_CLIENT_AVAILABLE = True
except ImportError:
    GRADIO_CLIENT_AVAILABLE = False
    logging.warning("gradio_client not available. Install with: pip install gradio_client")

logger = logging.getLogger(__name__)


class HttpTTSGenerator:
    """
    HTTP-based TTS generator using remote Gradio IndexTTS2 service.
    
    This class provides the same interface as IndexTTS2Generator but calls
    an external TTS service instead of loading models locally.
    
    Benefits:
    - No local model loading (saves 7+ minutes at startup)
    - Reduced container memory usage (no 4GB+ model in memory)
    - Potentially faster generation if host machine has better GPU
    - Easier to scale and maintain (service can be shared)
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:7860",
        default_format: str = "mp3",
        default_bitrate: str = "320k",
        enable_noise_reduction: bool = True,
        noise_reduction_strength: int = 3
    ):
        """
        Initialize HTTP TTS generator.
        
        Args:
            api_url: URL of Gradio IndexTTS2 service (default: localhost:7860)
            default_format: Default output format (mp3, wav, etc.)
            default_bitrate: Default audio bitrate (e.g., "320k")
            enable_noise_reduction: Enable noise reduction (not used for HTTP API)
            noise_reduction_strength: Noise reduction strength (not used for HTTP API)
        """
        if not GRADIO_CLIENT_AVAILABLE:
            raise RuntimeError("gradio_client is required but not installed. Run: pip install gradio_client")
        
        self.api_url = api_url
        self.default_format = default_format
        self.default_bitrate = default_bitrate
        self.enable_noise_reduction = enable_noise_reduction
        self.noise_reduction_strength = noise_reduction_strength
        
        logger.info(f"Initializing HTTP TTS Generator with API: {api_url}")
        
        try:
            self.client = Client(api_url)
            logger.info("✅ Connected to Gradio TTS API successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to TTS API at {api_url}: {e}")
    
    def generate_single_clip(
        self,
        reference_audio_path: Path,
        translated_text: str,
        output_path: Path,
        emo_alpha: float = 1.0,
        match_duration: bool = True,
        **kwargs
    ) -> bool:
        """
        Generate TTS audio for a SINGLE clip using HTTP API.
        
        This matches the interface of IndexTTS2Generator.generate_single_clip()
        
        Args:
            reference_audio_path: Audio file for voice cloning
            translated_text: Text to synthesize
            output_path: Where to save result
            emo_alpha: Emotion influence (0.0-1.0, default: 1.0)
            match_duration: Whether to adjust TTS output to match reference duration
            **kwargs: Additional params (use_random, time_stretch_tool, etc.)
        
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Validate input
            if not reference_audio_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
            
            # Get reference duration if matching is enabled
            target_duration = None
            if match_duration:
                target_duration = self.get_audio_duration(reference_audio_path)
                logger.debug(f"Reference duration: {target_duration:.3f}s")
            
            logger.info(f"Calling TTS API: {translated_text[:50]}...")
            
            # Call Gradio API with minimal required parameters
            # Using handle_file() to properly upload local files to the remote API
            # Only passing the required parameters to avoid API errors
            
            result_audio = self.client.predict(
                "Same as the voice reference",  # emo_control_method
                handle_file(str(reference_audio_path)),  # prompt (reference audio)
                translated_text,  # text to synthesize
                handle_file(str(reference_audio_path)),  # emo_ref_path (use same as prompt for consistency)
                api_name="/gen_single"
            )
            
            logger.info(f"✅ TTS API returned result")
            logger.debug(f"   Result type: {type(result_audio)}")
            logger.debug(f"   Result: {result_audio}")
            
            # Extract file path from result
            # Result format: {'visible': True, 'value': 'file=C:\\...\\output.wav', '__type__': 'update'}
            if isinstance(result_audio, dict):
                if 'value' in result_audio:
                    # Extract path from "file=..." format
                    value_str = result_audio['value']
                    if value_str.startswith('file='):
                        result_path = Path(value_str[5:])  # Remove "file=" prefix
                    else:
                        result_path = Path(value_str)
                elif 'path' in result_audio:
                    result_path = Path(result_audio['path'])
                else:
                    raise ValueError(f"Cannot extract file path from result: {result_audio}")
            elif isinstance(result_audio, str):
                if result_audio.startswith('file='):
                    result_path = Path(result_audio[5:])
                else:
                    result_path = Path(result_audio)
            else:
                raise ValueError(f"Unexpected result type from API: {type(result_audio)}")
            
            logger.debug(f"Extracted result path: {result_path}")
            
            if not result_path.exists():
                raise FileNotFoundError(f"TTS API result file not found: {result_path}")
            
            # Copy result to temporary working file
            temp_wav = output_path.with_suffix('.wav')
            shutil.copy2(result_path, temp_wav)
            logger.debug(f"Copied API result to: {temp_wav}")
            
            # Trim silence from beginning and end
            trimmed_wav = temp_wav.parent / f"trimmed_{temp_wav.name}"
            self._trim_silence(str(temp_wav), str(trimmed_wav))
            temp_wav.unlink()
            trimmed_wav.rename(temp_wav)
            logger.debug("Trimmed silence from TTS output")
            
            # Match duration if enabled (same logic as IndexTTS2Generator)
            if match_duration and target_duration:
                generated_duration = self.get_audio_duration(temp_wav)
                
                # Check for invalid duration (failed generation or empty file)
                if generated_duration <= 0.01:
                    logger.warning(f"TTS generated invalid audio (duration={generated_duration}s). Falling back to original.")
                    # Fallback: Copy original reference audio
                    if output_path.suffix.lower() != reference_audio_path.suffix.lower():
                        self._convert_audio(reference_audio_path, output_path, self.default_bitrate)
                    else:
                        shutil.copy2(reference_audio_path, output_path)
                    
                    # Clean up temp file
                    if temp_wav.exists():
                        temp_wav.unlink()
                    return True

                if generated_duration > target_duration:
                    duration_diff = generated_duration - target_duration
                    
                    if duration_diff > 0.05:  # Only adjust if difference > 50ms
                        stretch_factor = target_duration / generated_duration
                        logger.info(f"Time-stretching TTS: {generated_duration:.3f}s -> {target_duration:.3f}s (factor: {stretch_factor:.3f})")
                        
                        stretched_wav = temp_wav.parent / f"stretched_{temp_wav.name}"
                        time_stretch_tool = kwargs.get('time_stretch_tool', 'ffmpeg')
                        
                        self._time_stretch_audio(
                            str(temp_wav),
                            str(stretched_wav),
                            stretch_factor,
                            time_stretch_tool
                        )
                        
                        temp_wav.unlink()
                        stretched_wav.rename(temp_wav)
                        final_duration = self.get_audio_duration(temp_wav)
                        logger.info(f"Time-stretching completed: final={final_duration:.3f}s (target={target_duration:.3f}s)")
            
            # Convert to target format if needed
            if output_path.suffix.lower() != '.wav':
                self._convert_audio(temp_wav, output_path, self.default_bitrate)
                temp_wav.unlink()
            else:
                temp_wav.rename(output_path)
            
            logger.info(f"TTS generated successfully: {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"HTTP TTS generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_event_text(self, text: str) -> bool:
        """Check if text is pure event (e.g., [[ laughing ]])."""
        pattern = r'^\[\[.*\]\]$'
        return bool(re.match(pattern, text.strip()))
    
    def copy_event_clip(
        self,
        source_path: Path,
        destination_path: Path
    ) -> bool:
        """Copy original clip for events (no TTS generation needed)."""
        try:
            shutil.copy2(source_path, destination_path)
            logger.info(f"Copied event clip: {source_path.name} → {destination_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy event clip: {e}")
            return False
    
    def get_audio_duration(self, audio_file: Path) -> float:
        """Get audio duration in seconds using ffprobe."""
        cmd = [
            'ffprobe',
            '-i', str(audio_file),
            '-show_entries', 'format=duration',
            '-v', 'quiet',
            '-of', 'csv=p=0'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            val = result.stdout.strip()
            if val == 'N/A' or not val:
                return 0.0
            return float(val)
        except (ValueError, subprocess.CalledProcessError) as e:
            logger.warning(f"Could not determine audio duration for {audio_file}: {e}")
            return 0.0
    
    def _trim_silence(self, input_file: str, output_file: str, threshold: str = "-40dB", duration: float = 0.1):
        """Trim silence from beginning and end using ffmpeg."""
        cmd = [
            'ffmpeg', '-i', input_file,
            '-af', f'silenceremove=start_periods=1:start_threshold={threshold}:start_duration={duration}:'
                   f'stop_periods=-1:stop_threshold={threshold}:stop_duration={duration}',
            '-y', output_file
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    def _time_stretch_audio(
        self,
        input_file: str,
        output_file: str,
        duration_factor: float,
        tool: str = 'ffmpeg'
    ):
        """
        Time-stretch audio to match target duration.
        
        Args:
            input_file: Input audio file
            output_file: Output audio file
            duration_factor: Target/generated duration ratio (< 1.0 = speed up)
            tool: Tool to use ('ffmpeg' or 'rubberband')
        """
        # Convert duration_factor to speed factor (inverse relationship)
        # duration_factor = 0.9 means compress to 90% of original time
        # speed_factor = 1/0.9 = 1.111 means play 11.1% faster
        speed_factor = 1.0 / duration_factor
        
        if tool == 'ffmpeg':
            cmd = [
                'ffmpeg', '-i', input_file,
                '-filter:a', f'atempo={speed_factor}',
                '-y', output_file
            ]
        elif tool == 'rubberband':
            # Rubberband uses stretch factor (same as duration_factor)
            cmd = [
                'rubberband',
                '--time', str(duration_factor),
                '--pitch-high-quality',
                input_file,
                output_file
            ]
        else:
            raise ValueError(f"Unknown tool: {tool}")
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    def _convert_audio(self, input_file: Path, output_file: Path, bitrate: str):
        """Convert audio format using ffmpeg."""
        cmd = [
            'ffmpeg', '-i', str(input_file),
            '-b:a', bitrate,
            '-y', str(output_file)
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
