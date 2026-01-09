"""
TTS (Text-to-Speech) Generator for Step 6
Provides voice cloning and TTS generation using IndexTTS2
"""

import sys
import shutil
import subprocess
import json
from pathlib import Path
import logging
import re

# Add IndexTTS2 to Python path
INDEXTTS_PATH = Path("/app/volumn/.cache/index-tts")
if str(INDEXTTS_PATH) not in sys.path:
    sys.path.insert(0, str(INDEXTTS_PATH))

try:
    from indextts.infer_v2 import IndexTTS2 as IndexTTS2Model
except ImportError as e:
    # Silently fail - warning will be shown only when user tries to use local mode
    IndexTTS2Model = None

logger = logging.getLogger(__name__)


class IndexTTS2Generator:
    """
    IndexTTS2-based TTS generator with voice cloning.
    
    This class provides CORE TTS functionality for SINGLE clips.
    Batch processing should be handled by calling code.
    """
    
    def __init__(
        self,
        model_dir: str = "/app/volumn/.cache/index-tts/checkpoints",
        config_path: str = "/app/volumn/.cache/index-tts/checkpoints/config.yaml",
        use_fp16: bool = True,
        use_cuda_kernel: bool = False,
        use_deepspeed: bool = False,
        default_format: str = "mp3",
        default_bitrate: str = "320k",  # Increased from 192k to 320k for better quality
        enable_noise_reduction: bool = True,  # Enable adaptive noise reduction
        noise_reduction_strength: int = 3     # Noise reduction strength (0-20, 3 is light)
    ):
        """
        Initialize IndexTTS2 generator.
        
        Args:
            model_dir: Directory containing IndexTTS2 model checkpoints
            config_path: Path to config.yaml
            use_fp16: Use half-precision (faster, less VRAM) - RECOMMENDED: True
            use_cuda_kernel: Use compiled CUDA kernels - RECOMMENDED: False (test first)
            use_deepspeed: Use DeepSpeed acceleration - RECOMMENDED: False (may be slower)
            default_format: Default output format (mp3, wav, etc.)
            default_bitrate: Default audio bitrate (e.g., "320k")
            enable_noise_reduction: Enable adaptive noise reduction - RECOMMENDED: True
            noise_reduction_strength: Noise reduction strength (0-20, default: 3 = light)
                                     Lower = more natural but may have noise
                                     Higher = cleaner but may sound processed
        """
        self.model_dir = Path(model_dir)
        self.config_path = Path(config_path)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"IndexTTS2 model directory not found: {model_dir}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"IndexTTS2 config not found: {config_path}")
        
        if IndexTTS2Model is None:
            raise RuntimeError("IndexTTS2 is not available. Please check installation.")
        
        logger.info(f"Initializing IndexTTS2 with model_dir={model_dir}, use_fp16={use_fp16}")
        logger.info("This may take 30-60 seconds to load all models...")
        
        # Initialize IndexTTS2 model
        # NOTE: This may take 30-60 seconds to load all models
        self.tts = IndexTTS2Model(
            cfg_path=str(self.config_path),
            model_dir=str(self.model_dir),
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed
        )
        
        self.default_format = default_format
        self.default_bitrate = default_bitrate
        self.enable_noise_reduction = enable_noise_reduction
        self.noise_reduction_strength = noise_reduction_strength
        logger.info("IndexTTS2 model loaded successfully")
    
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
        CORE METHOD: Generate TTS audio for a SINGLE clip.
        
        This is the fundamental TTS operation:
        Input: 1 reference audio + 1 text → Output: 1 TTS audio file
        
        IndexTTS2 Process:
        1. Generate audio to WAV format (IndexTTS2 native output)
        2. Adjust duration to match reference if needed (pitch-preserving)
        3. Convert WAV to target format (MP3) using FFmpeg
        4. Clean up temporary files
        
        Args:
            reference_audio_path: Audio file for voice cloning (e.g., original_clip_5.mp3)
            translated_text: Text to synthesize (e.g., "Hello, how are you?")
            output_path: Where to save result (e.g., translated_clip_5.mp3)
            emo_alpha: Emotion influence (0.0-1.0, default: 1.0)
            match_duration: Whether to adjust TTS output to match reference duration (default: True)
            **kwargs: Additional params (use_random=False, time_stretch_tool='ffmpeg', etc.)
        
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
            
            # Step 1: Generate to WAV first (IndexTTS2 native format)
            temp_wav = output_path.with_suffix('.wav')
            use_random = kwargs.get('use_random', False)
            
            logger.info(f"Generating TTS: {translated_text[:50]}...")
            
            # Call IndexTTS2 inference
            # Only use parameters supported by the model
            self.tts.infer(
                spk_audio_prompt=str(reference_audio_path),  # Voice reference
                text=translated_text,                         # Text to speak
                output_path=str(temp_wav),                   # Output WAV
                emo_alpha=emo_alpha,                         # Emotion strength (default: 1.0)
                use_random=use_random,                       # Randomness (False recommended for consistency)
                verbose=False                                # Reduce output noise
            )
            
            # Trim silence from beginning and end of TTS output
            trimmed_wav = temp_wav.parent / f"trimmed_{temp_wav.name}"
            self._trim_silence(str(temp_wav), str(trimmed_wav))
            temp_wav.unlink()
            trimmed_wav.rename(temp_wav)
            logger.debug("Trimmed silence from TTS output")
            
            # Step 2: Match duration if enabled
            if match_duration and target_duration:
                generated_duration = self.get_audio_duration(temp_wav)
                
                # Only compress if TTS is LONGER than original (never expand if shorter)
                if generated_duration > target_duration:
                    duration_diff = generated_duration - target_duration
                    
                    # Only adjust if difference is significant (> 50ms)
                    if duration_diff > 0.05:
                        # Calculate duration stretch factor: < 1.0 means compress
                        # This will be converted to speed factor (inverse) in _time_stretch_ffmpeg
                        stretch_factor = target_duration / generated_duration
                        logger.info(f"Time-stretching TTS (speed up, pitch preserved): {generated_duration:.3f}s -> {target_duration:.3f}s (duration_factor: {stretch_factor:.3f})")
                        
                        stretched_wav = temp_wav.parent / f"stretched_{temp_wav.name}"
                        time_stretch_tool = kwargs.get('time_stretch_tool', 'ffmpeg')
                        
                        self._time_stretch_audio(
                            str(temp_wav),
                            str(stretched_wav),
                            stretch_factor,
                            time_stretch_tool
                        )
                        
                        # Replace original with stretched version
                        temp_wav.unlink()
                        stretched_wav.rename(temp_wav)
                        final_duration = self.get_audio_duration(temp_wav)
                        logger.info(f"Time-stretching completed: final duration = {final_duration:.3f}s (target was {target_duration:.3f}s)")
                    else:
                        logger.debug(f"TTS slightly longer ({generated_duration:.3f}s vs {target_duration:.3f}s) but within tolerance, keeping as-is")
                elif generated_duration < target_duration:
                    logger.debug(f"TTS shorter than original ({generated_duration:.3f}s < {target_duration:.3f}s), keeping as-is")
                else:
                    logger.debug(f"TTS matches original duration ({generated_duration:.3f}s)")
            
            # Step 3: Convert to target format if needed
            if output_path.suffix.lower() != '.wav':
                self._convert_audio(temp_wav, output_path, self.default_bitrate)
                temp_wav.unlink()  # Delete temp WAV
            else:
                temp_wav.rename(output_path)
            
            logger.info(f"TTS generated successfully: {output_path.name}")
            
            # Clear GPU cache to prevent memory accumulation
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return False
    
    def is_event_text(self, text: str) -> bool:
        """
        Check if text is pure event (e.g., [[ laughing ]]).
        
        Args:
            text: Text to check
            
        Returns:
            True if text is pure event, False otherwise
        """
        pattern = r'^\[\[.*\]\]$'
        return bool(re.match(pattern, text.strip()))
    
    def copy_event_clip(
        self,
        source_path: Path,
        destination_path: Path
    ) -> bool:
        """
        Copy original clip for events (no TTS generation needed).
        
        Args:
            source_path: Path to original audio clip
            destination_path: Path for output copy
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            shutil.copy2(source_path, destination_path)
            logger.info(f"Copied event clip: {source_path.name} → {destination_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy event clip: {e}")
            return False
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get audio duration in seconds using ffprobe.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    
    def _convert_audio(self, input_path: Path, output_path: Path, bitrate: str):
        """
        Convert audio format using FFmpeg (internal helper method).
        
        Args:
            input_path: Input audio file (WAV)
            output_path: Output audio file (MP3)
            bitrate: Target bitrate (e.g., "320k")
        """
        # Build audio filter chain
        filters = []
        
        # Apply noise reduction if enabled
        if self.enable_noise_reduction:
            # highpass/lowpass filters remove sub-bass rumble and ultrasonic noise
            filters.append('highpass=f=80')         # Remove sub-bass rumble below 80Hz
            filters.append('lowpass=f=15000')       # Remove ultrasonic noise above 15kHz
            # afftdn is FFmpeg's adaptive FFT denoiser
            # nr = noise reduction in dB (3 is light, 10 is moderate, 20 is aggressive)
            # nf = noise floor in dB (default: -50dB)
            filters.append(f'afftdn=nr={self.noise_reduction_strength}:nf=-25')
        
        filter_chain = ','.join(filters) if filters else 'anull'
        
        # Apply high-quality conversion
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-af', filter_chain,
            '-codec:a', 'libmp3lame',
            '-b:a', bitrate,
            '-q:a', '0',  # Highest quality VBR
            '-ar', '22050',  # 22.05kHz - match common TTS output, avoid upsampling artifacts
            str(output_path)
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
    
    def _time_stretch_audio(
        self,
        input_path: str,
        output_path: str,
        factor: float,
        tool: str = "ffmpeg"
    ):
        """
        Time-stretch audio while preserving pitch.
        
        Args:
            input_path: Input audio file
            output_path: Output audio file
            factor: Stretch factor (< 1.0 = compress, > 1.0 = expand)
            tool: Tool to use ('ffmpeg' or 'rubberband')
        """
        if tool == "rubberband":
            self._time_stretch_rubberband(input_path, output_path, factor)
        else:
            self._time_stretch_ffmpeg(input_path, output_path, factor)
    
    def _time_stretch_rubberband(
        self,
        input_path: str,
        output_path: str,
        factor: float
    ):
        """
        Time-stretch using rubberband CLI.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            factor: Stretch factor
        """
        try:
            cmd = [
                "rubberband",
                "-t", str(factor),
                "-p",
                input_path,
                output_path
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Rubberband failed or not found, falling back to FFmpeg")
            self._time_stretch_ffmpeg(input_path, output_path, factor)
    
    def _time_stretch_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        factor: float
    ):
        """
        Time-stretch using FFmpeg atempo filter.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            factor: Stretch factor
        """
        # Build atempo filter chain (atempo only accepts 0.5-2.0)
        # NOTE: FFmpeg atempo uses SPEED factor (inverse of duration factor)
        # To compress audio to 0.5x duration, use atempo=2.0 (double speed)
        # To expand audio to 2.0x duration, use atempo=0.5 (half speed)
        # So we need to use: atempo_value = 1 / duration_factor
        
        speed_factor = 1.0 / factor  # Convert duration factor to speed factor
        atempo_filters = []
        remaining_factor = speed_factor
        
        # If speed > 2.0 (compress a lot), chain multiple 2.0 filters
        while remaining_factor > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining_factor /= 2.0
        
        # If speed < 0.5 (expand a lot), chain multiple 0.5 filters
        while remaining_factor < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining_factor *= 2.0
        
        atempo_filters.append(f"atempo={remaining_factor:.6f}")
        filter_chain = ",".join(atempo_filters)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:a", filter_chain,
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg time-stretching failed: {result.stderr}")
    
    def _trim_silence(
        self,
        input_path: str,
        output_path: str,
        silence_threshold: str = "-60dB",  # Very lenient to preserve natural audio start/end
        silence_duration: str = "0.3"       # Longer duration to be more conservative
    ):
        """
        Trim silence from beginning and end of audio using FFmpeg silenceremove filter.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            silence_threshold: Threshold for silence detection (default: -60dB, very lenient)
            silence_duration: Minimum silence duration to detect (default: 0.3s, conservative)
        """
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-af', f'silenceremove=start_periods=1:start_threshold={silence_threshold}:start_duration={silence_duration},'
                   f'areverse,silenceremove=start_periods=1:start_threshold={silence_threshold}:start_duration={silence_duration},areverse',
            '-c:a', 'pcm_s16le',  # Use uncompressed PCM to avoid re-encoding artifacts
            output_path
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
