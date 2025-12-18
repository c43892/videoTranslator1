"""
Vocal/Instrumental Separation Processor (Step 2)
Separates audio into vocal and non-vocal (instrumental) tracks using Demucs.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging


class DemucsVocalSeparator:
    """
    Separates vocals from instrumental audio using locally installed Demucs.
    
    Uses the htdemucs model in two-stems mode for efficient separation into:
    - vocals: Isolated vocal track
    - no_vocals: Background/instrumental track
    """
    
    def __init__(
        self,
        model: str = 'htdemucs',
        device: str = 'cuda',
        output_format: str = 'mp3',
        two_stems: bool = True,
        float32: bool = False,
        clip_mode: str = 'rescale',
        segment_size: Optional[int] = None,
        keep_intermediate: bool = False,
        vocals_filename: str = 'vocals.mp3',
        no_vocals_filename: str = 'no_vocals.mp3'
    ):
        """
        Initialize the Demucs vocal separator.
        
        Args:
            model: Model name (default: 'htdemucs')
            device: 'auto', 'cpu', or 'cuda' (default: 'cuda')
            output_format: 'wav', 'mp3', 'flac' (default: 'mp3')
            two_stems: Use two-stem mode (default: True)
            float32: Use float32 precision (default: False)
            clip_mode: How to handle clipping ('rescale', 'clamp', etc.)
            segment_size: Segment size for processing (None for auto)
            keep_intermediate: Keep intermediate Demucs output files
            vocals_filename: Output filename for vocals (default: 'vocals.mp3')
            no_vocals_filename: Output filename for no_vocals (default: 'no_vocals.mp3')
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.device = device
        self.output_format = output_format
        self.two_stems = two_stems
        self.float32 = float32
        self.clip_mode = clip_mode
        self.segment_size = segment_size
        self.keep_intermediate = keep_intermediate
        self.vocals_filename = vocals_filename
        self.no_vocals_filename = no_vocals_filename
        
        # Validate Demucs installation
        self._validate_demucs_installation()
    
    def _validate_demucs_installation(self) -> None:
        """
        Check if Demucs is installed and accessible.
        
        Raises:
            RuntimeError: If Demucs is not installed or not accessible
        """
        try:
            # Try importing demucs module to validate installation
            import demucs
            self.logger.info(f"Demucs module found: {demucs.__version__ if hasattr(demucs, '__version__') else 'version unknown'}")
        except ImportError:
            raise RuntimeError(
                "Demucs is not installed. Please install it using: pip install demucs"
            )
        except Exception as e:
            self.logger.warning(f"Demucs validation warning: {str(e)}")
    
    def process(self, input_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Separate vocals from instrumental audio.
        
        Args:
            input_path: Path to the input audio file (audio_full.wav)
            output_dir: Directory where output files will be saved
        
        Returns:
            Tuple of (vocals_path, no_vocals_path)
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If separation fails
        """
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Starting vocal separation for: {input_path}")
        self.logger.info(f"Model: {self.model}, Device: {self.device}")
        
        # Create temporary directory for Demucs output
        temp_output_dir = os.path.join(output_dir, 'demucs_temp')
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            # Run Demucs separation
            self._run_demucs(input_path, temp_output_dir)
            
            # Get paths to separated stems
            input_filename = Path(input_path).stem
            demucs_output_path = os.path.join(temp_output_dir, self.model, input_filename)
            
            if self.two_stems:
                # In two-stems mode, Demucs outputs 'vocals' and 'no_vocals'
                vocals_src = os.path.join(demucs_output_path, f'vocals.{self.output_format}')
                no_vocals_src = os.path.join(demucs_output_path, f'no_vocals.{self.output_format}')
            else:
                # In four-stems mode, combine non-vocal stems
                vocals_src = os.path.join(demucs_output_path, f'vocals.{self.output_format}')
                no_vocals_src = self._combine_stems(demucs_output_path, output_dir)
            
            # Validate output files exist
            if not os.path.exists(vocals_src):
                raise RuntimeError(f"Vocals output not found: {vocals_src}")
            if not os.path.exists(no_vocals_src):
                raise RuntimeError(f"No-vocals output not found: {no_vocals_src}")
            
            # Move files to final destination
            vocals_dest = os.path.join(output_dir, self.vocals_filename)
            no_vocals_dest = os.path.join(output_dir, self.no_vocals_filename)
            
            shutil.copy2(vocals_src, vocals_dest)
            shutil.copy2(no_vocals_src, no_vocals_dest)
            
            # Validate output
            if not self.validate_output(vocals_dest, no_vocals_dest):
                raise RuntimeError("Output validation failed")
            
            self.logger.info(f"Vocal separation completed successfully")
            self.logger.info(f"Vocals: {vocals_dest}")
            self.logger.info(f"No vocals: {no_vocals_dest}")
            
            return vocals_dest, no_vocals_dest
        
        except Exception as e:
            self.logger.error(f"Vocal separation failed: {str(e)}")
            raise RuntimeError(f"Vocal separation failed: {str(e)}")
        
        finally:
            # Cleanup temporary files
            if not self.keep_intermediate and os.path.exists(temp_output_dir):
                try:
                    shutil.rmtree(temp_output_dir)
                    self.logger.debug("Cleaned up temporary files")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary files: {str(e)}")
            
            # Release GPU memory after Demucs processing
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("GPU memory released after vocal separation")
            except ImportError:
                pass  # PyTorch not available, skip GPU cleanup
            except Exception as e:
                self.logger.warning(f"Failed to release GPU memory: {str(e)}")
    
    def _run_demucs(self, input_path: str, output_dir: str) -> None:
        """
        Execute Demucs command for vocal separation.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory for Demucs output
        
        Raises:
            RuntimeError: If Demucs execution fails
        """
        # Build Demucs command
        cmd = ['demucs']
        
        # Add model selection
        cmd.extend(['-n', self.model])
        
        # Add output directory
        cmd.extend(['--out', output_dir])
        
        # Add device selection
        if self.device != 'auto':
            cmd.extend(['-d', self.device])
        
        # Add two-stems mode if enabled
        if self.two_stems:
            cmd.append('--two-stems=vocals')
        
        # Add float32 precision
        if self.float32:
            cmd.append('--float32')
        
        # Add clip mode
        cmd.extend(['--clip-mode', self.clip_mode])
        
        # Add segment size if specified
        if self.segment_size:
            cmd.extend(['--segment', str(self.segment_size)])
        
        # Add output format
        if self.output_format == 'mp3':
            cmd.append('--mp3')
        elif self.output_format == 'flac':
            cmd.append('--flac')
        # WAV is default, no flag needed
        
        # Add input file
        cmd.append(input_path)
        
        self.logger.debug(f"Executing Demucs command: {' '.join(cmd)}")
        
        # Execute command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout:
                self.logger.debug(f"Demucs stdout: {result.stdout}")
            if result.stderr:
                self.logger.debug(f"Demucs stderr: {result.stderr}")
        
        except subprocess.CalledProcessError as e:
            error_msg = f"Demucs command failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f"\nError output: {e.stderr}"
            
            # Check for common errors and provide helpful messages
            if 'CUDA out of memory' in str(e.stderr):
                error_msg += "\n\nSuggestion: Try using CPU mode or reduce segment size"
                self.logger.error("GPU out of memory - consider using CPU mode")
            elif 'No such file or directory' in str(e.stderr):
                error_msg += "\n\nSuggestion: Check input file path"
            
            raise RuntimeError(error_msg)
        
        except Exception as e:
            raise RuntimeError(f"Failed to execute Demucs: {str(e)}")
    
    def _combine_stems(self, stems_dir: str, output_dir: str) -> str:
        """
        Combine non-vocal stems (bass, drums, other) into no_vocals track.
        Only used when two_stems mode is False.
        
        Args:
            stems_dir: Directory containing individual stems
            output_dir: Output directory for combined file
        
        Returns:
            Path to the combined no_vocals file
        
        Raises:
            RuntimeError: If stem combination fails
        """
        self.logger.info("Combining non-vocal stems (bass, drums, other)")
        
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError(
                "pydub is required for stem combination. Install it with: pip install pydub"
            )
        
        # Load non-vocal stems
        bass_path = os.path.join(stems_dir, f'bass.{self.output_format}')
        drums_path = os.path.join(stems_dir, f'drums.{self.output_format}')
        other_path = os.path.join(stems_dir, f'other.{self.output_format}')
        
        # Check if all stems exist
        for stem_path, stem_name in [(bass_path, 'bass'), (drums_path, 'drums'), (other_path, 'other')]:
            if not os.path.exists(stem_path):
                raise RuntimeError(f"Missing stem: {stem_name} at {stem_path}")
        
        # Load and combine stems
        bass = AudioSegment.from_file(bass_path)
        drums = AudioSegment.from_file(drums_path)
        other = AudioSegment.from_file(other_path)
        
        # Mix the stems
        no_vocals = bass.overlay(drums).overlay(other)
        
        # Save combined track
        no_vocals_path = os.path.join(output_dir, f'no_vocals_temp.{self.output_format}')
        no_vocals.export(no_vocals_path, format=self.output_format)
        
        self.logger.info(f"Combined stems saved to: {no_vocals_path}")
        
        return no_vocals_path
    
    def validate_output(self, vocals_path: str, no_vocals_path: str) -> bool:
        """
        Validate that output files exist and are valid audio files.
        
        Args:
            vocals_path: Path to vocals output file
            no_vocals_path: Path to no_vocals output file
        
        Returns:
            True if validation passes, False otherwise
        """
        # Check if files exist
        if not os.path.exists(vocals_path):
            self.logger.error(f"Vocals file not found: {vocals_path}")
            return False
        
        if not os.path.exists(no_vocals_path):
            self.logger.error(f"No-vocals file not found: {no_vocals_path}")
            return False
        
        # Check file sizes (should be non-zero)
        vocals_size = os.path.getsize(vocals_path)
        no_vocals_size = os.path.getsize(no_vocals_path)
        
        if vocals_size == 0:
            self.logger.error("Vocals file is empty")
            return False
        
        if no_vocals_size == 0:
            self.logger.error("No-vocals file is empty")
            return False
        
        self.logger.info(f"Validation passed - Vocals: {vocals_size} bytes, No-vocals: {no_vocals_size} bytes")
        
        return True
