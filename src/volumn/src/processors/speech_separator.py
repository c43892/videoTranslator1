import logging
import os
import requests
import shutil
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import soundfile for chunking
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logger.warning("soundfile not available - large file chunking disabled")

# Load configuration
def load_config():
    """Load speech separator configuration from YAML file"""
    config_path = Path(__file__).parent.parent.parent / 'config' / 'speech_separator_config.yaml'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration if file doesn't exist
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {
            'chunking': {
                'threshold_mb': 30,
                'chunk_duration_sec': 600,
                'chunk_format': 'mp3',
                'mp3_bitrate': '320k',
                'flac_compression': 5
            }
        }

# Load config at module level
CONFIG = load_config()
CHUNK_THRESHOLD_MB = CONFIG['chunking']['threshold_mb']
CHUNK_DURATION_SEC = CONFIG['chunking']['chunk_duration_sec']
CHUNK_FORMAT = CONFIG['chunking']['chunk_format']
MP3_BITRATE = CONFIG['chunking']['mp3_bitrate']
FLAC_COMPRESSION = CONFIG['chunking']['flac_compression']

class SpeechSeparator:
    """
    Separates speech from non-speech (noise, music, etc.) in an audio file 
    using an external HTTP service. Automatically chunks large files.
    """
    
    def __init__(self, api_url: str = "https://dfn-service-105532883168.us-central1.run.app"):
        """
        Initialize SpeechSeparator.
        
        Args:
            api_url: Base URL of the speech separation service
        """
        self.api_url = api_url.rstrip('/')
        
    def process(self, input_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Process audio to separate speech.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save output files
            
        Returns:
            Tuple containing (speech_path, non_speech_path)
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output paths
        speech_path = output_dir / "vocals_speech.wav"
        non_speech_path = output_dir / "vocals_no_speech.wav"
        
        # Check service health
        try:
            health_resp = requests.get(f"{self.api_url}/health", timeout=60)
            if health_resp.status_code != 200:
                logger.warning(f"Speech separation service health check failed: {health_resp.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to speech separation service at {self.api_url}: {e}")
            raise ConnectionError(f"Speech separation service unavailable at {self.api_url}")

        # Check file size
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        
        # Use chunking for large files
        if file_size_mb > CHUNK_THRESHOLD_MB and HAS_SOUNDFILE:
            logger.info(f"File size ({file_size_mb:.1f} MB) exceeds {CHUNK_THRESHOLD_MB}MB limit. Using client-side chunking...")
            logger.info(f"Chunking config: {CHUNK_DURATION_SEC}s chunks, {CHUNK_FORMAT} format" + 
                       (f" @ {MP3_BITRATE}" if CHUNK_FORMAT == 'mp3' else ""))
            return self._process_chunked(input_path, output_dir, speech_path, non_speech_path)
        elif file_size_mb > 30 and not HAS_SOUNDFILE:
            logger.warning(f"File is {file_size_mb:.1f} MB but soundfile not available. Attempting direct upload (may fail)...")
        
        # Process single file
        return self._process_single(input_path, output_dir, speech_path, non_speech_path)
    
    def _process_single(self, input_path: Path, output_dir: Path, speech_path: Path, non_speech_path: Path) -> Tuple[str, str]:
        """Process a single file without chunking"""
        process_url = f"{self.api_url}/process"
        logger.info(f"Sending {input_path.name} to speech separation service...")
        
        with open(input_path, 'rb') as f:
            files = {'audio': (input_path.name, f, 'audio/mpeg')}
            data = {'timeout': 36000}
            
            try:
                response = requests.post(process_url, files=files, data=data, timeout=36000)
                
                if response.status_code != 200:
                    raise RuntimeError(f"Service returned error: {response.text}")
                
                result = response.json()
                request_id = result.get('request_id')
                
                if not request_id:
                    raise ValueError("Service did not return request_id")
                
                # Download files
                speech_url = f"{self.api_url}/download/{request_id}/speech.wav"
                non_speech_url = f"{self.api_url}/download/{request_id}/non_speech.wav"
                
                self._download_file(speech_url, speech_path)
                self._download_file(non_speech_url, non_speech_path)
                
                # Cleanup
                requests.post(f"{self.api_url}/cleanup/{request_id}")
                
                return str(speech_path), str(non_speech_path)
                
            except Exception as e:
                logger.error(f"Speech separation failed: {e}")
                raise
    
    def _process_chunked(self, input_path: Path, output_dir: Path, speech_path: Path, non_speech_path: Path) -> Tuple[str, str]:
        """Process large file by chunking"""
        logger.info(f"Reading audio file: {input_path}...")
        audio, sr = sf.read(str(input_path))
        
        # Calculate chunks
        chunk_samples = int(CHUNK_DURATION_SEC * sr)
        total_samples = len(audio)
        num_chunks = int(np.ceil(total_samples / chunk_samples))
        
        logger.info(f"Splitting into {num_chunks} chunks of {CHUNK_DURATION_SEC}s each...")
        
        processed_speech = []
        processed_nonspeech = []
        
        temp_dir = tempfile.mkdtemp()
        try:
            for i in range(num_chunks):
                start = i * chunk_samples
                end = min((i + 1) * chunk_samples, total_samples)
                chunk_audio = audio[start:end]
                
                percentage = ((i + 1) / num_chunks) * 100
                logger.info(f"Processing chunk {i+1}/{num_chunks} ({percentage:.1f}%) - {start/sr:.1f}s to {end/sr:.1f}s")
                print(f"  [Step 2.5] Chunk {i+1}/{num_chunks} ({percentage:.1f}%)")
                
                # Save chunk in configured format
                chunk_ext = CHUNK_FORMAT
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.{chunk_ext}")
                
                # Write with format-specific settings
                if CHUNK_FORMAT == 'mp3':
                    sf.write(chunk_path, chunk_audio, sr, format='MP3', subtype=MP3_BITRATE)
                elif CHUNK_FORMAT == 'flac':
                    sf.write(chunk_path, chunk_audio, sr, format='FLAC', compression=FLAC_COMPRESSION)
                else:  # wav
                    sf.write(chunk_path, chunk_audio, sr)
                
                # Process chunk
                chunk_output_dir = os.path.join(temp_dir, f"out_{i}")
                chunk_speech = Path(chunk_output_dir) / "vocals_speech.wav"
                chunk_nonspeech = Path(chunk_output_dir) / "vocals_no_speech.wav"
                
                self._process_single(Path(chunk_path), Path(chunk_output_dir), chunk_speech, chunk_nonspeech)
                
                # Read results
                s_data, _ = sf.read(str(chunk_speech))
                ns_data, _ = sf.read(str(chunk_nonspeech))
                
                processed_speech.append(s_data)
                processed_nonspeech.append(ns_data)
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Merge results
        logger.info("Merging chunks...")
        final_speech = np.concatenate(processed_speech)
        final_nonspeech = np.concatenate(processed_nonspeech)
        
        sf.write(str(speech_path), final_speech, sr)
        sf.write(str(non_speech_path), final_nonspeech, sr)
        
        logger.info(f"✓ All chunks processed and merged!")
        
        return str(speech_path), str(non_speech_path)
                
    def _download_file(self, url: str, output_path: Path):
        """Download a file from URL."""
        logger.info(f"Downloading processed audio to {output_path.name}...")
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
