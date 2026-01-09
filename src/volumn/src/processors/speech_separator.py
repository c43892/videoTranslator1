import logging
import os
import requests
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class SpeechSeparator:
    """
    Separates speech from non-speech (noise, music, etc.) in an audio file 
    using an external HTTP service.
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

        # Prepare request
        process_url = f"{self.api_url}/process"
        logger.info(f"Sending {input_path.name} to speech separation service...")
        
        with open(input_path, 'rb') as f:
            files = {'audio': (input_path.name, f, 'audio/mpeg')}
            data = {'timeout': 36000}  # Increase timeout to 10 hours
            
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
                
    def _download_file(self, url: str, output_path: Path):
        """Download a file from URL."""
        logger.info(f"Downloading processed audio to {output_path.name}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
