"""
Subtitle Extractor processor
Extracts embedded soft subtitles from video files.
"""

import logging
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .base import ProcessorResult

logger = logging.getLogger(__name__)

class SubtitleExtractor:
    """
    Extracts embedded subtitles from video files using FFmpeg.
    """
    
    def __init__(self):
        pass
        
    def get_subtitle_streams(self, video_path: Path) -> List[Dict]:
        """
        Get list of subtitle streams in the video.
        
        Args:
            video_path: Path to input video
            
        Returns:
            List of dicts containing stream info (index, language, title, codec_name)
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 's',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            streams = []
            for stream in data.get('streams', []):
                tags = stream.get('tags', {})
                streams.append({
                    'index': stream.get('index'),
                    'codec_name': stream.get('codec_name'),
                    'language': tags.get('language', 'unknown'),
                    'title': tags.get('title', 'unknown')
                })
            
            return streams
            
        except Exception as e:
            logger.warning(f"Failed to probe subtitle streams: {e}")
            return []

    def extract_subtitle(
        self, 
        video_path: Path, 
        stream_index: int, 
        output_path: Path
    ) -> bool:
        """
        Extract a specific subtitle stream to SRT file.
        
        Args:
            video_path: Path to input video
            stream_index: FFmpeg stream index
            output_path: Path to save SRT file
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-map', f'0:{stream_index}',
            '-y',  # Overwrite
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract subtitle stream {stream_index}: {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"Error extracting subtitle: {e}")
            return False

    def process(
        self, 
        video_path: Path, 
        output_path: Path, 
        target_language: Optional[str] = None
    ) -> bool:
        """
        Smart process: Find best matching subtitle stream and extract it.
        
        Args:
            video_path: Path to input video
            output_path: Path to save SRT file
            target_language: Preferred ISO language code (e.g. 'eng', 'chi')
            
        Returns:
            True if a subtitle was extracted, False otherwise
        """
        streams = self.get_subtitle_streams(video_path)
        
        if not streams:
            logger.info("No embedded subtitles found.")
            return False
            
        logger.info(f"Found {len(streams)} embedded subtitle streams:")
        for s in streams:
            logger.info(f"  Stream {s['index']}: Lang={s['language']}, Title={s['title']}, Codec={s['codec_name']}")
            
        selected_index = None
        
        # Strategy:
        # 1. If target_language is provided, look for exact match
        if target_language:
            for s in streams:
                # Simple check, assumes ISO 3-letter codes mostly
                if target_language.lower() in s['language'].lower() or s['language'].lower() in target_language.lower():
                    selected_index = s['index']
                    logger.info(f"Selected stream {selected_index} matching language '{target_language}'")
                    break
        
        # 2. If no match or no target language, pick the first one (usually default)
        if selected_index is None:
            selected_index = streams[0]['index']
            logger.info(f"Selected default stream {selected_index}")
            
        # Extract
        logger.info(f"Extracting stream {selected_index} to {output_path}...")
        success = self.extract_subtitle(video_path, selected_index, output_path)
        
        if success:
            logger.info("Extraction successful.")
        else:
            logger.warning("Extraction failed.")
            
        return success
