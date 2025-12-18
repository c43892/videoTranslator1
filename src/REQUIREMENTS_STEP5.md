# Step 5: Audio Clipping (Original) - Detailed Requirements

## Overview

**Step Name:** Audio Clipping (Original)

**Purpose:** Extract individual audio clips from the vocal track based on timestamps in the original SRT file, creating separate audio files for each sentence/subtitle entry.

**Position in Pipeline:** Fifth step - receives vocal track from Step 2 and SRT file from Step 3

---

## 1. Functional Requirements

### 1.1 Input Specifications

**Primary Inputs:**

1. **Vocal Audio File:**
   - File: `vocals.[format]` (output from Step 2)
   - Format: Any audio format supported by FFmpeg (WAV, MP3, FLAC, etc.)
   - Content: Isolated vocal track without background music/noise
   - Source: Step 2 output directory

2. **Original SRT File:**
   - File: `SRT_original.srt` (output from Step 3)
   - Format: Standard SRT (SubRip Subtitle) format
   - Encoding: UTF-8
   - Content: Timestamped transcription of the vocal track

**Input Validation:**
- Both input files must exist and be readable
- SRT file must be valid format with at least one subtitle entry
- Audio file must be a valid audio format
- SRT timestamps must be within audio duration bounds

### 1.2 Processing Requirements

**Core Functionality:**
1. Parse the SRT file to extract all subtitle entries with timestamps
2. For each subtitle entry (N):
   - Extract start time and end time
   - Use FFmpeg to clip the audio segment from `vocals.[format]`
   - Save as `original_clip_{N}.mp3`
3. Maintain exact timing as specified in SRT
4. Preserve audio quality during extraction

**SRT Parsing:**
- Read and parse all entries in `SRT_original.srt`
- Extract subtitle number, start timestamp, end timestamp, and text
- Handle various SRT formats (with/without BOM, different line endings)
- Validate timestamp format: `HH:MM:SS,mmm --> HH:MM:SS,mmm`

**Audio Clipping with FFmpeg:**
- Use FFmpeg's `-ss` (start time) and `-to` (end time) or `-t` (duration) parameters
- Employ `-c:a libmp3lame` for MP3 encoding (or `-c:a copy` if input is already MP3)
- Set consistent bitrate: 128kbps or 192kbps (configurable)
- Use `-avoid_negative_ts make_zero` to handle timestamp edge cases

**Timestamp Conversion:**
- Convert SRT format `HH:MM:SS,mmm` to seconds (float) for FFmpeg
- Example: `00:01:23,456` → `83.456` seconds
- Maintain millisecond precision

**Edge Case Handling:**
1. **Very Short Clips (< 0.1 seconds):**
   - Still extract but log warning
   - Ensure minimum clip length of at least 1 frame

2. **Overlapping Timestamps:**
   - Extract as specified in SRT (overlaps allowed)
   - Each clip is independent

3. **Gaps in Timeline:**
   - Only extract specified segments
   - Gaps between clips are intentional

4. **End Time Beyond Audio Duration:**
   - Clip to actual audio end
   - Log warning

5. **Non-dialogue Events (e.g., `[[ laughing ]]`):**
   - Still extract audio clip for these entries
   - Naming follows same pattern: `original_clip_{N}.mp3`

### 1.3 Output Specifications

**Output Files:**
- **Filename Pattern:** `original_clip_{N}.mp3`
  - `{N}` = Subtitle entry number from SRT (1-based index)
  - Examples: `original_clip_1.mp3`, `original_clip_2.mp3`, ..., `original_clip_157.mp3`

- **Format:** MP3 (or configurable)
- **Codec:** LAME MP3 encoder
- **Bitrate:** 192kbps (configurable, range: 128-320kbps)
- **Sample Rate:** Preserve original or 44100Hz/48000Hz
- **Channels:** Preserve original (typically mono or stereo)

**Output Location:**
- Default: `output/step5/` or configurable directory
- Create output directory if it doesn't exist
- Organize by job/video name if processing multiple videos

**Metadata Output:**
- Optional: Generate `clipping_metadata.json` containing:
  ```json
  {
    "source_audio": "vocals.wav",
    "source_srt": "SRT_original.srt",
    "total_clips": 157,
    "clips": [
      {
        "clip_number": 1,
        "filename": "original_clip_1.mp3",
        "start_time": "00:00:01,000",
        "end_time": "00:00:04,500",
        "start_seconds": 1.0,
        "end_seconds": 4.5,
        "duration": 3.5,
        "text": "This is the first sentence."
      },
      ...
    ]
  }
  ```

---

## 2. Interface Design

### 2.1 Abstract Base Class

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

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
class ClippingResult:
    """Result of audio clipping operation"""
    clips: List[AudioClip]
    total_clips: int
    source_audio: Path
    source_srt: Path
    output_dir: Path
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

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
            
        Format:
            [
                {
                    'number': 1,
                    'start': '00:00:01,000',
                    'end': '00:00:04,500',
                    'text': 'First sentence.'
                },
                ...
            ]
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
            
        Example:
            '00:01:23,456' -> 83.456
        """
        pass
```

### 2.2 FFmpeg Implementation

```python
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
        pass
    
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
        pass
    
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
        
        FFmpeg command structure:
            ffmpeg -i input.wav -ss START -to END -c:a libmp3lame -b:a BITRATE output.mp3
        
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
        pass
    
    def parse_srt(self, srt_path: Path) -> List[Dict[str, Any]]:
        """Parse SRT file using regex or line-by-line parsing"""
        pass
    
    def srt_time_to_seconds(self, srt_time: str) -> float:
        """Convert SRT timestamp format to seconds"""
        # Example: "00:01:23,456" -> 83.456
        pass
```

---

## 3. Configuration

### 3.1 Configuration File Support

```yaml
# config/audio_clipper_config.yaml
audio_clipper:
  implementation: "ffmpeg"  # Currently only ffmpeg supported
  
  ffmpeg:
    executable_path: "ffmpeg"  # Or full path: /usr/bin/ffmpeg
    
  output:
    format: "mp3"              # mp3, wav, flac, aac
    bitrate: "192k"            # 128k, 192k, 256k, 320k
    sample_rate: null          # null = preserve original, or specify: 44100, 48000
    channels: null             # null = preserve original, or specify: 1 (mono), 2 (stereo)
    
  naming:
    pattern: "original_clip_{N}.{ext}"  # {N} = clip number, {ext} = format
    
  processing:
    min_clip_duration: 0.01    # Minimum clip duration in seconds
    max_clip_duration: 60.0    # Maximum expected clip duration (for validation)
    parallel_processing: false # Future: process clips in parallel
    
  logging:
    log_level: "INFO"          # DEBUG, INFO, WARNING, ERROR
    save_metadata: true        # Save clipping_metadata.json
```

---

## 4. Error Handling

### 4.1 Input Validation Errors
- **Missing Files:** Clear error if `vocals.[format]` or `SRT_original.srt` not found
- **Invalid SRT:** Detailed error messages for malformed SRT format
- **Empty SRT:** Handle case where SRT has no entries

### 4.2 Processing Errors
- **FFmpeg Errors:** Capture and log FFmpeg stderr output
- **Timestamp Out of Bounds:** Warn and adjust if end time exceeds audio duration
- **Invalid Timestamps:** Skip entries with malformed timestamps, log warning

### 4.3 Recovery and Logging
- **Continue on Error:** If one clip fails, continue with others
- **Summary Report:** Log total clips, successful, failed
- **Detailed Logs:** Save logs for debugging (optional)

### 4.4 Exit Codes
- `0`: Success - all clips extracted
- `1`: Partial success - some clips failed
- `2`: Complete failure - no clips extracted

---

## 5. Testing Requirements

### 5.1 Unit Tests
- SRT parsing with various formats
- Timestamp conversion accuracy
- Edge cases: very short clips, overlapping timestamps
- Error handling: missing files, invalid SRT

### 5.2 Integration Tests
- Full pipeline: valid SRT + audio → clips
- Different audio formats (WAV, MP3, FLAC)
- Different SRT structures (with/without BOM, Unix/Windows line endings)

### 5.3 Test Cases

**Test Case 1: Standard SRT**
- Input: 10-entry SRT with normal speech segments
- Expected: 10 MP3 files, correct duration for each

**Test Case 2: Non-dialogue Events**
- Input: SRT with `[[ laughing ]]`, `[[ music ]]` entries
- Expected: Clips extracted for all entries including events

**Test Case 3: Very Short Clips**
- Input: SRT with 0.1-second segments
- Expected: Clips created, warning logged

**Test Case 4: Edge Timestamps**
- Input: SRT with clip ending beyond audio duration
- Expected: Clip trimmed to audio end, warning logged

**Test Case 5: Large SRT**
- Input: SRT with 500+ entries
- Expected: All clips extracted, reasonable processing time

---

## 6. Dependencies

### 6.1 Required Software
- **FFmpeg:** Version 4.0 or higher
  - Must be installed and accessible via PATH or configured path
  - Required codecs: libmp3lame (for MP3 output)

### 6.2 Python Libraries
```
pydub>=0.25.1          # Optional: alternative audio processing
```

---

## 7. Performance Considerations

### 7.1 Optimization
- Use FFmpeg's `-ss` before `-i` for faster seeking (input seeking)
- Consider parallel processing for large SRT files (future enhancement)
- Avoid re-encoding when possible (use `-c:a copy` if format matches)

### 7.2 Expected Performance
- **Small SRT (10-50 clips):** < 10 seconds
- **Medium SRT (50-200 clips):** < 1 minute
- **Large SRT (200-500 clips):** < 3 minutes

*Times are estimates on modern hardware, actual time depends on audio duration and system*

---

## 8. Usage Example

### 8.1 Command Line (Future)
```bash
python -m src.processors.audio_clipper \
    --vocals path/to/vocals.wav \
    --srt path/to/SRT_original.srt \
    --output path/to/output/step5/ \
    --format mp3 \
    --bitrate 192k
```

### 8.2 Python API
```python
from pathlib import Path
from src.processors.audio_clipper import FFmpegAudioClipper

# Initialize clipper
clipper = FFmpegAudioClipper(
    ffmpeg_path="ffmpeg",
    default_bitrate="192k"
)

# Perform clipping
result = clipper.clip_audio(
    vocal_audio_path=Path("output/step2/vocals.wav"),
    srt_path=Path("output/step3/SRT_original.srt"),
    output_dir=Path("output/step5/"),
    output_format="mp3",
    bitrate="192k"
)

# Check results
if result.success:
    print(f"Successfully extracted {result.total_clips} clips")
    for clip in result.clips:
        print(f"  {clip.filename}: {clip.duration:.2f}s - {clip.text[:50]}")
else:
    print(f"Clipping failed: {result.error_message}")
```

---

## 9. Output Directory Structure

```
output/
└── step5/
    ├── original_clip_1.mp3
    ├── original_clip_2.mp3
    ├── original_clip_3.mp3
    ├── ...
    ├── original_clip_N.mp3
    └── clipping_metadata.json  # Optional metadata file
```

---

## 10. FFmpeg Command Reference

### 10.1 Basic Clipping Command
```bash
ffmpeg -i vocals.wav -ss 1.0 -to 4.5 -c:a libmp3lame -b:a 192k original_clip_1.mp3
```

**Parameters:**
- `-i vocals.wav`: Input file
- `-ss 1.0`: Start time (seconds)
- `-to 4.5`: End time (seconds)
- `-c:a libmp3lame`: Use MP3 encoder
- `-b:a 192k`: Audio bitrate 192 kbps
- `original_clip_1.mp3`: Output file

### 10.2 Alternative: Using Duration
```bash
ffmpeg -i vocals.wav -ss 1.0 -t 3.5 -c:a libmp3lame -b:a 192k original_clip_1.mp3
```
- `-t 3.5`: Duration (instead of end time)

### 10.3 High-Quality Settings
```bash
ffmpeg -i vocals.wav -ss 1.0 -to 4.5 -c:a libmp3lame -b:a 320k -q:a 0 original_clip_1.mp3
```
- `-b:a 320k`: Maximum MP3 bitrate
- `-q:a 0`: Highest quality setting

### 10.4 WAV Output (Lossless)
```bash
ffmpeg -i vocals.wav -ss 1.0 -to 4.5 -c:a pcm_s16le original_clip_1.wav
```
- `-c:a pcm_s16le`: 16-bit PCM WAV format

---

## 11. Success Criteria

The Step 5 implementation is successful when:

1. ✓ All subtitle entries from SRT are processed and clips extracted
2. ✓ Clip filenames follow the exact pattern: `original_clip_{N}.mp3`
3. ✓ Clip timing matches SRT timestamps precisely
4. ✓ Audio quality is preserved (no noticeable degradation)
5. ✓ Edge cases are handled gracefully (short clips, boundary issues)
6. ✓ Non-dialogue events (`[[ laughing ]]`, etc.) are clipped correctly
7. ✓ Processing completes for large SRT files (500+ entries)
8. ✓ Clear error messages for failures
9. ✓ Metadata output is accurate and complete
10. ✓ All clips are playable and audible

---

## 12. Future Enhancements

### 12.1 Potential Improvements
- Parallel processing for faster extraction of large SRT files
- Audio normalization (volume leveling) during clipping
- Fade in/out at clip boundaries to reduce pops/clicks
- Support for multiple audio tracks (different languages)
- Automatic silence trimming at clip boundaries
- Progress bar for long-running operations

### 12.2 Alternative Implementations
- **pydub-based clipper:** Python-only implementation without FFmpeg
- **Cloud-based clipper:** Use cloud audio processing services
- **Hardware-accelerated:** GPU acceleration for faster processing

---

**Document Version:** 1.0  
**Last Updated:** November 16, 2025  
**Status:** Draft - Ready for Implementation
