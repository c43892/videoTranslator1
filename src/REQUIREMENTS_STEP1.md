# Step 1: Audio/Video Track Separation - Detailed Requirements

## Overview

**Step Name:** Audio/Video Track Separation

**Purpose:** Extract and separate the video stream and audio stream from an input video file into independent tracks for downstream processing.

**Position in Pipeline:** First step - receives raw input video file

---

## 1. Functional Requirements

### 1.1 Input Specifications

**Primary Input:**
- Video file path (string)
- Source: Command line argument or configuration
- Must support absolute and relative paths

**Supported Input Formats:**
- **Any format supported by FFmpeg**, including but not limited to:
  - **Container Formats:** MP4, AVI, MKV, MOV, FLV, WMV, WEBM, MPEG, TS, 3GP, OGV, etc.
  - **Video Codecs:** H.264, H.265/HEVC, VP8, VP9, MPEG-4, MPEG-2, AV1, ProRes, etc.
  - **Audio Codecs:** AAC, MP3, AC3, PCM, Opus, Vorbis, FLAC, DTS, etc.
- FFmpeg's extensive format support ensures compatibility with virtually all common video files

**Input Validation:**
- File must exist at specified path
- File must be readable
- File must contain at least one video stream
- File must contain at least one audio stream
- File size limits (configurable, default: warn if >2GB)

### 1.2 Processing Requirements

**Core Functionality:**
1. Open and parse the input video container
2. Identify all available streams (video, audio, subtitle, etc.)
3. Extract the primary video stream without audio
4. Extract the complete audio stream(s)
5. Maintain original quality (lossless extraction preferred)
6. Preserve timing and synchronization information

**Stream Selection Logic:**
- **Video Stream:** 
  - If multiple video streams exist, select the first/primary stream
  - Alternatively, allow configuration to specify stream index
  - Skip thumbnail or preview streams
  
- **Audio Stream:**
  - If multiple audio streams exist, select the first/primary stream
  - Consider language preferences if metadata available
  - Should be configurable to select specific audio track

**Quality Preservation:**
- Prefer stream copy (no re-encoding) when possible
- If re-encoding necessary, maintain original resolution and bitrate
- Preserve frame rate, aspect ratio, and color space
- Maintain audio sample rate and bit depth

### 1.3 Output Specifications

**Output Files:**

1. **Video Track (without audio):**
   - Filename: `video_only.mp4` (or configurable format)
   - Contains: Video stream only, no audio
   - Codec: Same as input (stream copy) or H.264 if re-encoding needed
   - Metadata: Preserve original video metadata where applicable

2. **Audio Track (complete):**
   - Filename: `audio_full.wav` (or configurable format)
   - Contains: Complete audio from input video
   - Format: Uncompressed WAV (recommended) or original format
   - Channels: Preserve original (mono, stereo, 5.1, etc.)
   - Sample Rate: Preserve original (typically 44.1kHz or 48kHz)
   - Bit Depth: 16-bit minimum, 24-bit preferred

**Output Location:**
- Default: Same directory as input file
- Configurable: Custom output directory via configuration
- Create output directory if it doesn't exist

**Naming Convention:**
- Default names: `video_only.*`, `audio_full.*`
- Optional: Prefix with input filename (e.g., `myvideo_video_only.mp4`)
- Should be configurable

---

## 2. Interface Design

### 2.1 Abstract Base Class

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class SeparationResult:
    """Result of video/audio separation operation"""
    video_path: Path
    audio_path: Path
    metadata: Dict[str, Any]  # Duration, resolution, codec info, etc.
    success: bool
    error_message: Optional[str] = None

class VideoAudioSeparator(ABC):
    """Abstract base class for video/audio separation implementations"""
    
    @abstractmethod
    def separate(
        self,
        input_video_path: Path,
        output_dir: Optional[Path] = None,
        video_stream_index: int = 0,
        audio_stream_index: int = 0,
        **kwargs
    ) -> SeparationResult:
        """
        Separate video and audio tracks from input video file.
        
        Args:
            input_video_path: Path to input video file
            output_dir: Directory for output files (default: same as input)
            video_stream_index: Index of video stream to extract (default: 0)
            audio_stream_index: Index of audio stream to extract (default: 0)
            **kwargs: Implementation-specific parameters
            
        Returns:
            SeparationResult containing paths to separated files and metadata
            
        Raises:
            FileNotFoundError: If input video doesn't exist
            ValueError: If video format is unsupported or file is corrupted
            RuntimeError: If separation process fails
        """
        pass
    
    @abstractmethod
    def get_stream_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get information about streams in the video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing stream information:
            - video_streams: List of video stream metadata
            - audio_streams: List of audio stream metadata
            - duration: Total duration in seconds
            - format: Container format
        """
        pass
    
    @abstractmethod
    def validate_input(self, video_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate that input video file is suitable for processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
```

### 2.2 Configuration Schema

```yaml
step1_video_audio_separation:
  # Implementation to use (e.g., 'ffmpeg', 'moviepy', 'opencv')
  implementation: "ffmpeg"
  
  # Input settings
  input:
    video_stream_index: 0  # Which video stream to extract (default: first)
    audio_stream_index: 0  # Which audio stream to extract (default: first)
    
  # Output settings
  output:
    directory: null  # null = same as input, or specify path
    video_filename: "video_only.mp4"
    audio_filename: "audio_full.wav"
    include_input_name: false  # Prefix output with input filename
    
  # Quality settings
  quality:
    video_codec: "copy"  # "copy" for no re-encoding, or specific codec
    audio_format: "wav"  # wav, flac, mp3, etc.
    audio_sample_rate: null  # null = preserve original
    audio_bit_depth: 16  # 16 or 24
    
  # Processing options
  processing:
    verify_output: true  # Verify output files after creation
    overwrite_existing: false  # Overwrite if output files exist
    
  # Error handling
  error_handling:
    max_retries: 2
    timeout_seconds: 300  # Max time for separation process
```

---

## 3. Non-Functional Requirements

### 3.1 Performance

- **Processing Speed:** 
  - Stream copy mode: Should complete in < 5% of video duration
  - Re-encoding mode: Should complete in < 50% of video duration (depends on hardware)
  
- **Memory Usage:**
  - Should not load entire video into memory
  - Stream processing for large files
  - Maximum memory footprint: 500MB for typical operations

- **File Size Support:**
  - Must handle files up to 10GB
  - Should gracefully handle larger files with appropriate warnings

### 3.2 Reliability

- **Error Recovery:**
  - Detect and report corrupted video files
  - Handle incomplete video files gracefully
  - Retry logic for transient failures
  
- **Validation:**
  - Verify output files are not empty
  - Check output file duration matches input duration (±1 second tolerance)
  - Validate video/audio stream integrity

- **Logging:**
  - Log input file metadata (duration, resolution, codec)
  - Log processing start and completion times
  - Log any warnings or errors with context
  - Include stream information in logs

### 3.3 Usability

- **Progress Reporting:**
  - Optional progress callback for long operations
  - Percentage completion for re-encoding operations
  - Estimated time remaining for long processes

- **Error Messages:**
  - Clear, actionable error messages
  - Suggest solutions for common issues
  - Include file path and technical details for debugging

### 3.4 Compatibility

- **Cross-Platform:**
  - Must work on Windows, macOS, and Linux
  - Handle path separators correctly
  - No platform-specific dependencies in interface

- **Python Version:**
  - Python 3.9+ required
  - Type hints for better IDE support

---

## 4. Implementation Considerations

### 4.1 Recommended Tech Stack

**Option 1: FFmpeg (Recommended)**
- **Pros:** 
  - Industry standard, highly reliable
  - Fast stream copy without re-encoding
  - Supports virtually all formats
  - Excellent documentation
- **Cons:** 
  - External dependency (must be installed)
  - Requires subprocess calls
- **Library:** `ffmpeg-python` or direct subprocess calls

**Option 2: MoviePy**
- **Pros:** 
  - Pure Python interface
  - Simple API
  - Good for small to medium files
- **Cons:** 
  - Slower than FFmpeg
  - Higher memory usage
  - Still uses FFmpeg under the hood
- **Library:** `moviepy`

**Option 3: PyAV**
- **Pros:** 
  - Direct Python bindings to FFmpeg libraries
  - More control than subprocess calls
  - Better error handling
- **Cons:** 
  - More complex API
  - Steeper learning curve
- **Library:** `av`

### 4.2 Dependencies

```python
# Primary dependencies (for FFmpeg implementation)
ffmpeg-python>=0.2.0
pydub>=0.25.1  # Optional, for audio validation

# For validation and utilities
pathlib  # Built-in
dataclasses  # Built-in (Python 3.7+)
typing  # Built-in
```

### 4.3 Error Scenarios to Handle

| Error Scenario | Detection | Handling Strategy |
|----------------|-----------|-------------------|
| File not found | Path validation | Raise FileNotFoundError with clear message |
| Unsupported format | FFmpeg error output | Raise ValueError with supported formats list |
| No audio stream | Stream inspection | Raise ValueError indicating audio required |
| No video stream | Stream inspection | Raise ValueError indicating video required |
| Corrupted file | FFmpeg error during processing | Raise RuntimeError with diagnostic info |
| Disk space insufficient | Pre-check available space | Raise RuntimeError before processing |
| Permission denied | File access check | Raise PermissionError with path details |
| Process timeout | Timeout monitoring | Terminate process, raise TimeoutError |

---

## 5. Testing Requirements

### 5.1 Unit Tests

- Test input validation (valid/invalid paths, formats)
- Test stream info extraction
- Test configuration parsing
- Test error handling for each error scenario
- Mock external dependencies (FFmpeg) for isolated testing

### 5.2 Integration Tests

- Test with real video files of various formats (MP4, MKV, AVI, MOV, WEBM, FLV, etc.)
- Test with different video resolutions (360p, 720p, 1080p, 4K)
- Test with different audio configurations (mono, stereo, 5.1)
- Test with multi-audio track videos
- Test with very short (<5s) and long (>1hr) videos
- Test with various codecs (H.264, H.265, VP9, MPEG-4, etc.)
- Verify output file integrity

### 5.3 Test Files Needed

- Sample MP4 file (H.264 + AAC)
- Sample MKV file (H.265 + MP3)
- Sample AVI file (MPEG-4 + MP3)
- Sample MOV file (ProRes + PCM)
- Sample WEBM file (VP9 + Opus)
- Sample with multiple audio tracks
- Sample with different frame rates (24fps, 30fps, 60fps)
- Corrupted video file for error testing

### 5.4 Validation Criteria

**Pass Criteria:**
- ✓ Video output has no audio stream
- ✓ Audio output duration matches video duration (±1s)
- ✓ Video resolution and framerate preserved
- ✓ Audio sample rate preserved
- ✓ No quality degradation (if using stream copy)
- ✓ Files are playable in standard media players

---

## 6. Example Usage

```python
from pathlib import Path
from processors.video_separator import FFmpegVideoAudioSeparator

# Initialize separator
separator = FFmpegVideoAudioSeparator()

# Validate input
input_path = Path("input_videos/sample.mp4")
is_valid, error = separator.validate_input(input_path)
if not is_valid:
    print(f"Invalid input: {error}")
    exit(1)

# Get stream info (optional)
info = separator.get_stream_info(input_path)
print(f"Duration: {info['duration']}s")
print(f"Video streams: {len(info['video_streams'])}")
print(f"Audio streams: {len(info['audio_streams'])}")

# Perform separation
result = separator.separate(
    input_video_path=input_path,
    output_dir=Path("output"),
    video_stream_index=0,
    audio_stream_index=0
)

if result.success:
    print(f"Video saved to: {result.video_path}")
    print(f"Audio saved to: {result.audio_path}")
    print(f"Metadata: {result.metadata}")
else:
    print(f"Separation failed: {result.error_message}")
```

---

## 7. Open Questions

- **Audio Format:** Should we always convert to WAV, or preserve original format?
  - **Recommendation:** Convert to WAV for consistency in downstream processing
  
- **Multiple Audio Tracks:** How to handle videos with multiple audio tracks?
  - **Recommendation:** Extract first track by default, allow configuration to select specific track
  
- **Subtitles:** Should we also extract subtitle tracks for potential future use?
  - **Recommendation:** Not required for MVP, can be added later
  
- **Stream Copy vs Re-encoding:** When should we force re-encoding?
  - **Recommendation:** Always try stream copy first, re-encode only on error

---

## 8. Success Metrics

**Step 1 is successful when:**
1. ✓ Can process any video format supported by FFmpeg (MP4, MKV, AVI, MOV, WEBM, FLV, etc.)
2. ✓ Produces valid video file without audio
3. ✓ Produces valid audio file matching video duration
4. ✓ Completes in reasonable time (< 10% of video duration)
5. ✓ No quality loss in output files
6. ✓ Handles errors gracefully with clear messages
7. ✓ Works with videos from 10 seconds to 2+ hours
8. ✓ Implementation can be swapped without changing pipeline

---

**Document Version:** 1.0  
**Last Updated:** November 16, 2025  
**Status:** Ready for Implementation  
**Dependencies:** None (first step in pipeline)
