# Step 8: Final Video Assembly - Detailed Requirements

## Overview
Combine the original video track (without audio) with the translated audio track to create the final translated video output.

---

## Input Files

### Required Inputs:
1. **Video Track** (from Step 1):
   - `video_only.[format]` - Original video without audio track
   - Contains all visual content from the original video

2. **Translated Audio** (from Step 7):
   - `audio_translated_full.mp3` - Complete translated audio track
   - Contains translated vocals mixed with background audio

### Optional Inputs:
3. **Original Video** (optional reference):
   - Original input video file
   - Used for metadata extraction and format reference

---

## Process Flow

### 1. Validate Input Files
- Verify `video_only.[format]` exists and is readable
- Verify `audio_translated_full.mp3` exists and is readable
- Check video file integrity (not corrupted)
- Check audio file integrity (not corrupted)

### 2. Extract Video Metadata
- Get video properties:
  - Duration
  - Resolution (width × height)
  - Frame rate (FPS)
  - Video codec
  - Aspect ratio
  - Color space

### 3. Verify Audio-Video Duration Matching
- Get audio duration from `audio_translated_full.mp3`
- Get video duration from `video_only.[format]`
- Compare durations:
  - **If durations match (within ±100ms tolerance):** Proceed normally
  - **If audio is shorter:** Pad audio with silence at the end
  - **If audio is longer:** Trim audio to match video duration
  - Log any duration mismatches as warnings

### 4. Combine Video and Audio
- Merge video track with audio track using FFmpeg
- Ensure perfect synchronization from start
- Maintain original video quality and properties
- Use appropriate codecs for output format

### 5. Export Final Video
- Save to `video_translated.[format]`
- Apply output format configuration
- Preserve video quality
- Embed metadata if applicable

---

## Output Files

### Primary Output:
- **`video_translated.mp4`** (or specified format)
  - Complete video with translated audio
  - Maintains original video quality and resolution
  - Perfect audio-video synchronization

### Metadata Output:
- **`assembly_metadata.json`**
  ```json
  {
    "video_source": "video_only.mp4",
    "audio_source": "audio_translated_full.mp3",
    "output_file": "video_translated.mp4",
    "video_properties": {
      "duration": 59.03,
      "resolution": "1920x1080",
      "fps": 30.0,
      "codec": "h264",
      "bitrate": "5000k"
    },
    "audio_properties": {
      "duration": 59.03,
      "sample_rate": 44100,
      "channels": 2,
      "bitrate": "192k"
    },
    "duration_match": true,
    "adjustments": [],
    "warnings": [],
    "timestamp": "2025-11-17T16:57:30Z"
  }
  ```

---

## Technical Requirements

### Video Processing:

#### FFmpeg Command Structure:
```bash
ffmpeg -i video_only.mp4 -i audio_translated_full.mp3 \
  -c:v copy \
  -c:a aac \
  -b:a 192k \
  -map 0:v:0 -map 1:a:0 \
  -shortest \
  video_translated.mp4
```

**Parameters Explanation:**
- `-i video_only.mp4`: Input video file
- `-i audio_translated_full.mp3`: Input audio file
- `-c:v copy`: Copy video stream without re-encoding (preserves quality)
- `-c:a aac`: Encode audio to AAC (widely compatible)
- `-b:a 192k`: Audio bitrate 192 kbps
- `-map 0:v:0`: Map video from first input
- `-map 1:a:0`: Map audio from second input
- `-shortest`: Trim to shortest input duration

#### Quality Preservation:
- **Video**: Use copy codec (`-c:v copy`) to avoid re-encoding
- **Audio**: Use high-quality AAC encoding (192k or higher)
- **No quality loss** from original video

#### Format Support:
- **Primary output format**: MP4 (H.264 video + AAC audio)
- **Alternative formats**: 
  - MKV (Matroska container)
  - AVI (legacy compatibility)
  - MOV (QuickTime)
- **Codec flexibility**: Support H.264, H.265/HEVC

### Duration Handling:

#### Duration Mismatch Strategies:

**Audio Shorter than Video:**
```python
# Pad audio with silence
ffmpeg -i audio_translated_full.mp3 \
  -af "apad=whole_dur={video_duration}" \
  audio_padded.mp3
```

**Audio Longer than Video:**
```python
# Trim audio to video duration
ffmpeg -i audio_translated_full.mp3 \
  -t {video_duration} \
  audio_trimmed.mp3
```

**Tolerance:** ±100ms considered acceptable match

### Metadata Preservation:
- Copy metadata from original video where applicable
- Add custom metadata tags:
  - Processing timestamp
  - Translation language
  - Software version
- Preserve original creation date if possible

---

## Implementation Structure

### Class: `VideoAssembler`

**Location:** `src/processors/video_assembler.py`

**Base Class:** Inherits from `BaseProcessor`

**Key Methods:**

```python
class VideoAssembler(BaseProcessor):
    def process(self, input_data: dict) -> dict:
        """
        Main processing method
        
        Args:
            input_data: {
                'video_file': path to video_only.[format],
                'audio_file': path to audio_translated_full.mp3,
                'output_dir': path for output files,
                'output_format': video format (default: 'mp4'),
                'video_bitrate': video bitrate (default: None = auto),
                'audio_bitrate': audio bitrate (default: '192k')
            }
            
        Returns:
            {
                'video_translated': path to output video,
                'metadata_file': path to metadata JSON,
                'metadata': assembly metadata dict
            }
        """
        
    def get_video_info(self, video_path: str) -> dict:
        """Get video file properties using ffprobe"""
        
    def get_audio_info(self, audio_path: str) -> dict:
        """Get audio file properties using ffprobe"""
        
    def check_duration_match(self, video_duration: float, audio_duration: float, tolerance: float = 0.1) -> tuple:
        """
        Check if video and audio durations match
        
        Returns:
            (match: bool, difference: float, action: str)
        """
        
    def adjust_audio_duration(self, audio_path: str, target_duration: float, output_path: str) -> bool:
        """Pad or trim audio to match target duration"""
        
    def combine_video_audio(self, video_path: str, audio_path: str, output_path: str, **kwargs) -> bool:
        """Combine video and audio tracks using FFmpeg"""
```

---

## Error Handling

### Critical Errors (halt processing):
1. **Input file not found**
   - Error: "Video file not found: {path}"
   - Error: "Audio file not found: {path}"

2. **File corruption**
   - Error: "Video file is corrupted or unreadable"
   - Error: "Audio file is corrupted or unreadable"

3. **FFmpeg failure**
   - Error: "Failed to combine video and audio: {stderr}"

### Warnings (log but continue):
1. **Duration mismatch**
   - Warning: "Audio duration ({audio_dur}s) differs from video duration ({video_dur}s) by {diff}s"
   - Action: Auto-adjust based on configuration

2. **Metadata issues**
   - Warning: "Could not extract video metadata: {error}"
   - Action: Use defaults and continue

3. **Format compatibility**
   - Warning: "Output format {format} may have limited compatibility"

---

## Configuration Options

### Output Settings:
```yaml
step8:
  output_format: "mp4"           # mp4, mkv, avi, mov
  video_codec: "copy"            # copy (no re-encode) or h264, h265
  audio_codec: "aac"             # aac, mp3, opus
  audio_bitrate: "192k"          # 128k, 192k, 256k, 320k
  video_bitrate: null            # null = auto, or specify like "5000k"
  duration_tolerance: 0.1        # seconds (±100ms)
  auto_adjust_duration: true     # automatically pad/trim audio
  preserve_metadata: true        # copy metadata from original
```

---

## Testing Requirements

### Unit Tests:
1. **Video info extraction**
   - Test with various video formats
   - Test with different codecs
   - Test with corrupted files

2. **Audio info extraction**
   - Test with MP3, WAV, AAC formats
   - Test duration parsing accuracy

3. **Duration matching**
   - Test exact match
   - Test audio shorter by various amounts
   - Test audio longer by various amounts
   - Test tolerance thresholds

4. **Audio adjustment**
   - Test padding with silence
   - Test trimming audio
   - Verify no quality loss

### Integration Tests:
1. **Complete assembly**
   - Test with real video and audio files
   - Verify output plays correctly
   - Check audio-video synchronization
   - Validate metadata

2. **Format compatibility**
   - Test with different input video formats
   - Test different output format configurations

3. **Error scenarios**
   - Missing files
   - Corrupted files
   - Extreme duration mismatches

---

## Performance Considerations

### Optimization:
- **Video copy mode**: No re-encoding preserves quality and is fast
- **Hardware acceleration**: Use GPU acceleration if available
  - `-hwaccel cuda` (NVIDIA)
  - `-hwaccel videotoolbox` (macOS)
  - `-hwaccel qsv` (Intel Quick Sync)

### Progress Tracking:
- Monitor FFmpeg progress output
- Estimate completion time based on video duration
- Display progress percentage to user

---

## Dependencies

### Required:
- **FFmpeg**: Video/audio processing
- **Python libraries**:
  - `subprocess`: FFmpeg execution
  - `json`: Metadata handling
  - `pathlib`: File path operations

### Optional:
- **ffmpeg-python**: Python wrapper for FFmpeg (alternative to subprocess)

---

## Command-Line Interface

### Test Script: `test_step8.py`

```bash
python test_step8.py <video_only> <audio_translated> <output_dir> [options]

Arguments:
  video_only          Path to video_only.[format] (from Step 1)
  audio_translated    Path to audio_translated_full.mp3 (from Step 7)
  output_dir          Directory for output files

Options:
  --output-format     Output video format (default: mp4)
  --video-codec       Video codec (default: copy)
  --audio-codec       Audio codec (default: aac)
  --audio-bitrate     Audio bitrate (default: 192k)
  --verbose           Enable verbose logging
```

**Example:**
```bash
python test_step8.py \
  test/output/test02/step1/video_only.mp4 \
  test/output/test02/step7/audio_translated_full.mp3 \
  test/output/test02/step8 \
  --output-format mp4 \
  --audio-bitrate 192k \
  --verbose
```

---

## Success Criteria

Step 8 is successful when:
1. ✓ Video and audio are combined without errors
2. ✓ Output video plays correctly in standard media players
3. ✓ Audio-video synchronization is perfect (no drift)
4. ✓ Video quality matches original (no re-encoding artifacts)
5. ✓ Audio quality is high (no distortion or clipping)
6. ✓ Duration mismatches are handled gracefully
7. ✓ Metadata is preserved or added appropriately
8. ✓ Output file size is reasonable (not bloated)

---

## Notes

### Sync Considerations:
- Audio should start exactly at frame 0 of the video
- No delay or offset between audio and video
- Constant sync throughout entire duration

### Format Recommendations:
- **Best compatibility**: MP4 with H.264 video and AAC audio
- **Best quality**: MKV with H.265 video and FLAC/PCM audio
- **Legacy support**: AVI with older codecs

### Future Enhancements:
1. Support for multiple audio tracks
2. Subtitle embedding
3. Chapter markers
4. Thumbnail generation
5. Batch processing multiple videos
6. Cloud storage upload

---

**Document Version:** 1.0  
**Last Updated:** November 17, 2025  
**Status:** Final
