# Step 7: Audio Track Stitching - Detailed Requirements

## Overview
Combine all translated vocal clips with the instrumental/background audio track to create the complete translated audio output.

---

## Input Files

### Required Inputs:
1. **Translated Clips** (from Step 6):
   - `translated_clip_1.mp3`
   - `translated_clip_2.mp3`
   - `translated_clip_N.mp3`
   - Sequential numbering matching sentence IDs

2. **Background Audio** (from Step 2):
   - `no_vocals.[format]` - Instrumental/background audio track without vocals

3. **Timing Reference** (from Step 3):
   - `SRT_original.srt` - Original subtitle file containing timestamps for each sentence

---

## Process Flow

### 1. Load Background Track
- Start with `no_vocals.[format]` as the base audio layer
- This serves as the continuous background throughout the video

### 2. Parse Timestamps
- Read `SRT_original.srt` to extract start and end timestamps for each sentence
- Map sentence numbers to their time ranges

### 3. Process Each Translated Clip
For each `translated_clip_N.mp3`:

#### 3.1 Get Original Timestamp
- Retrieve sentence N's start and end time from `SRT_original.srt`
- Calculate original clip duration: `duration_original = end_time - start_time`

#### 3.2 Check Clip Duration
- Get actual duration of `translated_clip_N.mp3`: `duration_translated`
- Compare with `duration_original`
- **Note:** Step 6 already matches translated clip durations to original clip durations
- Significant mismatches here indicate potential issues in the pipeline

#### 3.3 Handle Duration Mismatches

**Important Context:**
- Step 6 TTS generation matches output durations to original clip durations using time-stretching
- Most clips should already match original durations within tolerance (±50ms)
- Duration mismatches here are typically due to SRT timestamp vs actual audio file duration differences

**If translated clip is LONGER than original:**
- Apply time-stretching to compress the clip
- Calculate compression factor: `factor = duration_original / duration_translated`
- Use **rubberband** for pitch-preserving time-stretching
- Preserve tone/pitch while reducing duration
- Tool: `pyrubberband` library or rubberband CLI
- Note: This should be rare since Step 6 handles most duration matching

**If translated clip is SHORTER than original:**
- Keep the translated clip at natural duration
- Insert at original start timestamp
- Add silence/gap after the clip to fill remaining time

**If durations match (within tolerance):**
- Use clip as-is without modification
- This is the expected case for most clips

#### 3.4 Apply Crossfading
- Apply fade-in at the beginning of each clip (e.g., 50-100ms)
- Apply fade-out at the end of each clip (e.g., 50-100ms)
- Purpose: Suppress audio clipping and create smooth transitions

#### 3.5 Insert Clip into Timeline
- Position the processed clip at the original start timestamp from SRT
- Overlay/mix the clip onto the background track
- Maintain same volume level as original vocals

### 4. Mix Audio Layers
- Combine all translated vocal clips with the background track
- Ensure proper audio mixing without distortion
- Volume levels should match the original vocal track volume

### 5. Export Final Audio
- Save the complete mixed audio as `audio_translated_full.mp3`
- Format: MP3
- Maintain quality comparable to input files

---

## Output Files

### Primary Output:
- **`audio_translated_full.mp3`**
  - Complete audio track with translated vocals and original background
  - MP3 format
  - Contains all mixed audio layers

### Metadata Output:
- **`stitching_metadata.json`**
  ```json
  {
    "no_vocals_file": "no_vocals.wav",
    "srt_reference": "SRT_original.srt",
    "total_clips": 25,
    "output_file": "audio_translated_full.mp3",
    "clips": [
      {
        "clip_number": 1,
        "source_file": "translated_clip_1.mp3",
        "timestamp_start": "00:00:01,500",
        "timestamp_end": "00:00:04,200",
        "duration_original": 2.7,
        "duration_translated": 3.1,
        "time_stretch_applied": true,
        "stretch_factor": 0.871,
        "final_duration": 2.7,
        "fade_in_ms": 50,
        "fade_out_ms": 50
      },
      {
        "clip_number": 2,
        "source_file": "translated_clip_2.mp3",
        "timestamp_start": "00:00:04,500",
        "timestamp_end": "00:00:07,800",
        "duration_original": 3.3,
        "duration_translated": 2.9,
        "time_stretch_applied": false,
        "gap_added_ms": 400,
        "final_duration": 3.3,
        "fade_in_ms": 50,
        "fade_out_ms": 50
      }
    ],
    "warnings": [
      "Clip 1 required time-stretching by factor 0.871",
      "Clip 5 required time-stretching by factor 0.623 (significant compression)"
    ]
  }
  ```

---

## Technical Requirements

### Audio Processing:

#### Time-Stretching (for longer clips):
- **Primary Tool**: Rubberband (via `pyrubberband` Python library)
- **Fallback**: FFmpeg with `atempo` filter (chain multiple filters if needed)
- **Preserve**: Pitch/tone must remain unchanged
- **Quality**: High-quality time-stretching algorithm

#### Crossfading:
- Apply exponential or linear fade curves
- Default fade duration: 50-100ms
- Configurable via settings if needed

#### Audio Mixing:
- Use pydub or FFmpeg for mixing operations
- Ensure no clipping (audio levels exceed maximum)
- Normalize if necessary

### Volume Matching:
- Translated vocal clips should have the same volume as original vocals
- No automatic gain adjustment unless specified in config
- Maintain consistency across all clips

### Timestamp Precision:
- Use millisecond precision for all timing operations
- Parse SRT timestamps accurately (HH:MM:SS,mmm format)
- Account for floating-point rounding in duration calculations

---

## Implementation Structure

### Class: `AudioStitcher`

**Location:** `src/processors/audio_stitcher.py`

**Base Class:** Inherits from `BaseProcessor`

**Key Methods:**

```python
class AudioStitcher(BaseProcessor):
    def process(self, input_data: dict) -> dict:
        """
        Main processing method
        
        Args:
            input_data: {
                'translated_clips_dir': path to directory with translated_clip_*.mp3,
                'no_vocals_file': path to background audio,
                'srt_file': path to SRT_original.srt,
                'output_dir': path for output files
            }
            
        Returns:
            {
                'audio_translated_full': path to output MP3,
                'metadata': stitching metadata dict
            }
        """
        
    def load_background_audio(self, no_vocals_file: str) -> AudioSegment:
        """Load the background/instrumental track"""
        
    def parse_srt_timestamps(self, srt_file: str) -> list[dict]:
        """Parse SRT file and extract timestamps for each sentence"""
        
    def process_clip(self, clip_file: str, original_duration: float) -> tuple:
        """
        Process a single translated clip
        
        Returns: (processed_audio, metadata_dict)
        """
        
    def time_stretch_audio(self, audio: AudioSegment, factor: float) -> AudioSegment:
        """Apply time-stretching with pitch preservation"""
        
    def apply_crossfade(self, audio: AudioSegment, fade_in_ms: int, fade_out_ms: int) -> AudioSegment:
        """Apply fade-in and fade-out to audio clip"""
        
    def mix_clips_onto_background(self, background: AudioSegment, clips: list) -> AudioSegment:
        """Overlay all processed clips onto background track"""
        
    def save_output(self, audio: AudioSegment, output_path: str):
        """Export final mixed audio as MP3"""
        
    def generate_metadata(self, clips_info: list) -> dict:
        """Generate stitching metadata JSON"""
```

---

## Configuration

### Config File Section: `audio_stitching`

```yaml
audio_stitching:
  output_format: "mp3"
  crossfade:
    enabled: true
    fade_in_ms: 50
    fade_out_ms: 50
  time_stretching:
    tool: "rubberband"  # Options: "rubberband", "ffmpeg"
    quality: "high"      # For rubberband: "low", "medium", "high"
  duration_tolerance_ms: 50  # Threshold for considering durations equal
  volume_matching: true
  export_quality: "192k"     # MP3 bitrate
```

---

## Dependencies

### Python Libraries:
```
pydub>=0.25.1           # Audio manipulation
pyrubberband>=0.3.0     # Time-stretching with pitch preservation
numpy>=1.24.0           # Audio sample processing
```

### External Tools:
- **FFmpeg** - Required by pydub for audio format conversion and mixing
- **Rubberband** - Optional but recommended for high-quality time-stretching
  - Install: `apt-get install rubberband-cli` (Linux) or `brew install rubberband` (Mac)
  - Windows: Download from https://breakfastquay.com/rubberband/

---

## Error Handling

### Expected Errors:

1. **Missing Input Files**
   - Translated clips not found
   - Background audio missing
   - SRT file not found
   - **Action**: Log error, raise exception with clear message

2. **Timestamp Parsing Errors**
   - Malformed SRT file
   - Missing timestamps
   - **Action**: Validate SRT format, report line number of error

3. **Clip Count Mismatch**
   - Number of translated clips doesn't match SRT entries
   - **Action**: Log warning, process available clips

4. **Time-Stretching Failure**
   - Extreme compression factors (e.g., > 2.0x or < 0.5x)
   - **Action**: Log warning, attempt processing, flag in metadata

5. **Audio Loading Errors**
   - Corrupted audio files
   - Unsupported formats
   - **Action**: Skip clip, log error, continue with others

6. **Memory Issues**
   - Large audio files causing memory overflow
   - **Action**: Process in chunks, clear intermediate data

---

## Quality Assurance

### Validation Checks:

1. **Duration Verification**
   - Final audio duration should match original audio duration
   - Tolerance: ±100ms

2. **Audio Quality**
   - No clipping (audio level checks)
   - No distortion in mixed output
   - Consistent volume across clips

3. **Synchronization**
   - Clips positioned at correct timestamps
   - No gaps or overlaps (except intentional)

4. **Output File Integrity**
   - Valid MP3 file format
   - Playable without errors
   - Metadata embedded correctly

---

## Testing Requirements

### Unit Tests:

```python
test_load_background_audio()
test_parse_srt_timestamps()
test_time_stretch_longer_clip()
test_add_gap_for_shorter_clip()
test_apply_crossfade()
test_mix_single_clip()
test_mix_multiple_clips()
test_handle_missing_clip()
test_extreme_time_stretch()
```

### Integration Test:

```python
def test_step7_complete():
    """
    Full end-to-end test for Step 7
    
    Input:
    - test/output/test02/step6/translated_clip_*.mp3 (multiple clips)
    - test/output/test02/step2/no_vocals.wav
    - test/output/test02/step3/SRT_original.srt
    
    Expected Output:
    - test/output/test02/step7/audio_translated_full.mp3
    - test/output/test02/step7/stitching_metadata.json
    
    Verification:
    - Output file exists and is valid MP3
    - Duration matches original audio duration (±100ms)
    - Metadata contains all processed clips
    - No audio clipping or distortion
    """
```

### Test Data Requirements:
- Sample clips with varying durations (longer, shorter, matching)
- Background audio with sufficient length
- Valid SRT file with multiple entries
- Edge cases: very short clips (<0.5s), long clips (>10s)

---

## Performance Considerations

### Optimization Strategies:

1. **Lazy Loading**
   - Don't load all clips into memory at once
   - Process and mix clips iteratively

2. **Caching**
   - Cache loaded background audio
   - Cache parsed SRT data

3. **Parallel Processing**
   - Time-stretching can be parallelized for multiple clips
   - Consider using multiprocessing for large batches

4. **Memory Management**
   - Clear processed clips from memory after mixing
   - Use generators where possible

### Expected Performance:
- Processing time: ~1-5 seconds per clip (depending on time-stretching)
- Total time for 50 clips: ~1-3 minutes
- Memory usage: ~500MB for typical video (5-10 min duration)

---

## Success Criteria

Step 7 is successful when:

1. ✓ All translated clips are correctly positioned at original timestamps
2. ✓ Background audio is preserved and mixed properly
3. ✓ Time-stretching maintains pitch/tone for longer clips
4. ✓ Gaps are properly inserted for shorter clips
5. ✓ Crossfading eliminates audio clipping artifacts
6. ✓ Output audio duration matches original audio (±100ms)
7. ✓ Volume levels are consistent and match original
8. ✓ Output file is valid MP3 format
9. ✓ Metadata accurately reflects all processing operations
10. ✓ No audio distortion or quality degradation

---

## Example Workflow

```python
# Step 7 execution example
from src.processors.audio_stitcher import AudioStitcher

stitcher = AudioStitcher(config)

input_data = {
    'translated_clips_dir': 'test/output/test02/step6/',
    'no_vocals_file': 'test/output/test02/step2/no_vocals.wav',
    'srt_file': 'test/output/test02/step3/SRT_original.srt',
    'output_dir': 'test/output/test02/step7/'
}

result = stitcher.process(input_data)

# Result:
# {
#     'audio_translated_full': 'test/output/test02/step7/audio_translated_full.mp3',
#     'metadata': {...}
# }
```

---

**Document Version:** 1.0  
**Date:** November 17, 2025  
**Status:** Final - Ready for Implementation
