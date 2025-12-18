# Step 2: Vocal/Instrumental Separation - Detailed Requirements

## Overview

**Step Name:** Vocal/Instrumental Separation (Step 2)

**Purpose:** Separate the complete audio track into isolated vocal and instrumental (background) components using locally installed Demucs.

**Technology:** Demucs (Deep Extractor for Music Sources) - locally installed

---

## Input/Output Specification

### Input
- **File:** `audio_full.[format]` (from Step 1 - Audio/Video Track Separation)
- **Format:** WAV, MP3, FLAC, or other common audio formats
- **Source:** Output directory from Step 1

### Output
- **File 1:** `vocals.[format]` - Isolated vocal track containing only voice/singing
- **File 2:** `no_vocals.[format]` - Instrumental/background audio track (everything except vocals)
- **Format:** Same as input or configurable output format
- **Location:** Designated output directory (configurable)

---

## Implementation Details

### 2.1 Demucs Configuration

**Installation:**
- Demucs should be installed locally via pip: `pip install demucs`
- Requires Python 3.8+
- Dependencies include: PyTorch, torchaudio

**Model Selection:**
- Default model: `htdemucs` (Hybrid Transformer Demucs - best quality)
- Alternative models available: `htdemucs_ft`, `mdx_extra`, `mdx_extra_q`
- Model selection should be configurable

**Separation Targets:**
- Primary outputs needed: `vocals` and `no_vocals`
- Demucs produces 4 stems by default:
  - vocals
  - bass
  - drums
  - other
- Combine bass + drums + other = `no_vocals` track

### 2.2 Processing Workflow

```
1. Load audio_full.[format]
2. Run Demucs separation with selected model
3. Extract vocals stem → save as vocals.[format]
4. Combine (bass + drums + other) stems → save as no_vocals.[format]
5. Validate output files
6. Clean up intermediate files (optional)
```

### 2.3 Command Execution

**Demucs CLI Command:**
```bash
demucs --two-stems=vocals [options] audio_full.wav
```

**Key Options:**
- `--two-stems=vocals`: Separates only vocals and accompaniment (faster)
- `--out`: Output directory path
- `--device`: CPU or CUDA (GPU acceleration if available)
- `-n MODEL_NAME`: Specify model (e.g., `htdemucs`)
- `--mp3`: Output MP3 format
- `--float32`: Use float32 for processing (better quality)

---

## Implementation Requirements

### 3.1 Class Design

**Class:** `VocalSeparator` (inherits from `BaseProcessor`)

**Methods:**
- `separate(audio_path: str, output_dir: str) -> Tuple[str, str]`
  - Executes Demucs separation
  - Returns paths to `vocals` and `no_vocals` files
  
- `validate_output(vocals_path: str, no_vocals_path: str) -> bool`
  - Verifies output files exist and are valid audio
  
- `_run_demucs(input_path: str, output_dir: str) -> None`
  - Internal method to execute Demucs command
  
- `_combine_stems(bass_path: str, drums_path: str, other_path: str, output_path: str) -> None`
  - Combines non-vocal stems into single `no_vocals` track

**Configuration Parameters:**
- `model_name`: Demucs model to use (default: `htdemucs`)
- `device`: Processing device - `cpu`, `cuda`, or `auto` (default: `auto`)
- `output_format`: Audio format - `wav`, `mp3`, `flac` (default: `wav`)
- `two_stems_mode`: Boolean - use faster 2-stem separation (default: `True`)
- `float32`: Use float32 precision (default: `True`)
- `clip_mode`: Handling for clipping - `rescale`, `clamp`, `none` (default: `rescale`)
- `segment_size`: Segment size for processing, affects memory usage (default: `None` - uses model default)

### 3.2 Error Handling

**Expected Errors:**
1. **Demucs Not Installed**
   - Detection: Check if `demucs` command is available
   - Action: Raise clear error with installation instructions
   
2. **Insufficient Memory**
   - Detection: Monitor for out-of-memory errors
   - Action: Suggest using smaller segment size or CPU mode
   
3. **Invalid Input Audio**
   - Detection: Demucs fails to load audio file
   - Action: Validate audio file before processing, provide error message
   
4. **GPU/CUDA Errors**
   - Detection: CUDA-related exceptions
   - Action: Fall back to CPU processing automatically
   
5. **Output File Issues**
   - Detection: Output files missing or corrupted
   - Action: Retry with different settings or report failure

**Logging:**
- Log Demucs command being executed
- Log processing time for performance monitoring
- Log warnings for quality concerns (clipping, etc.)
- Log model and device being used

### 3.3 Quality Requirements

**Audio Quality:**
- Maintain original sample rate and bit depth where possible
- Minimize artifacts and bleeding between vocal/instrumental tracks
- Preserve dynamic range

**Timing:**
- Output files must have exact same duration as input
- No timing drift or synchronization issues

**Performance:**
- Support GPU acceleration when available (significantly faster)
- Progress indication for long audio files
- Estimated time remaining for user feedback

---

## Configuration File Example

```yaml
step2_vocal_separation:
  processor: demucs
  demucs:
    model: htdemucs              # Model name
    device: auto                 # auto, cpu, or cuda
    two_stems: true              # Use 2-stem mode (vocals + accompaniment)
    output_format: wav           # wav, mp3, flac
    float32: true                # Use float32 precision
    clip_mode: rescale           # rescale, clamp, or none
    segment_size: null           # null for default, or integer (e.g., 10)
  output:
    vocals_filename: vocals.wav
    no_vocals_filename: no_vocals.wav
    keep_intermediate: false     # Keep individual stems (bass, drums, other)
```

---

## File System Structure

```
output/
├── step1/
│   ├── video_only.mp4
│   └── audio_full.wav          # INPUT for Step 2
├── step2/
│   ├── vocals.wav              # OUTPUT 1
│   ├── no_vocals.wav           # OUTPUT 2
│   └── demucs/                 # Intermediate files (optional)
│       ├── htdemucs/
│       │   └── audio_full/
│       │       ├── vocals.wav
│       │       ├── bass.wav
│       │       ├── drums.wav
│       │       └── other.wav
```

---

## Dependencies

### Python Packages
```
demucs>=4.0.0
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.19.0
soundfile>=0.12.0
```

### System Requirements
- **Minimum:** 4GB RAM for CPU processing
- **Recommended:** 8GB+ RAM, NVIDIA GPU with 4GB+ VRAM
- **Disk Space:** ~2-3x input file size for temporary files
- **Python:** 3.8 or higher

---

## Testing Requirements

### Unit Tests
1. Test Demucs installation detection
2. Test configuration loading and validation
3. Test output file path generation
4. Test error handling for missing dependencies

### Integration Tests
1. Test separation with sample audio file (vocals present)
2. Test separation with instrumental-only audio
3. Test separation with various audio formats (WAV, MP3, FLAC)
4. Test GPU and CPU modes
5. Test stem combination (bass + drums + other)
6. Validate output duration matches input
7. Test cleanup of intermediate files

### Performance Tests
- Benchmark processing time for different file lengths
- Monitor memory usage during processing
- Test GPU vs CPU performance difference

---

## Success Criteria

Step 2 is successful when:
1. ✓ Demucs successfully separates vocals from background audio
2. ✓ Output files (`vocals.wav` and `no_vocals.wav`) are created
3. ✓ Output duration exactly matches input duration
4. ✓ Audio quality is preserved without significant artifacts
5. ✓ No audible vocals remain in `no_vocals.wav`
6. ✓ Processing completes without errors for common audio formats
7. ✓ GPU acceleration works when available
8. ✓ Clear error messages when Demucs is not installed
9. ✓ Handles both vocal-heavy and instrumental-heavy audio

---

## Known Limitations

1. **Processing Time:** Separation can take 1-3x the audio duration on CPU, faster on GPU
2. **Quality Trade-offs:** Perfect separation is not possible; some bleeding may occur
3. **Memory Usage:** Long audio files require significant RAM
4. **Model Size:** Demucs models are 300-500MB; first run downloads the model
5. **Format Support:** Some exotic audio formats may not be supported

---

## Future Enhancements

- Support for alternative separation models (e.g., Spleeter, UVR)
- Real-time separation preview
- Quality metrics/scoring for separation results
- Batch processing optimization
- Model caching and management
- Custom model training support

---

**Document Version:** 1.0  
**Last Updated:** November 16, 2025  
**Status:** Final - Ready for Implementation
