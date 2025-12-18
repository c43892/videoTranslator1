# VideoTranslator

A modular tool for translating video content by processing and replacing audio tracks while preserving the original video.

## Project Structure

```
volumn/
├── src/
│   ├── __init__.py
│   └── processors/
│       ├── __init__.py
│       ├── base.py              # Abstract base classes
│       └── video_separator.py   # Step 1: Video/Audio separation
├── test_step1.py               # Test script for Step 1
└── requirements.txt            # Python dependencies
```

## Step 1: Video/Audio Separation

### Features
- Separate any FFmpeg-supported video format into pure video and audio tracks
- Uses FFmpeg for fast, lossless stream extraction
- Support for multiple video/audio streams
- Comprehensive validation and error handling

### Prerequisites

**FFmpeg Installation:**

The FFmpeg binary must be installed and accessible in your system PATH.

- **Ubuntu/Debian:**
  ```bash
  apt-get update
  apt-get install -y ffmpeg
  ```

- **macOS:**
  ```bash
  brew install ffmpeg
  ```

- **Windows:**
  Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

### Usage

#### As a Library

```python
from pathlib import Path
from src.processors.video_separator import FFmpegVideoAudioSeparator

# Initialize separator
separator = FFmpegVideoAudioSeparator()

# Validate input
is_valid, error = separator.validate_input(Path("input.mp4"))
if not is_valid:
    print(f"Invalid input: {error}")
    exit(1)

# Get stream information
info = separator.get_stream_info(Path("input.mp4"))
print(f"Duration: {info['duration']}s")

# Perform separation
result = separator.separate(
    input_video_path=Path("input.mp4"),
    output_dir=Path("output"),
    overwrite=True
)

if result.success:
    print(f"Video: {result.video_path}")
    print(f"Audio: {result.audio_path}")
else:
    print(f"Failed: {result.error_message}")
```

#### Test Script

```bash
# Run the test script
python test_step1.py /path/to/video.mp4
```

The test script will:
1. Validate FFmpeg installation
2. Validate the input video
3. Display stream information
4. Separate video and audio tracks
5. Verify output files

### Output

- **video_only.mp4** - Video track without audio (stream copy, no quality loss)
- **audio_full.wav** - Audio track in WAV format (16-bit PCM, 44.1kHz, stereo)

### Supported Formats

Any format supported by FFmpeg, including:
- Container formats: MP4, MKV, AVI, MOV, WEBM, FLV, TS, 3GP, etc.
- Video codecs: H.264, H.265/HEVC, VP8, VP9, AV1, MPEG-4, etc.
- Audio codecs: AAC, MP3, AC3, Opus, Vorbis, FLAC, etc.

## Development

### Running Tests

```bash
# Run Step 1 test with a sample video
python test_step1.py sample_video.mp4
```

### Future Steps

- Step 2: Vocal/Instrumental separation
- Step 3: Speech transcription (SRT format)
- Step 4: Translation
- Step 5: Audio clipping
- Step 6: TTS generation with voice cloning
- Step 7: Audio track stitching
- Step 8: Final video assembly

## License

TBD

## Requirements

- Python 3.9+
- FFmpeg (must be installed separately)

## Contributing

TBD
