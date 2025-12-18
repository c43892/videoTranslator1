# VideoTranslator - Requirements Document

## 1. Project Overview

**Project Name:** VideoTranslator

**Purpose:** A modular tool that translates video content by separating, processing, and replacing the vocal audio track while preserving the original video and background audio.

**Language:** Python

**Architecture Principle:** Modular design with pluggable implementations for each processing step, allowing easy replacement of tech stacks without affecting the overall workflow.

---

## 2. Tool Usage

**Input:**
- Video file path
- Target language (e.g., "English", "Spanish", "Chinese")
- (Optional) Terminology JSON file for consistent translation

**Output:**
- `video_translated.mp4` - Translated video with voice-cloned translated audio
- `SRT_original.srt` - Original transcribed subtitles
- `SRT_translated.srt` - Translated subtitles

**Command:**
```bash
python translate_video.py <input_video> --target-lang <language> --output-dir <output_directory> [--terminology <terminology.json>]
```

**Example:**
```bash
# Basic usage
python translate_video.py my_video.mp4 --target-lang English --output-dir output/

# With terminology file for consistent term translation
python translate_video.py my_video.mp4 --target-lang English --output-dir output/ --terminology terms.json
```

**Terminology File Format:**
The terminology file should be a JSON object mapping source language terms to target language terms:
```json
{
  "机器学习": "Machine Learning",
  "神经网络": "Neural Network",
  "深度学习": "Deep Learning"
}
```

---

## 3. Workflow Overview

The tool follows an 8-step pipeline that processes the video automatically:

```
Input Video → Audio/Video Separation → Vocal Separation → Transcription → 
Translation → Audio Clipping → TTS Generation → Audio Stitching → 
Final Video Assembly → Output (Video + Subtitles)
```

Each step uses the output from previous steps as input, creating a complete automated pipeline.

---

## 4. Detailed Requirements by Step

### Pipeline Dependencies

Each step depends on outputs from previous steps:
- **Step 1** uses: Input video file
- **Step 2** uses: `audio_full.[format]` from Step 1
- **Step 3** uses: `vocals.mp3` from Step 2
- **Step 4** uses: `SRT_original.srt` from Step 3
- **Step 5** uses: `vocals.mp3` from Step 2 + `SRT_original.srt` from Step 3
- **Step 6** uses: `original_clip_*.mp3` from Step 5 + both SRT files from Steps 3 & 4
- **Step 7** uses: `translated_clip_*.mp3` from Step 6 + `no_vocals.mp3` from Step 2 + `SRT_original.srt` from Step 3
- **Step 8** uses: `video_only.[format]` from Step 1 + `audio_translated_full.mp3` from Step 7

### Step 1: Audio/Video Track Separation
**Input:** Video file (path provided via command line)

**Process:** Separate the input video into:
- Pure video track (no audio)
- Audio track (complete)

**Output:**
- `video_only.[format]` - Video track without audio
- `audio_full.[format]` - Complete audio track

**Requirements:**
- Must support common video formats (MP4, AVI, MKV, etc.)
- Implementation should be abstracted behind an interface/class
- Error handling for corrupted or unsupported files

---

### Step 2: Vocal/Instrumental Separation
**Input:** `audio_full.[format]` from Step 1

**Process:** Separate the audio into vocal and non-vocal components

**Output:**
- `vocals.mp3` - Isolated vocal track (MP3 format for smaller file size)
- `no_vocals.mp3` - Instrumental/background audio track (MP3 format)

**Requirements:**
- Implementation must be pluggable (different vocal separation services/libraries)
- Should preserve audio quality and timing
- Handle cases where vocals are minimal or absent
- Output in MP3 format to reduce file size (benefits downstream processing)
- Use appropriate bitrate (e.g., 192kbps) to balance quality and size

---

### Step 3: Transcription
**Input:** `vocals.mp3` from Step 2

**Process:** Transcribe the vocal track to text with timestamps

**Output:**
- `SRT_original.srt` - Subtitle file in SRT format with:
  - Sequential sentence numbering
  - Start and end timestamps for each sentence
  - Original language text

**Requirements:**
- Must output valid SRT format
- Sentence segmentation should be natural and meaningful
- Implementation should be abstracted (support for different transcription services)
- Preserve accurate timing information

---

### Step 4: Translation
**Input:** `SRT_original.srt` from Step 3

**Process:** Translate all subtitle content to target language

**Output:**
- `SRT_translated.srt` - Translated subtitle file maintaining:
  - Same structure as original (numbering, timestamps)
  - Translated text content

**Requirements:**
- Preserve SRT formatting and timestamps exactly
- Support pluggable translation services
- Maintain sentence-to-sentence correspondence with original
- Target language should be configurable
- Optional: Support terminology file (JSON) for consistent translation of specific terms
  - Format: `{"source_term": "target_term", ...}`
  - If provided, prioritize terminology translations over general translation
- For large subtitle files (>20 entries), split into chunks of 10-20 entries per API call
  - Prevents API timeout issues with long videos
  - Maintains context within each chunk
  - Default chunk size: 15 entries

---

### Step 5: Audio Clipping (Original)
**Input:** 
- `vocals.mp3` from Step 2
- `SRT_original.srt` from Step 3

**Process:** Extract audio clips for each sentence based on timestamps in SRT

**Output:**
- `original_clip_1.mp3` - Audio clip for sentence 1
- `original_clip_2.mp3` - Audio clip for sentence 2
- `original_clip_N.mp3` - Audio clip for sentence N

**Requirements:**
- Clip naming must follow sequential pattern: `original_clip_{N}.mp3`
- Extract based on exact timestamps from SRT
- Maintain audio quality
- Handle edge cases (overlapping timestamps, very short clips)

---

### Step 6: TTS (Text-to-Speech) Generation
**Input:**
- All `original_clip_*.mp3` files from Step 5
- `SRT_original.srt` from Step 3
- `SRT_translated.srt` from Step 4

**Process:** For each clip (N):
1. Find corresponding sentence in `SRT_original.srt` (sentence N)
2. Find corresponding sentence in `SRT_translated.srt` (sentence N)
3. Send to TTS service:
   - Original audio clip
   - Original text
   - Translated text
4. Generate voice-cloned translated audio

**Output:**
- `translated_clip_1.mp3` - TTS audio for sentence 1
- `translated_clip_2.mp3` - TTS audio for sentence 2
- `translated_clip_N.mp3` - TTS audio for sentence N

**Requirements:**
- Support pluggable TTS services
- Voice cloning capability (match original speaker's voice)
- Maintain emotional tone and speaking style
- Duration should approximately match original clip timing
- Sequential processing with error recovery

---

### Step 7: Audio Track Stitching
**Input:**
- All `translated_clip_*.mp3` files from Step 6
- `no_vocals.mp3` from Step 2
- `SRT_original.srt` from Step 3 (for timestamps)

**Process:** 
1. Start with the `no_vocals` track as base
2. For each translated clip (N):
   - Get original timestamp from `SRT_original.srt` (sentence N)
   - Insert `translated_clip_N.mp3` at that timestamp
3. Mix/overlay the clips onto the background track

**Output:**
- `audio_translated_full.[format]` - Complete audio track with:
  - Original background/instrumental audio
  - Translated vocal clips at original timestamps

**Requirements:**
- Precise timestamp alignment
- Proper audio mixing (volume levels, overlays)
- Handle timing mismatches (translated clip longer/shorter than original)
- Maintain audio quality and synchronization

---

### Step 8: Final Video Assembly
**Input:**
- `video_only.[format]` from Step 1
- `audio_translated_full.[format]` from Step 7

**Process:** Combine video and translated audio tracks

**Output:**
- `video_translated.mp4` - Final translated video file
- `SRT_original.srt` - Original transcribed subtitles (copied from Step 3)
- `SRT_translated.srt` - Translated subtitles (copied from Step 4)

**Requirements:**
- Maintain original video quality and format
- Perfect audio/video synchronization
- Support output format configuration
- Preserve video metadata where appropriate
- Include translated subtitles in final output directory

---

## 5. Complete Pipeline Output

When the pipeline completes successfully, the final output directory contains:

1. **`video_translated.mp4`** - Main output: translated video with voice-cloned audio
2. **`SRT_original.srt`** - Original transcribed subtitles in source language
3. **`SRT_translated.srt`** - Translated subtitles in target language

### Intermediate Files (Optional Cleanup)
All intermediate files from Steps 1-7 can be preserved for debugging or removed to save space:
- Step 1: `video_only.mp4`, `audio_full.wav`
- Step 2: `vocals.mp3`, `no_vocals.mp3`
- Step 3: `SRT_original.srt`, transcription metadata
- Step 4: Translation metadata
- Step 5: `original_clip_*.mp3` files
- Step 6: `translated_clip_*.mp3` files
- Step 7: `audio_translated_full.mp3`

---

## 6. Non-Functional Requirements

### 6.1 Modularity
- Each step must be implemented as a separate, swappable component
- Use abstract base classes or interfaces for each processing step
- Allow configuration-based selection of implementations
- Minimize coupling between steps

### 6.2 Configuration
- Support configuration file (YAML/JSON) for:
  - Service selection per step
  - API keys and credentials
  - File paths and naming conventions
  - Language settings
  - Output formats

### 6.3 Error Handling
- Graceful error handling at each step
- Logging for debugging and monitoring
- Ability to resume from checkpoint if a step fails
- Clear error messages for user

### 6.4 Command Line Interface
- Accept input video path as argument
- Support additional parameters:
  - Target language (required)
  - Output directory
  - Configuration file path
  - Verbose/debug mode
  - Keep intermediate files option

### 6.5 File Management
- Organized output directory structure
- Optional cleanup of intermediate files
- Preserve original input file

### 6.6 Performance
- Support for parallel processing where applicable
- Progress indicators for long-running operations
- Efficient memory usage for large video files

---

## 7. Project Structure

```
VideoTranslator/
├── volumn/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                  # Abstract base classes
│   │   │   ├── video_separator.py      # Step 1 implementation
│   │   │   ├── vocal_separator.py      # Step 2 implementation
│   │   │   ├── transcriber.py          # Step 3 implementation
│   │   │   ├── translator.py           # Step 4 implementation
│   │   │   ├── audio_clipper.py        # Step 5 implementation
│   │   │   ├── tts_generator.py        # Step 6 implementation
│   │   │   ├── audio_stitcher.py       # Step 7 implementation
│   │   │   └── video_assembler.py      # Step 8 implementation
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── srt_handler.py          # SRT parsing/writing utilities
│   ├── config/
│   │   └── transcription_config.yaml   # Default configuration
│   ├── translate_video.py              # Main pipeline script
│   ├── test_step1.py                   # Individual step tests
│   ├── test_step2.py
│   ├── test_step3.py
│   ├── test_step4.py
│   ├── test_step5.py
│   ├── test_step6.py
│   ├── test_step7.py
│   ├── test_step8.py
│   ├── test_full_pipeline.py           # Complete pipeline test
│   └── requirements.txt                # Python dependencies
├── docker-compose.yml                  # Docker configuration
├── REQUIREMENTS.md                     # This document
├── REQUIREMENTS_STEP1.md               # Step-by-step requirements
├── REQUIREMENTS_STEP2.md
├── REQUIREMENTS_STEP3.md
├── REQUIREMENTS_STEP4.md
├── REQUIREMENTS_STEP5.md
├── REQUIREMENTS_STEP6.md
├── REQUIREMENTS_STEP7.md
└── REQUIREMENTS_STEP8.md
```

---

## 8. Future Considerations

### 8.1 Potential Enhancements
- Batch processing multiple videos
- GUI interface
- Support for multiple audio tracks
- Subtitle embedding in output video
- Quality presets (fast/balanced/high-quality)
- Resume from checkpoint on failure
- GPU acceleration support

### 8.2 Integration Points
- Each step documents expected input/output formats
- Version compatibility tracking for service implementations
- Plugin system for community-contributed implementations

---

## 9. Success Criteria

The tool is considered successful when:
1. ✓ All 8 steps execute sequentially without errors
2. ✓ Output video has translated audio with original video intact
3. ✓ Translated subtitles are generated
4. ✓ Audio synchronization is maintained (lip-sync acceptable within reason)
5. ✓ Voice quality matches original speaker characteristics
6. ✓ Background audio/music is preserved
7. ✓ Any processing step can be swapped with alternative implementation
8. ✓ Clear error messages guide users when issues occur
9. ✓ Process completes for common video formats (MP4, etc.)
10. ✓ Single command execution from input video to final output

---

## 10. Implementation Status

### Completed Steps:
- ✅ Step 1: Audio/Video separation (FFmpeg)
- ✅ Step 2: Vocal separation (Demucs)
- ✅ Step 3: Transcription (FunASR/SenseVoice)
- ✅ Step 4: Translation (OpenAI GPT-5-mini)
- ✅ Step 5: Audio clipping (FFmpeg)
- ✅ Step 6: TTS generation (IndexTTS2)
- ✅ Step 7: Audio stitching (FFmpeg)
- ✅ Step 8: Video assembly (FFmpeg)

### Tech Stack Decisions:
- **Video/Audio Processing**: FFmpeg
- **Vocal Separation**: Demucs (AI-based source separation)
- **Transcription**: FunASR with SenseVoice model
- **Translation**: OpenAI GPT-5-mini (default, configurable)
- **TTS**: IndexTTS2 (voice cloning)
- **Audio Mixing**: FFmpeg with time-stretching support

---

## 11. Next Steps

1. ✓ Implement all 8 processing steps
2. ✓ Create individual test scripts for each step
3. ⏳ Create complete pipeline script (`translate_video.py`)
4. ⏳ Create full pipeline test (`test_full_pipeline.py`)
5. Add comprehensive error handling and logging
6. Create user documentation and examples
7. Add configuration file support
8. Optimize performance and memory usage

---

**Document Version:** 2.0  
**Last Updated:** November 17, 2025  
**Status:** Implementation Complete - Pipeline Integration Pending
3. Implement abstract base classes for each processor
4. Create a minimal working pipeline with stub implementations
5. Implement each step incrementally with chosen tech stack
6. Add comprehensive testing
7. Create user documentation

---

**Document Version:** 1.0  
**Last Updated:** November 16, 2025  
**Status:** Draft - Pending Tech Stack Decisions
