# Step 3: Transcription - Detailed Requirements

## Overview
Transcribe the vocal audio track to text with timestamps using OpenAI's Whisper library, with automatic language detection and non-dialogue event detection.

---

## Input
- **File:** `vocals.[format]` (output from Step 2)
- **Format:** Audio file (MP3, WAV, or other common audio formats)
- **Content:** Isolated vocal track from the video

---

## Output
- **File:** `SRT_original.srt`
- **Format:** Standard SRT (SubRip Subtitle) format
- **Encoding:** UTF-8

### SRT Format Structure
```
1
00:00:01,000 --> 00:00:04,500
This is the first sentence.

2
00:00:04,600 --> 00:00:08,200
This is the second sentence.

3
00:00:08,300 --> 00:00:09,500
[[ laughing ]]

4
00:00:09,600 --> 00:00:13,000
This continues with dialogue.
```

---

## Technical Requirements

### 3.1 Transcription Engine
- **Library:** OpenAI Whisper (local installation)
- **Model Selection:** Configurable (tiny, base, small, medium, large)
  - Default: `base` or `small` for balance of speed and accuracy
  - Allow user to specify model via configuration
- **Installation:** Must be locally installed (no cloud API calls)

### 3.2 Language Detection
- **Auto-detection:** Automatically detect the spoken language
- **Language Output:** Log/save the detected language for use in Step 4 (translation)
- **Multi-language Handling:** If multiple languages detected, use the primary/dominant language
- **Language Metadata:** Include detected language in output metadata or separate file

### 3.3 Transcription Features
- **Timestamps:** Generate accurate start and end times for each segment
- **Sentence Segmentation:** 
  - Break transcription into natural sentence/phrase boundaries
  - Each subtitle entry should represent a complete thought or sentence
  - Maximum length per subtitle: 2-3 lines (configurable)
  - Target duration: 1-7 seconds per subtitle (typical SRT standard)
  
- **Text Formatting:**
  - Proper capitalization
  - Correct punctuation
  - Remove filler words (optional, configurable)

### 3.4 Non-Dialogue Event Detection
**Requirement:** Detect and mark non-speech audio events with special notation

**Event Types to Detect:**
- Laughter: `[[ laughing ]]`
- Crying/Sobbing: `[[ crying ]]`
- Breathing (heavy/audible): `[[ breathing ]]`
- Panting: `[[ panting ]]`
- Sighing: `[[ sighing ]]`
- Coughing: `[[ coughing ]]`
- Screaming: `[[ screaming ]]`
- Gasping: `[[ gasping ]]`
- Music/Singing (if not separated): `[[ music ]]`
- Unintelligible sounds: `[[ inaudible ]]`
- Other: `[[ sound ]]`

**Implementation:**
- Use Whisper's output to detect non-speech segments
- If transcription confidence is low or output contains markers like [LAUGH], [MUSIC], etc., convert to bracketed format
- Preserve timestamps for these events
- Each non-dialogue event gets its own subtitle entry

**Format:**
```
5
00:00:15,000 --> 00:00:16,200
[[ laughing ]]
```

### 3.5 Timestamp Accuracy
- **Precision:** Millisecond precision (HH:MM:SS,mmm format)
- **Alignment:** Timestamps should align with actual speech boundaries
- **Gaps:** Minimal gaps between consecutive subtitles (< 200ms where possible)
- **Overlap:** No overlapping timestamps

### 3.6 Quality Requirements
- **Accuracy:** Aim for >90% word accuracy for clear speech
- **Completeness:** Capture all audible speech and significant non-dialogue events
- **Consistency:** Maintain consistent formatting throughout

---

## Implementation Design

### 3.7 Class Structure
```python
# Abstract base class
class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, output_path: str, **kwargs) -> dict:
        """
        Transcribe audio to SRT format
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to output SRT file
            **kwargs: Additional configuration
            
        Returns:
            dict: Metadata including detected language, model used, duration, etc.
        """
        pass

# Whisper implementation
class WhisperTranscriber(BaseTranscriber):
    def __init__(self, model_name: str = "base", device: str = "auto"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: "cpu", "cuda", or "auto"
        """
        pass
    
    def transcribe(self, audio_path: str, output_path: str, **kwargs) -> dict:
        pass
    
    def _detect_non_dialogue(self, segment: dict) -> str:
        """Detect if segment is non-dialogue and return appropriate marker"""
        pass
    
    def _format_srt(self, segments: list) -> str:
        """Convert Whisper segments to SRT format"""
        pass
```

### 3.8 Configuration Parameters
```yaml
transcription:
  engine: "whisper"  # Future: could support other engines
  whisper:
    model: "base"  # tiny, base, small, medium, large
    device: "auto"  # auto, cpu, cuda
    language: null  # null for auto-detection, or specify (e.g., "en", "zh")
    task: "transcribe"  # transcribe or translate (Whisper built-in)
    
  output:
    max_chars_per_line: 42  # Standard for readability
    max_lines_per_subtitle: 2
    min_duration: 0.5  # Minimum subtitle duration in seconds
    max_duration: 7.0  # Maximum subtitle duration in seconds
    
  non_dialogue:
    detect: true
    confidence_threshold: 0.7  # Below this, check for non-dialogue
    markers:
      laughing: "[[ laughing ]]"
      crying: "[[ crying ]]"
      breathing: "[[ breathing ]]"
      # ... other markers
```

### 3.9 Processing Steps
1. **Load Audio:** Read the `vocals.[format]` file
2. **Initialize Whisper:** Load specified model
3. **Transcribe:** Process audio through Whisper
4. **Analyze Segments:** For each segment:
   - Check if it's dialogue or non-dialogue
   - Apply appropriate formatting
   - Generate timestamp
5. **Format SRT:** Convert to proper SRT format
6. **Save Output:** Write to `SRT_original.srt`
7. **Return Metadata:** Language detected, duration, segment count, etc.

---

## Error Handling

### 3.10 Error Scenarios
- **Missing Audio File:** Clear error message with expected file path
- **Corrupted Audio:** Attempt to repair or inform user
- **Empty/Silent Audio:** Generate warning, create empty SRT or mark as `[[ silence ]]`
- **Whisper Model Not Found:** Auto-download or instruct user on installation
- **Out of Memory:** Suggest smaller model or chunk processing
- **Unsupported Language:** Log warning, attempt transcription anyway
- **Write Permission:** Check output directory permissions before processing

### 3.11 Logging
- Log detected language and confidence
- Log model used and device (CPU/GPU)
- Log processing time and audio duration
- Log segment count and any non-dialogue events detected
- Warning for low-confidence segments

---

## Performance Requirements

### 3.12 Speed
- **Target:** Process 1 minute of audio in < 1 minute (for base model on GPU)
- **Progress:** Display progress indicator for long audio files
- **Chunking:** For very long audio (>30 min), consider chunked processing

### 3.13 Resource Usage
- **Memory:** Should handle 30-60 minute audio files without excessive memory usage
- **GPU:** Utilize GPU if available for faster processing
- **CPU Fallback:** Must work on CPU-only systems (slower but functional)

---

## Output Validation

### 3.14 SRT Validation
- Validate SRT format is correct:
  - Sequential numbering (1, 2, 3, ...)
  - Valid timestamp format
  - No duplicate timestamps
  - No missing entries
- Provide validation function to verify output

---

## Dependencies

### 3.15 Required Libraries
```txt
openai-whisper>=20230918
ffmpeg-python>=0.2.0
torch>=2.0.0  # For Whisper
numpy>=1.24.0
```

### 3.16 System Requirements
- **FFmpeg:** Must be installed on system (Whisper dependency)
- **Python:** 3.8+
- **Optional:** CUDA-capable GPU for faster processing

---

## Testing Requirements

### 3.17 Test Cases
1. **Standard Speech:** Clean dialogue in common language (English)
2. **Accented Speech:** Various accents and speaking styles
3. **Multiple Languages:** Test auto-detection with different languages
4. **Non-Dialogue Events:** Audio with laughter, crying, etc.
5. **Mixed Content:** Dialogue interspersed with non-dialogue
6. **Silent Periods:** Long pauses or silence
7. **Poor Audio Quality:** Noisy or low-quality audio
8. **Edge Cases:** Very short clips, very long clips

### 3.18 Success Criteria
- ✓ Valid SRT file generated for all test cases
- ✓ Language correctly detected (>95% accuracy for clear speech)
- ✓ Non-dialogue events properly marked
- ✓ Timestamps accurate within ±500ms
- ✓ Text transcription >85% accurate for clear speech
- ✓ Proper handling of all error scenarios

---

## Integration Points

### 3.19 Input from Step 2
- Expects `vocals.[format]` in designated output directory
- File format should be compatible with Whisper (MP3, WAV, etc.)

### 3.20 Output to Step 4
- `SRT_original.srt` with proper formatting
- Detected language information (for translation target detection)
- Metadata about transcription quality/confidence

### 3.21 Output to Step 5
- `SRT_original.srt` with accurate timestamps for audio clipping
- Ensure timestamp precision sufficient for accurate clip extraction

---

## Future Enhancements

### 3.22 Potential Improvements
- **Speaker Diarization:** Identify and label different speakers
- **Word-level Timestamps:** More precise timing for each word
- **Custom Vocabulary:** Support for domain-specific terms
- **Post-processing:** Grammar and spelling correction
- **Multiple Language Support:** Handle videos with multiple languages
- **Confidence Scores:** Include transcription confidence in metadata
- **Alternative Engines:** Support for Azure Speech, Google STT, etc.

---

## Command Line Usage Example
```bash
# Basic usage with default settings
python -m src.processors.transcriber --input vocals.mp3 --output SRT_original.srt

# With custom Whisper model
python -m src.processors.transcriber --input vocals.mp3 --output SRT_original.srt --model medium

# Specify language (skip auto-detection)
python -m src.processors.transcriber --input vocals.mp3 --output SRT_original.srt --language en

# Disable non-dialogue detection
python -m src.processors.transcriber --input vocals.mp3 --output SRT_original.srt --no-detect-events
```

---

## Notes
- Whisper sometimes outputs text markers like [MUSIC], [LAUGHTER] - these should be converted to the `[[ ... ]]` format
- Consider using Whisper's `word_timestamps=True` for more precise segmentation
- The `task` parameter in Whisper can be "transcribe" or "translate" - for this step, always use "transcribe"
- Whisper's built-in translation feature is NOT used here; translation happens in Step 4
