# Step 6: TTS (Text-to-Speech) Generation - Detailed Requirements

## Overview

**Step Name:** TTS (Text-to-Speech) Generation

**Purpose:** Generate translated audio clips with voice cloning based on original audio clips and translated text, matching the original speaker's voice characteristics.

**Position in Pipeline:** Sixth step - receives audio clips from Step 5, original SRT from Step 3, and translated SRT from Step 4

---

## 1. Functional Requirements

### 1.1 Input Specifications

**Primary Inputs:**

1. **Original Audio Clips:**
   - Files: `original_clip_1.mp3`, `original_clip_2.mp3`, ..., `original_clip_N.mp3` (from Step 5)
   - Format: MP3 (or any audio format supported by TTS service)
   - Content: Individual vocal clips extracted from the original audio
   - Source: Step 5 output directory
   - Purpose: Voice reference for cloning/matching speaker characteristics

2. **Original SRT File:**
   - File: `SRT_original.srt` (from Step 3)
   - Format: Standard SRT (SubRip Subtitle) format
   - Encoding: UTF-8
   - Content: Timestamped transcription in source language
   - Purpose: Provides original text for each clip

3. **Translated SRT File:**
   - File: `SRT_translated.srt` (from Step 4)
   - Format: Standard SRT (SubRip Subtitle) format
   - Encoding: UTF-8
   - Content: Timestamped translation in target language
   - Purpose: Provides translated text to synthesize

**Input Validation:**
- All three input sources must exist and be readable
- Both SRT files must have the same number of entries
- Number of audio clips must match number of SRT entries
- Each `original_clip_{N}.mp3` must correspond to entry N in SRT files
- Audio clips must be valid audio files
- SRT entries must have valid structure

### 1.2 Processing Requirements

**Core Functionality:**

For each subtitle entry (N):
1. **Load Inputs:**
   - Read `original_clip_{N}.mp3` as voice reference
   - Extract entry N from `SRT_original.srt` to get original text
   - Extract entry N from `SRT_translated.srt` to get translated text

2. **Content Classification:**
   - **Pure Non-Dialogue Events** (e.g., `[[ laughing ]]`, `[[ music ]]`):
     - Pattern: Text matching `^\[\[.*\]\]$` (entire text is enclosed in double brackets)
     - Action: **COPY** the original audio clip as-is (no TTS generation)
     - Output: `translated_clip_{N}.mp3` = copy of `original_clip_{N}.mp3`
   
   - **Mixed Content** (e.g., `[[ laughing ]] Hello`, `Welcome [[ applause ]]`):
     - Pattern: Text containing `[[ ... ]]` markers plus other content
     - From Step 4: Event markers already removed, only dialogue remains in translated text
     - Action: Generate TTS for the translated dialogue text
   
   - **Pure Dialogue:**
     - Action: Generate TTS for the translated text

3. **TTS Generation (for dialogue content):**
   - **IndexTTS2 API Call:**
     ```python
     self.tts.infer(
         spk_audio_prompt=str(original_clip_path),  # Voice reference
         text=translated_text,                       # Text to synthesize
         output_path=str(temp_wav_path),            # Output WAV first
         emo_alpha=1.0,                             # Emotion strength (0.0-1.0)
         use_random=False,                          # Keep False for best quality
         verbose=True                                # Show progress
     )
     ```
   
   - **Voice Cloning:**
     - Uses `spk_audio_prompt` parameter to clone voice from reference audio
     - Matches speaker's timbre, pitch, and speaking characteristics
     - Zero-shot cloning (no training required)
   
   - **Silence Trimming:**
     - After TTS generation, trim silence from beginning and end
     - Use FFmpeg `silenceremove` filter to remove leading/trailing silence
     - Process: Remove start silence → reverse → remove start silence (end) → reverse back
     - Threshold: -40dB (anything quieter considered silence)
     - Ensures duration comparison is based on actual speech content, not silence
   
   - **Duration Matching:**
     - Get original clip duration using ffprobe
     - Get trimmed TTS duration (after silence removal)
     - **If TTS is LONGER than original (difference > 50ms):**
       - Apply pitch-preserving time-stretching to compress (speed up)
       - Use FFmpeg `atempo` filter
       - Preserve pitch/tone while playing faster
     - **If TTS is SHORTER than original:**
       - Keep as-is, no stretching (natural speech pace preserved)
   
   - **Post-Processing:**
     - IndexTTS2 generates WAV format
     - Trim silence from output
     - Apply duration compression if needed (time-stretching)
     - Convert WAV → MP3 using FFmpeg:
       ```bash
       ffmpeg -y -i temp.wav -codec:a libmp3lame -b:a 192k output.mp3
       ```
     - Delete temporary WAV file

4. **Save Output:**
   - Filename: `translated_clip_{N}.mp3`
   - Format: MP3 (converted from WAV)
   - Bitrate: 192kbps (configurable)
   - Quality: High-quality voice cloning

**Sequential Processing:**
- Process clips in order (1, 2, 3, ..., N)
- **DO NOT** use parallel processing (GPU memory constraints)
- Track progress: log every clip or every 10 clips
- Continue processing even if individual clips fail (with logging)

**Duration Handling:**

⚠️ **UPDATED:** Time-stretching with silence trimming is implemented to handle longer TTS outputs.

**Implementation:**
- After TTS generation, trim silence from beginning and end using FFmpeg `silenceremove` filter
- Compare trimmed TTS duration with original clip duration
- **If TTS is SHORTER than original:** Keep as-is, no modification
- **If TTS is LONGER than original (difference > 50ms):**
  - Apply pitch-preserving time-stretching using FFmpeg `atempo` filter
  - **Speed up the audio** (NOT truncate/cut) to match original duration exactly
  - Preserve pitch/tone while playing faster
  - Example: 3.0s audio → 2.0s (factor=0.67) = plays 1.5x faster with same pitch
- This ensures translated speech doesn't exceed original timing

**Duration Matching Process:**
1. Generate TTS audio from IndexTTS2
2. Trim silence from beginning and end:
   - Use FFmpeg silenceremove filter with -40dB threshold
   - Remove leading silence → reverse → remove trailing silence → reverse back
   - Ensures comparison based on actual speech, not silence padding
3. Get original clip duration via ffprobe
4. Get trimmed TTS duration
5. Compare durations:
   - **trimmed_duration < original_duration:** Keep TTS as-is (no stretching)
   - **trimmed_duration > original_duration + 50ms:** Apply time-stretching to speed up
6. Time-stretching (when needed):
   - Use `ffmpeg -filter:a "atempo={factor}"` to speed up (factor < 1.0 = faster playback)
   - **NOT truncation:** Entire audio is preserved, just played faster
   - **Chain multiple atempo filters if factor outside 0.5-2.0 range:**
     - FFmpeg atempo filter only accepts values between 0.5 and 2.0
     - For factor < 0.5: Chain multiple 0.5 filters, then multiply remaining factor by 2.0 for each
       - Example: factor=0.25 → atempo=0.5,atempo=0.5 (0.5 × 0.5 = 0.25)
       - **CRITICAL:** Use `remaining_factor *= 2.0` NOT `remaining_factor /= 0.5`
     - For factor > 2.0: Chain multiple 2.0 filters, then divide remaining factor by 2.0 for each
       - Example: factor=4.0 → atempo=2.0,atempo=2.0 (2.0 × 2.0 = 4.0)
       - Correct: `remaining_factor /= 2.0`
   - Or use rubberband for higher quality: `rubberband -t {factor} -p input output`
7. Preserve pitch/tone during speed change (sounds same, just faster)

**Benefits:**
- **Natural Timing:** Shorter translations remain natural, no artificial stretching
- **No Overlaps:** Longer translations sped up to fit original timing
- **Complete Audio:** Entire audio preserved (time-stretched, not truncated/cut)
- **Preserved Quality:** Pitch/tone maintained during speed adjustment
- **Flexibility:** Handles variable translation lengths gracefully

**Timing Information:**
- Preserve original timestamps from `SRT_original.srt` for use in Step 7
- Track duration adjustments in metadata (original_duration, pre-stretch duration, final duration)
- Log time-stretch operations for debugging
- Example logging:
  ```python
  if time_stretch_applied:
      logger.info(f"Clip {N}: Duration adjusted {pre_stretch_dur:.3f}s -> {final_dur:.3f}s (factor: {factor:.3f})")
  ```

### 1.3 Output Specifications

**Output Files:**
- **Filename Pattern:** `translated_clip_{N}.mp3`
  - `{N}` = Subtitle entry number from SRT (1-based index)
  - Must match numbering of original clips
  - Examples: `translated_clip_1.mp3`, `translated_clip_2.mp3`, ..., `translated_clip_N.mp3`

- **Format:** MP3 (or configurable: WAV, FLAC, AAC)
- **Codec:** LAME MP3 encoder or equivalent
- **Bitrate:** 192kbps (configurable, range: 128-320kbps)
- **Sample Rate:** 24000Hz or higher (service-dependent)
- **Channels:** Mono or Stereo (typically mono for speech)

**Output Location:**
- Default: `output/step6/` or configurable directory
- Create output directory if it doesn't exist
- Organize by job/video name if processing multiple videos

**Metadata Output:**
- Generate `tts_metadata.json` containing:
  ```json
  {
    "source_clips_dir": "output/step5/",
    "original_srt": "SRT_original.srt",
    "translated_srt": "SRT_translated.srt",
    "total_clips": 46,
    "source_language": "Chinese",
    "target_language": "English",
    "tts_service": "OpenAI TTS",
    "tts_model": "tts-1",
    "voice_model": "alloy",
    "clips": [
      {
        "clip_number": 1,
        "original_filename": "original_clip_1.mp3",
        "translated_filename": "translated_clip_1.mp3",
        "original_text": "你好",
        "translated_text": "Hello",
        "original_duration": 0.8,
        "translated_duration": 0.8,
        "duration_ratio": 1.0,
        "time_stretch_applied": false,
        "is_event": false,
        "processing_status": "success",
        "retry_count": 0,
        "error_message": null
      },
      {
        "clip_number": 2,
        "original_filename": "original_clip_2.mp3",
        "translated_filename": "translated_clip_2.mp3",
        "original_text": "[[ laughing ]]",
        "translated_text": "[[ laughing ]]",
        "original_duration": 1.2,
        "translated_duration": 1.2,
        "duration_ratio": 1.0,
        "is_event": true,
        "processing_status": "copied",
        "retry_count": 0,
        "error_message": null
      },
      {
        "clip_number": 5,
        "original_filename": "original_clip_5.mp3",
        "translated_filename": "translated_clip_5.mp3",
        "original_text": "How are you?",
        "translated_text": "你好吗?",
        "original_duration": 1.1,
        "translated_duration": 1.0,
        "duration_ratio": 0.91,
        "time_stretch_applied": false,
        "is_event": false,
        "processing_status": "success",
        "retry_count": 2,
        "error_message": null
      },
      ...
    ],
    "statistics": {
      "total_processed": 46,
      "successful": 44,
      "copied_events": 2,
      "failed": 0,
      "succeeded_on_retry": 3,
      "avg_retry_count": 0.15,
      "avg_duration_ratio": 0.92,
      "total_processing_time": 156.7
    }
  }
  ```

---

## 2. Interface Design

### 2.1 Core Architecture

**Separation of Concerns:**

1. **Core TTS Engine (`IndexTTS2Generator`)** - Located in `volumn/src/processors/tts_generator.py`
   - Handles IndexTTS2 model initialization
   - Provides **single-clip TTS generation** method
   - No looping or batch processing logic
   - Focused solely on: one audio clip + text → generate TTS

2. **Batch Processor** - Located in higher-level code or test scripts
   - Loads SRT files
   - Loops through all subtitle entries
   - Calls core TTS engine for each clip
   - Handles metadata generation and statistics

3. **Test Script (`test_step6.py`)** - Separate file for testing
   - Located in `volumn/test_step6.py`
   - Implements the batch processing loop
   - Validates inputs and outputs
   - Generates reports and statistics

### 2.2 Abstract Base Class

**Location:** `volumn/src/processors/tts_generator.py`

**Dependencies:**
```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import re
import shutil
import logging
from pydub import AudioSegment
```

**Class Definitions:**

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class ProcessingStatus(Enum):
    """Status of TTS clip processing"""
    SUCCESS = "success"
    COPIED = "copied"  # Event clip copied from original
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TTSClip:
    """Information about a single TTS-generated clip"""
    clip_number: int
    original_filename: str
    translated_filename: str
    original_text: str
    translated_text: str
    original_duration: float
    translated_duration: float
    duration_ratio: float
    is_event: bool
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    output_path: Optional[Path] = None

class TTSGenerator(ABC):
    """
    Abstract base class for TTS generation implementations.
    
    This class provides CORE TTS functionality for SINGLE clips only.
    Batch processing should be implemented in higher-level code.
    """
    
    @abstractmethod
    def generate_single_clip(
        self,
        reference_audio_path: Path,
        translated_text: str,
        output_path: Path,
        emo_alpha: float = 1.0,
        **kwargs
    ) -> bool:
        """
        CORE METHOD: Generate TTS audio for a SINGLE clip.
        
        This is the fundamental TTS operation:
        Input: 1 audio file + 1 text → Output: 1 generated audio file
        
        Args:
            reference_audio_path: Path to reference audio (for voice cloning)
            translated_text: Text to synthesize
            output_path: Where to save generated audio
            emo_alpha: Emotion influence (0.0-1.0, default: 1.0)
            **kwargs: Additional TTS parameters (use_random, etc.)
            
        Returns:
            bool: True if successful, False if failed
            
        Raises:
            FileNotFoundError: If reference audio doesn't exist
            RuntimeError: If TTS generation fails
        """
        pass
    
    @abstractmethod
    def is_event_text(self, text: str) -> bool:
        """
        Check if text is a pure non-dialogue event.
        
        Args:
            text: Text to check
            
        Returns:
            True if text matches event pattern (e.g., "[[ laughing ]]")
        """
        pass
    
    @abstractmethod
    def copy_event_clip(
        self,
        source_path: Path,
        destination_path: Path
    ) -> bool:
        """
        Copy original clip for event entries (no TTS needed).
        
        Args:
            source_path: Path to original audio clip
            destination_path: Path for output copy
            
        Returns:
            bool: True if successful, False if failed
        """
        pass
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get audio duration in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0
```
```

### 2.3 IndexTTS2 Implementation

**Critical Implementation Details:**

**Core Class: Focused on Single-Clip TTS Generation**

The `IndexTTS2Generator` class should focus ONLY on:
1. Initializing the IndexTTS2 model
2. Generating TTS for ONE audio clip at a time
3. Utility methods (event detection, duration, format conversion)

**NO batch processing logic in this class!**

**Imports and Path Setup:**
```python
import sys
import shutil
import subprocess
from pathlib import Path
from pydub import AudioSegment
import logging

# Add IndexTTS2 to Python path
INDEXTTS_PATH = Path("/app/volumn/.cache/index-tts")
if str(INDEXTTS_PATH) not in sys.path:
    sys.path.insert(0, str(INDEXTTS_PATH))

from indextts.infer_v2 import IndexTTS2 as IndexTTS2Model

logger = logging.getLogger(__name__)
```

**Class Implementation:**

```python
class IndexTTS2Generator(TTSGenerator):
    """
    IndexTTS2-based TTS generator with voice cloning.
    
    This class provides CORE TTS functionality for SINGLE clips.
    Batch processing should be handled by calling code.
    """
    
    def __init__(
        self,
        model_dir: str = "/app/volumn/.cache/index-tts/checkpoints",
        config_path: str = "/app/volumn/.cache/index-tts/checkpoints/config.yaml",
        use_fp16: bool = True,
        use_cuda_kernel: bool = False,
        use_deepspeed: bool = False,
        default_format: str = "mp3",
        default_bitrate: str = "192k"
    ):
        """
        Initialize IndexTTS2 generator.
        
        Args:
            model_dir: Directory containing IndexTTS2 model checkpoints
            config_path: Path to config.yaml
            use_fp16: Use half-precision (faster, less VRAM) - RECOMMENDED: True
            use_cuda_kernel: Use compiled CUDA kernels - RECOMMENDED: False (test first)
            use_deepspeed: Use DeepSpeed acceleration - RECOMMENDED: False (may be slower)
            default_format: Default output format (mp3, wav, etc.)
            default_bitrate: Default audio bitrate (e.g., "192k")
        """
        self.model_dir = Path(model_dir)
        self.config_path = Path(config_path)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"IndexTTS2 model directory not found: {model_dir}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"IndexTTS2 config not found: {config_path}")
        
        logger.info(f"Initializing IndexTTS2 with model_dir={model_dir}, use_fp16={use_fp16}")
        
        # Initialize IndexTTS2 model
        # NOTE: This may take 30-60 seconds to load all models
        self.tts = IndexTTS2Model(
            cfg_path=str(self.config_path),
            model_dir=str(self.model_dir),
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed
        )
        
        self.default_format = default_format
        self.default_bitrate = default_bitrate
        logger.info("IndexTTS2 model loaded successfully")
    
    def generate_single_clip(
        self,
        reference_audio_path: Path,
        translated_text: str,
        output_path: Path,
        emo_alpha: float = 1.0,
        **kwargs
    ) -> bool:
        """
        CORE METHOD: Generate TTS audio for a SINGLE clip.
        
        This is the fundamental TTS operation:
        Input: 1 reference audio + 1 text → Output: 1 TTS audio file
        
        IndexTTS2 Process:
        1. Generate audio to WAV format (IndexTTS2 native output)
        2. Convert WAV to target format (MP3) using FFmpeg
        3. Clean up temporary WAV file
        
        Args:
            reference_audio_path: Audio file for voice cloning (e.g., original_clip_5.mp3)
            translated_text: Text to synthesize (e.g., "Hello, how are you?")
            output_path: Where to save result (e.g., translated_clip_5.mp3)
            emo_alpha: Emotion influence (0.0-1.0, default: 1.0)
            **kwargs: Additional params (use_random=False, etc.)
        
        Returns:
            bool: True if successful, False if failed
        
        Example Implementation:
        ```python
        try:
            # Validate input
            if not reference_audio_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
            
            # Step 1: Generate to WAV first (IndexTTS2 native format)
            temp_wav = output_path.with_suffix('.wav')
            use_random = kwargs.get('use_random', False)
            
            logger.info(f"Generating TTS: {translated_text[:50]}...")
            
            # Call IndexTTS2 inference
            self.tts.infer(
                spk_audio_prompt=str(reference_audio_path),  # Voice reference
                text=translated_text,                         # Text to speak
                output_path=str(temp_wav),                   # Output WAV
                emo_alpha=emo_alpha,                         # Emotion strength
                use_random=use_random,                       # Randomness (False recommended)
                verbose=False                                # Reduce output noise
            )
            
            # Step 2: Convert to target format if needed
            if output_path.suffix.lower() != '.wav':
                self._convert_audio(temp_wav, output_path, self.default_bitrate)
                temp_wav.unlink()  # Delete temp WAV
            else:
                temp_wav.rename(output_path)
            
            logger.info(f"TTS generated successfully: {output_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return False
        ```
        """
        pass
    
    def is_event_text(self, text: str) -> bool:
        """Check if text is pure event (e.g., [[ laughing ]])."""
        import re
        pattern = r'^\[\[.*\]\]$'
        return bool(re.match(pattern, text.strip()))
    
    def copy_event_clip(
        self,
        clip_number: int,
        source_path: Path,
        destination_path: Path
    ) -> TTSClip:
        """
        Copy original clip for events (no TTS generation needed).
        
        Args:
            clip_number: Subtitle entry number
            source_path: Path to original_clip_N.mp3
            destination_path: Path for translated_clip_N.mp3
        
        Returns:
            TTSClip with COPIED status
        """
        try:
            shutil.copy2(source_path, destination_path)
            
            # Get duration
            duration = self._get_audio_duration(source_path)
            
            return TTSClip(
                clip_number=clip_number,
                original_filename=source_path.name,
                translated_filename=destination_path.name,
                original_text="[[ event ]]",
                translated_text="[[ event ]]",
                original_duration=duration,
                translated_duration=duration,
                duration_ratio=1.0,
                is_event=True,
                processing_status=ProcessingStatus.COPIED,
                output_path=destination_path
            )
        except Exception as e:
            logger.error(f"Failed to copy event clip {clip_number}: {e}")
            return TTSClip(
                clip_number=clip_number,
                original_filename=source_path.name,
                translated_filename=destination_path.name,
                original_text="[[ event ]]",
                translated_text="[[ event ]]",
                original_duration=0.0,
                translated_duration=0.0,
                duration_ratio=0.0,
                is_event=True,
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds using pydub."""
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0
    
    def _convert_audio(self, input_path: Path, output_path: Path, bitrate: str):
        """Convert audio format using FFmpeg."""
        cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-codec:a', 'libmp3lame',
            '-b:a', bitrate,
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
```

---

## 3. TTS Service Configuration

### 3.1 Service Provider

**Primary: IndexTTS2**
- **Repository:** https://github.com/index-tts/index-tts
- **Type:** Local/self-hosted zero-shot TTS with voice cloning
- **Model:** IndexTTS-2 (autoregressive with duration control)
- **Key Features:**
  - Zero-shot voice cloning from reference audio
  - Emotion control (via audio reference or emotion vectors)
  - Duration control (controllable and uncontrollable modes)
  - Multi-lingual support
  - High-quality voice matching
- **Installation Location:** `/app/volumn/.cache/index-tts/` (in container)
- **Best For:** High-quality voice cloning, emotional expressiveness, local processing

### 4.2 IndexTTS2 Installation

**IMPORTANT:** IndexTTS2 should already be installed at `/app/volumn/.cache/index-tts/` in the Docker container.

**Requirements:**
- Python 3.10+
- CUDA-enabled GPU (for reasonable inference speed)
- `uv` package manager (for dependency management)
- Git with Git LFS

**Installation Steps (if not already installed):**
```bash
# 1. Clone repository
cd /app/volumn/.cache
git clone https://github.com/index-tts/index-tts.git
cd index-tts
git lfs pull

# 2. Install uv package manager
pip install -U uv

# 3. Install dependencies (use mirror for faster download in China)
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"

# 4. Download model checkpoints
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# Alternative (if HuggingFace is slow):
export HF_ENDPOINT="https://hf-mirror.com"
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

**Verification:**
```bash
# Test installation
cd /app/volumn/.cache/index-tts
uv run python -c "from indextts.infer_v2 import IndexTTS2; print('IndexTTS2 ready')"
```

**NOTE:** Installation may take 10-30 minutes depending on network speed and GPU availability.

### 4.3 Configuration

**IndexTTS2 Parameters:**
```yaml
tts:
  service: "indextts2"
  model_dir: "/app/volumn/.cache/index-tts/checkpoints"
  config_path: "/app/volumn/.cache/index-tts/checkpoints/config.yaml"
  use_fp16: true          # Enable half-precision (lower VRAM, faster) - RECOMMENDED
  use_cuda_kernel: false  # Compiled CUDA kernels (may speed up, test first)
  use_deepspeed: false    # DeepSpeed acceleration (system-dependent, may be slower)
  use_random: false       # Enable randomness in generation (reduces voice cloning fidelity)
  emo_alpha: 1.0          # Emotion influence (0.0-1.0, use 0.6-0.8 for natural speech)
  output_format: "wav"    # Output format (WAV first, then convert to MP3)
```

**Critical Notes:**
- **No API Key Required** - IndexTTS2 runs locally on GPU
- **Duration Control NOT Available:** IndexTTS2's duration control is not yet enabled in the current release according to README. Generate naturally and handle timing in Step 7.
- **FP16 Recommended:** Significantly faster inference (~2x) with minimal quality loss
- **DeepSpeed:** May help or hurt performance depending on hardware - test on your system
- **Use Random:** Setting to `True` reduces voice cloning quality - keep `False` for best results

---

## 5. Processing Logic Details

### 5.1 Event Detection and Handling

**Event Pattern Recognition:**
```python
import re

def is_pure_event(text: str) -> bool:
    """Check if text is pure event."""
    pattern = r'^\[\[.*\]\]$'
    return bool(re.match(pattern, text.strip()))

# Examples:
is_pure_event("[[ laughing ]]")  # True
is_pure_event("[[ music ]]")     # True
is_pure_event("Hello [[ coughing ]]")  # False
is_pure_event("Welcome")  # False
```

**Processing Decision Tree:**
```
Input: translated_text from SRT_translated.srt

IF is_event_text(translated_text):
    # Copy original audio clip (handled by calling code)
    success = tts_generator.copy_event_clip(
        source_path=original_clip,
        destination_path=translated_clip
    )
    status = COPIED if success else FAILED
ELSE:
    # Generate TTS (handled by calling code)
    success = tts_generator.generate_single_clip(
        reference_audio_path=original_clip,
        translated_text=translated_text,
        output_path=translated_clip,
        emo_alpha=emo_alpha
    )
    status = SUCCESS if success else FAILED
```

### 5.2 Error Handling

**Individual Clip Failures with Retry Queue:**
- Core TTS methods return `bool` (True/False) for success/failure
- **On failure:** Add clip to retry queue instead of immediately giving up
- **Retry mechanism:**
  - Each clip can be retried up to **3 times maximum**
  - Track retry count for each failed clip
  - Process retry queue after main processing loop completes
  - If clip fails after 3 attempts, mark as failed and skip
- **Workflow:**
  1. First pass: Process all clips sequentially (1, 2, 3, ..., N)
  2. On failure: Add to retry queue with retry_count = 1
  3. After first pass: Process retry queue
  4. On retry failure: Increment retry_count
  5. If retry_count < 3: Keep in queue for next retry round
  6. If retry_count >= 3: Give up, mark as permanently failed
  7. Repeat retry rounds until queue is empty or all clips hit max retries
- **Logging:**
  ```
  [TTS] Clip 5 failed, adding to retry queue (attempt 1/3)
  [TTS] Retry queue: [5, 12, 23]
  [TTS] Retrying clip 5 (attempt 2/3)...
  [TTS] Clip 5 succeeded on retry
  [TTS] Clip 12 failed again (attempt 2/3), will retry
  [TTS] Clip 23 failed 3 times, giving up
  ```
- **Statistics tracking:**
  - Track clips that succeeded on retry (and which attempt)
  - Track permanently failed clips
  - Include retry statistics in metadata output

**Critical Failures:**
- Invalid SRT files (mismatched entry counts) - handled by calling code
- Missing input directories - handled by calling code
- Model initialization failures - raised by `IndexTTS2Generator.__init__()`
- All clips failing on first pass - still attempt retries before giving up

**Retry Queue Implementation:**
```python
retry_queue = []  # List of (clip_number, retry_count) tuples
max_retries = 3

# First pass
for i in range(1, total_clips + 1):
    success = process_clip(i)
    if not success:
        retry_queue.append((i, 1))
        logger.warning(f"Clip {i} failed, adding to retry queue (attempt 1/{max_retries})")

# Retry rounds
while retry_queue:
    current_queue = retry_queue[:]
    retry_queue = []
    
    for clip_number, retry_count in current_queue:
        logger.info(f"Retrying clip {clip_number} (attempt {retry_count + 1}/{max_retries})...")
        success = process_clip(clip_number)
        
        if success:
            logger.info(f"Clip {clip_number} succeeded on retry attempt {retry_count + 1}")
        elif retry_count + 1 < max_retries:
            retry_queue.append((clip_number, retry_count + 1))
            logger.warning(f"Clip {clip_number} failed again (attempt {retry_count + 1}/{max_retries}), will retry")
        else:
            logger.error(f"Clip {clip_number} failed {max_retries} times, giving up")
```

### 5.3 Progress Tracking

**Logging (in calling code):**
```
[TTS] Processing clip 1/46: "你好" -> "Hello"
[TTS] Generated translated_clip_1.mp3 (0.6s, ratio: 0.75)
[TTS] Processing clip 2/46: "[[ laughing ]]"
[TTS] Copied event clip 2/46
...
[TTS] Progress: 10/46 (21.7%)
...
[TTS] Completed: 44 success, 2 copied, 0 failed
```

**Progress Tracking (in calling code):**
- Log every clip or every N clips
- Track success/failure counts
- Track duration ratios
- Estimate time remaining

---

## 6. Quality Considerations

### 6.1 Voice Matching

**Ideal Implementation (with voice cloning):**
- Clone speaker's voice from reference audio
- Match tone, pitch, speaking rate
- Preserve emotional characteristics

**Practical Implementation (OpenAI TTS):**
- Use consistent voice across all clips
- Select voice that best approximates speaker characteristics
- Maintain natural prosody and intonation

### 6.2 Audio Quality

**Target Quality:**
- Sample Rate: ≥24kHz (preferably 48kHz)
- Bitrate: ≥192kbps for MP3
- Dynamic Range: Match or exceed reference
- Noise Floor: Minimal background noise

**Quality Checks (in calling code):**
- Validate generated audio is not silent
- Check duration is reasonable (not 0 or excessively long)
- Verify file size is appropriate

### 6.3 Text Processing

**Before TTS Generation (in calling code):**
- Trim whitespace from translated text
- Normalize unicode characters
- Handle special characters appropriately
- Validate text is not empty (except for events)

---

## 7. Edge Cases and Special Handling

### 7.1 Empty or Very Short Text
- **Case:** Translated text is 1-2 characters or empty
- **Action:** Generate TTS anyway, will be very short
- **Note:** May result from translation of interjections

### 7.2 Very Long Text
- **Case:** Translated entry is multiple sentences or very long
- **Action:** Generate as single clip
- **Note:** Duration mismatch will be large, acceptable for Step 7

### 7.3 Mixed Language Content
- **Case:** Translated text contains source language words
- **Action:** TTS service should handle multilingual content
- **Note:** Quality depends on TTS service capabilities

### 7.4 Special Characters and Numbers
- **Case:** Text contains numbers, symbols, abbreviations
- **Action:** TTS should pronounce appropriately for target language
- **Example:** "123" in English vs Chinese pronunciation

### 7.5 Identical Original and Translated Text
- **Case:** No translation occurred (name, technical term, event)
- **Action:** Still generate TTS in target language pronunciation
- **Example:** "API" pronounced differently in English vs Chinese

---

## 8. Testing Requirements

### 8.1 Unit Tests (for Core TTS Generator)

**Test Core TTS Methods:**
- Test `generate_single_clip()` with valid inputs
- Test `is_event_text()` with various patterns
- Test `copy_event_clip()` file operations
- Test error handling (missing files, invalid inputs)

### 8.2 Integration Tests (in Test Script)

**Test Full Processing (in test_step6.py):**
- Process complete set of clips
- Handle mixed content types (events + dialogue)
- Generate metadata correctly
- Verify all outputs exist
- Validate statistics (success/failure counts)

**Test Error Handling:**
- Missing input files
- Mismatched SRT entries
- TTS generation failures
- Permission errors

### 8.3 Quality Tests

**Test Audio Quality:**
- Verify generated audio is valid
- Check duration is reasonable
- Validate file size
- Test playback compatibility

---

## 9. Performance Considerations

### 9.1 Processing Time

**Expected Duration:**
- IndexTTS2 model loading: ~30-60 seconds (one-time at startup)
- TTS generation: ~1-3 seconds per clip (GPU-dependent, with FP16)
- TTS generation: ~2-5 seconds per clip (without FP16)
- Event copying: <0.1 seconds per clip
- Total for 50 clips: ~2-5 minutes (on decent GPU with FP16)

**Optimization Strategies:**
- **Use FP16** (half-precision) for ~2x faster inference and 50% less VRAM - HIGHLY RECOMMENDED
- **Sequential processing ONLY** - No parallel processing (GPU contention, VRAM limits)
- Cache TTS results for repeated text (optional enhancement)
- Skip regeneration if output exists (resume capability - optional)
- **DeepSpeed:** Test on your hardware - may help or hurt performance

**GPU Memory Usage:**
- FP16 mode: ~8-10GB VRAM
- FP32 mode: ~16-20GB VRAM
- Recommendation: Use FP16 unless you have >24GB VRAM

### 9.2 API Rate Limits

**IndexTTS2 (Local):**
- No rate limits (runs locally on GPU)
- Limited by GPU processing speed
- Typical: 1-3 seconds per clip (depends on GPU and clip length)
- Strategy: Sequential processing recommended (parallel may cause VRAM issues)

### 9.3 Resource Usage

**Memory:**
- Load audio files one at a time
- Stream large files when possible
- Clean up temporary files

**Disk Space:**
- Estimate: ~50KB per clip (MP3 192kbps)
- For 100 clips: ~5MB total

---

## 10. Command-Line Interface

### 10.1 Test Script Interface

**File:** `volumn/test_step6.py`

**Separation of Concerns:**
- Test script handles ALL batch processing logic
- Core `IndexTTS2Generator` only handles single-clip TTS
- Test script is responsible for:
  - Loading SRT files
  - Looping through entries
  - Calling TTS generator for each clip
  - Tracking statistics
  - Generating metadata
  - Displaying results

**Basic Usage:**
```bash
# Minimal command
python test_step6.py \
    test/output/test02/step5 \
    test/output/test02/step3/SRT_original.srt \
    test/output/test02/step4/SRT_translated.srt \
    test/output/test02/step6

# With recommended options
python test_step6.py \
    test/output/test02/step5 \
    test/output/test02/step3/SRT_original.srt \
    test/output/test02/step4/SRT_translated.srt \
    test/output/test02/step6 \
    --use-fp16 \
    --emo-alpha 0.8 \
    --verbose
```

**Full Example with All Options:**
```bash
python test_step6.py \
    test/output/test02/step5 \
    test/output/test02/step3/SRT_original.srt \
    test/output/test02/step4/SRT_translated.srt \
    test/output/test02/step6 \
    --model-dir /app/volumn/.cache/index-tts/checkpoints \
    --config-path /app/volumn/.cache/index-tts/checkpoints/config.yaml \
    --use-fp16 \
    --format mp3 \
    --bitrate 192k \
    --source-lang Chinese \
    --target-lang English \
    --emo-alpha 0.8 \
    --verbose
```

### 10.2 Arguments

**Positional (Required):**
1. `original_clips_dir`: Directory containing original_clip_*.mp3 files (from Step 5)
2. `original_srt`: Path to SRT_original.srt (from Step 3)
3. `translated_srt`: Path to SRT_translated.srt (from Step 4)
4. `output_dir`: Output directory for translated clips (will be created if missing)

**Optional Arguments:**
- `--model-dir`: IndexTTS2 model directory  
  Default: `/app/volumn/.cache/index-tts/checkpoints`
  
- `--config-path`: IndexTTS2 config.yaml path  
  Default: `/app/volumn/.cache/index-tts/checkpoints/config.yaml`
  
- `--use-fp16`: Enable FP16 half-precision inference (flag, no value)  
  **RECOMMENDED** - Faster and uses less VRAM
  
- `--use-cuda-kernel`: Enable compiled CUDA kernels (flag, no value)  
  May speed up inference, test on your hardware
  
- `--use-deepspeed`: Enable DeepSpeed acceleration (flag, no value)  
  May help or hurt performance, test first
  
- `--emo-alpha`: Emotion influence factor (float, 0.0-1.0)  
  Default: `1.0`  
  Recommended: `0.6-0.8` for more natural speech
  
- `--use-random`: Enable randomness in generation (flag, no value)  
  Not recommended - reduces voice cloning quality
  
- `--format`: Output audio format  
  Default: `mp3`  
  Options: `mp3`, `wav`, `flac`
  
- `--bitrate`: Audio bitrate for MP3  
  Default: `192k`  
  Options: `128k`, `192k`, `256k`, `320k`
  
- `--source-lang`: Source language name  
  Default: `Unknown`  
  Example: `Chinese`, `English`, `Japanese`
  
- `--target-lang`: Target language name  
  Default: `Unknown`  
  Example: `English`, `Chinese`, `Spanish`
  
- `--verbose`: Enable verbose logging (flag, no value)  
  Shows detailed progress and debug information

### 10.3 Example argparse Implementation

```python
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Step 6: TTS Generation with IndexTTS2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional arguments
    parser.add_argument("original_clips_dir", help="Directory with original_clip_*.mp3 files")
    parser.add_argument("original_srt", help="Path to SRT_original.srt")
    parser.add_argument("translated_srt", help="Path to SRT_translated.srt")
    parser.add_argument("output_dir", help="Output directory for translated clips")
    
    # IndexTTS2 configuration
    parser.add_argument("--model-dir", default="/app/volumn/.cache/index-tts/checkpoints",
                        help="IndexTTS2 model directory")
    parser.add_argument("--config-path", default="/app/volumn/.cache/index-tts/checkpoints/config.yaml",
                        help="IndexTTS2 config.yaml path")
    
    # Performance options
    parser.add_argument("--use-fp16", action="store_true",
                        help="Enable FP16 half-precision (RECOMMENDED)")
    parser.add_argument("--use-cuda-kernel", action="store_true",
                        help="Enable compiled CUDA kernels")
    parser.add_argument("--use-deepspeed", action="store_true",
                        help="Enable DeepSpeed acceleration")
    
    # TTS options
    parser.add_argument("--emo-alpha", type=float, default=1.0,
                        help="Emotion influence (0.0-1.0, default: 1.0)")
    parser.add_argument("--use-random", action="store_true",
                        help="Enable randomness (reduces quality)")
    
    # Output options
    parser.add_argument("--format", default="mp3",
                        choices=["mp3", "wav", "flac"],
                        help="Output audio format")
    parser.add_argument("--bitrate", default="192k",
                        help="Audio bitrate (e.g., 192k)")
    
    # Language options
    parser.add_argument("--source-lang", default="Unknown",
                        help="Source language name")
    parser.add_argument("--target-lang", default="Unknown",
                        help="Target language name")
    
    # Logging
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # ... rest of implementation
```

---

## 11. Success Criteria

**Step 6 is successful when:**
1. ✓ IndexTTS2 is properly installed and models are downloaded
2. ✓ `IndexTTS2Generator` class provides clean single-clip TTS interface
3. ✓ Test script successfully processes all clips in batch
4. ✓ Event clips are correctly identified and copied from originals
5. ✓ Dialogue clips have voice-cloned TTS audio matching speaker characteristics
6. ✓ File naming matches pattern: `translated_clip_{N}.mp3`
7. ✓ Number of output clips equals number of SRT entries
8. ✓ Audio quality is clear and intelligible
9. ✓ Voice cloning successfully matches original speaker's timbre
10. ✓ Metadata file is generated with complete information
11. ✓ Processing completes without critical errors
12. ✓ Core TTS generator is reusable in other contexts (not tied to batch processing)

---

## 12. Dependencies

### 12.1 Python Libraries

**Required in requirements.txt:**
```
pydub>=0.25.1      # Audio file manipulation and duration detection
```

**Already Available (no need to add):**
- `pathlib` - Path handling (built-in)
- `json` - JSON processing (built-in)
- `logging` - Logging (built-in)
- `subprocess` - Process execution (built-in)
- `re` - Regular expressions (built-in)
- `shutil` - File operations (built-in)

**For SRT Parsing (already implemented):**
- Use existing parser: `from utils.srt_handler import load_srt, SRTEntry`

**IndexTTS2 Dependencies:**
- **NOT** in requirements.txt - managed separately by `uv` in IndexTTS2 repo
- IndexTTS2 has its own environment at `/app/volumn/.cache/index-tts/`
- Import from IndexTTS2: `from indextts.infer_v2 import IndexTTS2`
- Required: torch, torchaudio, transformers, accelerate, numpy, scipy, etc.
- All handled by `uv sync --all-extras` in IndexTTS2 directory

### 12.2 External Dependencies

**System Requirements:**
- CUDA Toolkit 12.8+ (for GPU acceleration)
- FFmpeg (for audio format conversion WAV→MP3)
- Git with Git LFS (for model downloads - installation only)
- ~10GB GPU VRAM (recommended for FP16 mode)
- ~20GB GPU VRAM (for FP32 mode)

### 12.3 Import Path Setup

**Critical:** Must add IndexTTS2 to Python path before importing:

```python
import sys
from pathlib import Path

# Add IndexTTS2 to path
INDEXTTS_PATH = Path("/app/volumn/.cache/index-tts")
if str(INDEXTTS_PATH) not in sys.path:
    sys.path.insert(0, str(INDEXTTS_PATH))

# Now can import
from indextts.infer_v2 import IndexTTS2
```

---

## 13. Configuration Example

### 13.1 YAML Configuration

```yaml
step6:
  tts:
    service: "indextts2"
    model_dir: "/app/volumn/.cache/index-tts/checkpoints"
    config_path: "/app/volumn/.cache/index-tts/checkpoints/config.yaml"
    use_fp16: true
    use_cuda_kernel: false
    use_deepspeed: false
    use_random: false
    emo_alpha: 1.0  # Emotion influence (0.0-1.0, lower for more natural)
  
  output:
    format: "mp3"
    bitrate: "192k"
    sample_rate: 24000
  
  processing:
    parallel_workers: 1  # IndexTTS2 uses GPU, parallel may cause issues
    retry_attempts: 3
    retry_delay: 2.0
    skip_existing: false
  
  languages:
    source: "Chinese"
    target: "English"
```

---

## 14. Future Enhancements

### 14.1 Advanced Features
- True voice cloning with reference audio
- Emotion/sentiment preservation
- Multi-speaker detection and handling
- Speed adjustment to match original timing
- Prosody transfer from original audio

### 14.2 Optimization
- Caching TTS results for duplicate text
- Parallel processing with worker pools
- Resume from checkpoint on failure
- Incremental processing (skip existing outputs)

---

**Document Version:** 1.0  
**Last Updated:** November 16, 2025  
**Status:** Draft - Initial Requirements
