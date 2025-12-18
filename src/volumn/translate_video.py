"""
VideoTranslator - Complete Pipeline
Translates video content by processing through all 8 steps automatically.

Usage:
    python translate_video.py <input_video> --target-lang <language> --output-dir <output_dir> [options]

Example:
    python translate_video.py input.mp4 --target-lang English --output-dir output/
"""

import os
import sys
import logging
import argparse
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from processors.video_separator import FFmpegVideoAudioSeparator
from processors.vocal_separator import DemucsVocalSeparator
from processors.speech_separator import SpeechSeparator
from processors.transcriber import WhisperTranscriber
from processors.openai_transcriber import OpenAITranscriber
from processors.subtitle_extractor import SubtitleExtractor
from processors.translator import GPT4Translator
from processors.audio_clipper import FFmpegAudioClipper
from processors.http_tts_generator import HttpTTSGenerator
# Import IndexTTS2Generator only when needed (to avoid import errors in HTTP-only mode)
try:
    from processors.tts_generator import IndexTTS2Generator
    INDEXTTS_AVAILABLE = True
except ImportError:
    INDEXTTS_AVAILABLE = False
from processors.audio_stitcher import AudioStitcher
from processors.video_assembler import VideoAssembler


def setup_logging(output_dir: Path, verbose: bool = False):
    """Configure logging for the pipeline."""
    log_file = output_dir / f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return log_file


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


class VideoTranslationPipeline:
    """Complete video translation pipeline orchestrator."""
    
    def __init__(
        self,
        target_language: str,
        output_dir: Path,
        terminology_file: Optional[Path] = None,
        keep_intermediate: bool = True,
        verbose: bool = False,
        speech_sep_url: str = "http://127.0.0.1:5000",
        whisper_mode: str = "local",
        tts_mode: str = "local",
        tts_api_url: str = "http://localhost:7860",
        source_language: Optional[str] = None,
        target_srt_path: Optional[Path] = None,
        source_srt_path: Optional[Path] = None,
        force_transcribe: bool = False
    ):
        """
        Initialize the pipeline.
        
        Args:
            target_language: Target language for translation
            output_dir: Base output directory
            terminology_file: Optional path to terminology JSON file
            keep_intermediate: Whether to keep intermediate files
            verbose: Enable verbose logging
            tts_mode: TTS generation mode ("local" or "http")
            tts_api_url: TTS API URL when using HTTP mode
            source_language: Optional source language code (e.g., "Spanish", "zh")
            target_srt_path: Optional path to an existing target language subtitle file
            source_srt_path: Optional path to an existing source language subtitle file
            force_transcribe: If True, ignore embedded subtitles and force transcription
        """
        self.target_language = target_language
        self.output_dir = Path(output_dir)
        self.terminology_file = Path(terminology_file) if terminology_file else None
        self.keep_intermediate = keep_intermediate
        self.verbose = verbose
        self.speech_sep_url = speech_sep_url
        self.whisper_mode = whisper_mode
        self.tts_mode = tts_mode
        self.tts_api_url = tts_api_url
        self.source_language = source_language
        self.target_srt_path = Path(target_srt_path) if target_srt_path else None
        self.source_srt_path = Path(source_srt_path) if source_srt_path else None
        self.force_transcribe = force_transcribe
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_dirs = {
            1: self.output_dir / "step1_video_audio_separation",
            2: self.output_dir / "step2_vocal_separation",
            3: self.output_dir / "step3_transcription",
            4: self.output_dir / "step4_translation",
            5: self.output_dir / "step5_audio_clipping",
            6: self.output_dir / "step6_tts_generation",
            7: self.output_dir / "step7_audio_stitching",
            8: self.output_dir / "step8_final_assembly"
        }
        
        for step_dir in self.step_dirs.values():
            step_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = setup_logging(self.output_dir, verbose)
        self.logger = logging.getLogger(__name__)
        
    def run(self, input_video: Path, start_step: int = 1) -> dict:
        """
        Run the complete translation pipeline.
        
        Args:
            input_video: Path to input video file
            start_step: Step to start from (1-8). Default: 1
            
        Returns:
            Dictionary with paths to final outputs
        """
        self.logger.info(f"Starting video translation pipeline for: {input_video}")
        self.logger.info(f"Target language: {self.target_language}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Start step: {start_step}")
        
        start_time = datetime.now()
        
        # Initialize results dictionary with potential breakdown
        step1_result = {}
        step2_result = {}
        step2_5_result = {}
        step3_result = {}
        step4_result = {}
        step5_result = {}
        step6_result = {}
        step7_result = {}
        step8_result = {}
        
        try:
            # Step 1: Video/Audio Separation
            if start_step <= 1:
                print_section("Step 1/8: Video and Audio Separation")
                self.logger.info("Progress: 0/8 steps completed")
                step1_result = self.step1_video_audio_separation(input_video)
                print(f"✓ Step 1 complete - Progress: 1/8 (12.5%)")
                self.logger.info("Progress: 1/8 steps completed (12.5%)")
            else:
                print(f"⏭️  Skipping Step 1 (Start Step: {start_step})")
                # Recover Step 1 state
                step1_result = {
                    'video_only': str(self.step_dirs[1] / 'video_only.mp4'),
                    'audio_full': str(self.step_dirs[1] / 'audio_full.wav')
                }
                # Verification
                if not Path(step1_result['video_only']).exists() or not Path(step1_result['audio_full']).exists():
                    raise FileNotFoundError("Step 1 output files missing. Cannot skip Step 1.")
            
            # Step 2: Vocal Separation
            if start_step <= 2:
                print_section("Step 2/8: Vocal and Background Separation")
                self.logger.info("Progress: 1/8 steps completed")
                step2_result = self.step2_vocal_separation(step1_result['audio_full'])
                print(f"✓ Step 2 complete - Progress: 2/8 (25.0%)")
                self.logger.info("Progress: 2/8 steps completed (25.0%)")
            else:
                print(f"⏭️  Skipping Step 2 (Start Step: {start_step})")
                # Recover Step 2 state
                step2_result = {
                    'vocals': str(self.step_dirs[2] / 'vocals.mp3'),
                    'no_vocals': str(self.step_dirs[2] / 'no_vocals.mp3')
                }
                if not Path(step2_result['vocals']).exists():
                    raise FileNotFoundError("Step 2 output (vocals) missing. Cannot skip Step 2.")
            
            # Step 2.5: Speech Separation
            # Convention: 2.5 is ran if start_step <= 3 (treated as part of pre-transcription)
            # But user requested integer steps 1-8. Let's assume start_step 3 starts AT transcription (Step 3).
            # So separation (2.5) should be grouped with Step 2 generally, OR effectively Step 2.5.
            # However, logic flow:
            # If start_step <= 2: Runs Step 2, then Step 2.5.
            # If start_step == 3: User wants to start at TRANSCRIPTION. So we must SKIP 2.5 AND RECOVER IT.
            
            # Wait, Step 2.5 is intermediate. If user says start_step=3 (Transcription), 
            # they expect inputs for step 3 to be ready. Step 3 input is `step2_5_result['speech']`.
            # So if start_step > 2, we must verify/recover 2.5 output.
            
            if start_step <= 2:
                # Step 2.5 runs effectively after Step 2
                print_section("Step 2.5/8: Speech Separation")
                step2_5_result = self.step2_5_speech_separation(step2_result['vocals'])
                print(f"✓ Step 2.5 complete")
            else:
                # Skipping 2.5 implies we have its output if we are starting at 3 or later
                # Recover Step 2.5 state
                step2_5_dir = self.output_dir / "step2_5_speech_separation"
                step2_5_result = {
                    'speech': str(step2_5_dir / 'vocals_speech.wav'),
                    'non_speech': str(step2_5_dir / 'vocals_no_speech.wav')
                }
                if not Path(step2_5_result['speech']).exists():
                     # If Step 2.5 files are missing but we are starting at 3, maybe we have Step 2 only?
                     # If the user ran up to step 2 before, but not 2.5 (because 2.5 is new), this might fail.
                     # But for now assume strictly sequential previous run.
                     if start_step > 2:
                         raise FileNotFoundError("Step 2.5 output (vocals_speech.wav) missing. Cannot start from Step 3 without it.")

            # Step 3: Transcription (and checking for Target Subtitles)
            if start_step <= 3:
                print_section("Step 3/8: Audio Transcription / Subtitle Checking")
                self.logger.info("Progress: 2/8 steps completed")
                
                # Logic: Check for TARGET language subtitles first (Calculated Priority)
                # If found, we use them for BOTH Step 3 (Sync Source) and Step 4 (Translation Source)
                
                target_subs_found = False
                source_srt_path = self.step_dirs[3] / 'SRT_original.srt'
                
                # A. Check CLI provided target SRT (Priority 1)
                if self.target_srt_path and self.target_srt_path.exists():
                    print(f"\n📄 Using provided target subtitle file: {self.target_srt_path}")
                    shutil.copy2(self.target_srt_path, source_srt_path)
                    target_subs_found = True
                    self.logger.info("Using provided target SRT file.")
                    
                # B. Check for Embedded TARGET language SRT (Priority 2)
                elif input_video and not self.force_transcribe:
                    print(f"\n🔍 Checking for embedded {self.target_language} subtitles in video...")
                    extractor = SubtitleExtractor()
                    success = extractor.process(
                        video_path=input_video,
                        output_path=source_srt_path,
                        target_language=self.target_language 
                    )
                    if success:
                        print(f"✓ Found and extracted embedded {self.target_language} subtitles!")
                        target_subs_found = True
                        self.logger.info(f"Found embedded target language ({self.target_language}) subtitles.")
                
                step3_result = {}
                
                if target_subs_found:
                    print(f"⏭️  Skipping Transcription (Target subtitles available)")
                    step3_result = {
                        'srt_file': str(source_srt_path),
                        'is_target_language': True
                    }
                else:
                    # C. Check CLI provided SOURCE SRT (Priority 3)
                    source_subs_found = False
                    if self.source_srt_path and self.source_srt_path.exists():
                        print(f"\n📄 Using provided source subtitle file: {self.source_srt_path}")
                        shutil.copy2(self.source_srt_path, source_srt_path)
                        source_subs_found = True
                        self.logger.info("Using provided source SRT file.")
                    
                    if source_subs_found:
                        print(f"⏭️  Skipping Transcription (Source subtitles provided)")
                        step3_result = {'srt_file': str(source_srt_path), 'is_target_language': False}
                    else:
                        # D. Normal Flow: Embedded Source OR Transcribe
                        
                        # 1. Try extracting embedded SOURCE subtitles 
                        extracted_source = False
                        if input_video and not self.force_transcribe:
                            print(f"\n🔍 Checking for embedded source subtitles (fallback)...")
                            extractor = SubtitleExtractor()
                            extracted_source = extractor.process(
                                video_path=input_video,
                                output_path=source_srt_path,
                                target_language=self.source_language
                            )
                        
                        if extracted_source:
                            print(f"✓ Extracted embedded source subtitles to: {source_srt_path.name}")
                            self.logger.info("Successfully extracted embedded source subtitles.")
                            step3_result = {'srt_file': str(source_srt_path), 'is_target_language': False}
                        else:
                            # 2. Fallback to Whisper transcription
                            print(f"\n🎤 Transcribing audio to text...")
                            
                            if self.whisper_mode == "openai":
                                print(f"⚙️  Using OpenAI Whisper API")
                                transcriber = OpenAITranscriber(model_name='whisper-1')
                            else:
                                print(f"⚙️  Using Whisper (medium model) on GPU")
                                transcriber = WhisperTranscriber(model_name='medium', device='cuda')
                            
                            print("⏳ Processing audio, this may take a few minutes...")
                            result = transcriber.transcribe(
                                audio_path=Path(step2_5_result['speech']),
                                output_path=source_srt_path,
                                language=self.source_language,
                                verbose=True
                            )
                            
                            if not result.success:
                                raise Exception(f"Transcription failed: {result.error_message}")
                            
                            print(f"✓ Detected language: {result.detected_language}")
                            print(f"✓ SRT file: {source_srt_path.name}")
                            step3_result = {'srt_file': str(result.srt_path), 'is_target_language': False}

                print(f"✓ Step 3 complete - Progress: 3/8 (37.5%)")
                self.logger.info("Progress: 3/8 steps completed (37.5%)")
            else:
                print(f"⏭️  Skipping Step 3 (Start Step: {start_step})")
                step3_result = {
                    'srt_file': str(self.step_dirs[3] / 'SRT_original.srt')
                }
                if not Path(step3_result['srt_file']).exists():
                    raise FileNotFoundError("Step 3 output (SRT) missing. Cannot skip Step 3.")
            
            # Step 4: Translation
            if start_step <= 4:
                print_section("Step 4/8: Subtitle Translation")
                self.logger.info("Progress: 3/8 steps completed")
                
                # Check if Step 3 provided us with target language subs directly
                if step3_result.get('is_target_language', False):
                    print(f"⏭️  Skipping Translation (Subtitles are already in target language)")
                    
                    target_path = self.step_dirs[4] / 'SRT_translated.srt'
                    shutil.copy2(step3_result['srt_file'], target_path)
                    
                    print(f"✓ Using existing subtitles: {target_path.name}")
                    step4_result = {
                        'srt_translated': str(target_path)
                    }
                else:
                    step4_result = self.step4_translation(step3_result['srt_file'])
                
                print(f"✓ Step 4 complete - Progress: 4/8 (50.0%)")
                self.logger.info("Progress: 4/8 steps completed (50.0%)")
            else:
                print(f"⏭️  Skipping Step 4 (Start Step: {start_step})")
                step4_result = {
                    'srt_translated': str(self.step_dirs[4] / 'SRT_translated.srt')
                }
                if not Path(step4_result['srt_translated']).exists():
                    raise FileNotFoundError("Step 4 output (Translated SRT) missing. Cannot skip Step 4.")
            
            # Step 5: Audio Clipping
            if start_step <= 5:
                print_section("Step 5/8: Audio Clipping")
                self.logger.info("Progress: 4/8 steps completed")
                step5_result = self.step5_audio_clipping(
                    step2_result['vocals'],
                    step3_result['srt_file']
                )
                print(f"✓ Step 5 complete - Progress: 5/8 (62.5%)")
                self.logger.info("Progress: 5/8 steps completed (62.5%)")
            else:
                print(f"⏭️  Skipping Step 5 (Start Step: {start_step})")
                step5_result = {
                    'clips_dir': str(self.step_dirs[5])
                }
                if not Path(step5_result['clips_dir']).exists():
                     # Check for at least one file?
                    raise FileNotFoundError("Step 5 output directory missing. Cannot skip Step 5.")
            
            # Step 6: TTS Generation
            if start_step <= 6:
                print_section("Step 6/8: TTS Generation")
                self.logger.info("Progress: 5/8 steps completed")
                step6_result = self.step6_tts_generation(
                    step5_result['clips_dir'],
                    step3_result['srt_file'],
                    step4_result['srt_translated']
                )
                print(f"✓ Step 6 complete - Progress: 6/8 (75.0%)")
                self.logger.info("Progress: 6/8 steps completed (75.0%)")
            else:
                print(f"⏭️  Skipping Step 6 (Start Step: {start_step})")
                step6_result = {
                    'translated_clips_dir': str(self.step_dirs[6])
                }
                if not Path(step6_result['translated_clips_dir']).exists():
                    raise FileNotFoundError("Step 6 output directory missing. Cannot skip Step 6.")
            
            # Step 7: Audio Stitching
            if start_step <= 7:
                print_section("Step 7/8: Audio Track Stitching")
                self.logger.info("Progress: 6/8 steps completed")
                step7_result = self.step7_audio_stitching(
                    step6_result['translated_clips_dir'],
                    step2_result['no_vocals'],
                    step3_result['srt_file'],
                    step2_5_result['non_speech']
                )
                print(f"✓ Step 7 complete - Progress: 7/8 (87.5%)")
                self.logger.info("Progress: 7/8 steps completed (87.5%)")
            else:
                print(f"⏭️  Skipping Step 7 (Start Step: {start_step})")
                step7_result = {
                    'audio_translated_full': str(self.step_dirs[7] / 'audio_translated_full.mp3')
                }
                if not Path(step7_result['audio_translated_full']).exists():
                    raise FileNotFoundError("Step 7 output (Audio) missing. Cannot skip Step 7.")
            
            # Step 8: Final Video Assembly
            if start_step <= 8:
                print_section("Step 8/8: Final Video Assembly")
                self.logger.info("Progress: 7/8 steps completed")
                step8_result = self.step8_video_assembly(
                    step1_result['video_only'],
                    step7_result['audio_translated_full'],
                    step4_result['srt_translated']
                )
                print(f"✓ Step 8 complete - Progress: 8/8 (100%)")
                self.logger.info("Progress: 8/8 steps completed (100%)")
            
            # Copy translated subtitles to final output
            final_srt = self.output_dir / "SRT_translated.srt"
            shutil.copy2(step4_result['srt_translated'], final_srt)
            
            # Calculate total time
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            print_section("Translation Complete!")
            print(f"\n✅ Video translation completed successfully!")
            print(f"\n📹 Final Output:")
            print(f"  Video: {step8_result['video_translated']}")
            print(f"  Subtitles: {final_srt}")
            print(f"\n⏱️  Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"\n📋 Log file: {self.log_file}")
            
            # Cleanup intermediate files if requested
            if not self.keep_intermediate:
                print("\n🧹 Cleaning up intermediate files...")
                self.cleanup_intermediate_files()
            
            return {
                'video_translated': step8_result['video_translated'],
                'srt_translated': str(final_srt),
                'log_file': str(self.log_file),
                'total_time': total_time
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            print(f"\n❌ Pipeline failed: {e}")
            raise
    
    def step1_video_audio_separation(self, input_video: Path) -> dict:
        """Step 1: Separate video and audio tracks."""
        self.logger.info("Step 1: Video/Audio separation")
        print(f"\n🎬 Processing video: {input_video.name}")
        print(f"📂 Output directory: {self.step_dirs[1]}")
        
        separator = FFmpegVideoAudioSeparator()
        print("⚙️  Separating video and audio tracks...")
        result = separator.separate(
            input_video_path=input_video,
            output_dir=self.step_dirs[1],
            overwrite=True
        )
        
        if not result.success:
            raise Exception(f"Video/Audio separation failed: {result.error_message}")
        
        print(f"✓ Video track: {Path(result.video_path).name}")
        print(f"✓ Audio track: {Path(result.audio_path).name}")
        
        return {
            'video_only': str(result.video_path),
            'audio_full': str(result.audio_path)
        }
    
    def step2_vocal_separation(self, audio_file: str) -> dict:
        """Step 2: Separate vocals from background."""
        self.logger.info("Step 2: Vocal separation")
        print(f"\n🎵 Separating vocals from background audio...")
        print(f"⚙️  Using Demucs AI model (GPU mode for faster processing)")
        
        # Use GPU mode with segment size within model limits (max 7.8)
        separator = DemucsVocalSeparator(
            device='cuda',
            segment_size=7  # Within model limit of 7.8
        )
        print("⏳ Processing on GPU - this will be much faster...")
        vocals, no_vocals = separator.process(
            input_path=audio_file,
            output_dir=str(self.step_dirs[2])
        )
        
        print(f"✓ Vocals: {Path(vocals).name}")
        print(f"✓ Background: {Path(no_vocals).name}")
        
        return {
            'vocals': vocals,
            'no_vocals': no_vocals
        }

    def step2_5_speech_separation(self, vocal_file: str) -> dict:
        """Step 2.5: Separate speech from non-speech vocals."""
        self.logger.info("Step 2.5: Speech separation")
        print(f"\n🗣️  Separating speech from vocals...")
        print(f"⚙️  Using external Speech Separation service at {self.speech_sep_url}")
        
        separator = SpeechSeparator(api_url=self.speech_sep_url)
        output_dir = self.output_dir / "step2_5_speech_separation"
        
        print("⏳ Sending audio to separation service...")
        speech, non_speech = separator.process(
            input_path=vocal_file,
            output_dir=str(output_dir)
        )
        
        print(f"✓ Speech track: {Path(speech).name}")
        print(f"✓ Non-speech track: {Path(non_speech).name}")
        
        return {
            'speech': speech,
            'non_speech': non_speech
        }
    
    def step3_transcription(self, vocal_file: str, input_video: Optional[Path] = None) -> dict:
        """Step 3: Transcribe vocals to text OR extract embedded subtitles."""
        self.logger.info("Step 3: Transcription")
        
        srt_output = self.step_dirs[3] / 'SRT_original.srt'
        
        # 1. Try extracting embedded subtitles first
        if input_video and not self.force_transcribe:
            print(f"\n🔍 Checking for embedded subtitles in {input_video.name}...")
            extractor = SubtitleExtractor()
            success = extractor.process(
                video_path=input_video,
                output_path=srt_output,
                target_language=self.source_language
            )
            
            if success:
                print(f"✓ Extracted embedded subtitles to: {srt_output.name}")
                self.logger.info("Successfully extracted embedded subtitles. Skipping Whisper transcription.")
                return {
                    'srt_file': str(srt_output)
                }
            else:
                print(f"ℹ️  No suitable embedded subtitles found.")
        
        # 2. Fallback to Whisper transcription
        print(f"\n🎤 Transcribing audio to text...")
        
        if self.whisper_mode == "openai":
            print(f"⚙️  Using OpenAI Whisper API")
            transcriber = OpenAITranscriber(
                model_name='whisper-1'
            )
        else:
            print(f"⚙️  Using Whisper (medium model) on GPU")
            transcriber = WhisperTranscriber(
                model_name='medium',
                device='cuda'
            )
        
        srt_output = self.step_dirs[3] / 'SRT_original.srt'
        print("⏳ Processing audio, this may take a few minutes...")
        result = transcriber.transcribe(
            audio_path=Path(vocal_file),
            output_path=srt_output,
            language=self.source_language,  # Use specified source language if provided
            verbose=True    # Enable Whisper progress output
        )
        
        if not result.success:
            raise Exception(f"Transcription failed: {result.error_message}")
        
        print(f"✓ Detected language: {result.detected_language}")
        print(f"✓ Segments: {result.segment_count}")
        print(f"✓ SRT file: {srt_output.name}")
        
        return {
            'srt_file': str(result.srt_path)
        }
    
    def step4_translation(self, srt_file: str) -> dict:
        """Step 4: Translate subtitles."""
        self.logger.info("Step 4: Translation")
        print(f"\n🌐 Translating subtitles to {self.target_language}...")
        print(f"⚙️  Using OpenAI GPT for translation")
        
        translator = GPT4Translator(
            # Hardcoded API key for local dev as requested
            api_key=os.getenv('OPENAI_API_KEY'),
            target_language=self.target_language
        )
        
        input_data = {
            'srt_file': srt_file,
            'output_dir': str(self.step_dirs[4]),
            'source_language': self.source_language
        }
        
        # Add terminology file if provided
        if self.terminology_file:
            input_data['terminology_file'] = str(self.terminology_file)
            print(f"📖 Using terminology file: {self.terminology_file.name}")
            self.logger.info(f"Using terminology file: {self.terminology_file}")
        
        print("⏳ Sending to translation API...")
        result = translator.process(input_data)
        
        print(f"✓ Translation complete: {Path(result['srt_translated']).name}")
        
        return {
            'srt_translated': result['srt_translated']
        }
    
    def step5_audio_clipping(self, vocal_file: str, srt_file: str) -> dict:
        """Step 5: Clip audio into segments."""
        self.logger.info("Step 5: Audio clipping")
        print(f"\n✂️  Clipping audio into segments...")
        print(f"⚙️  Using FFmpeg for precise audio extraction")
        
        clipper = FFmpegAudioClipper()
        print("⏳ Extracting audio clips based on subtitle timestamps...")
        result = clipper.clip_audio(
            vocal_audio_path=Path(vocal_file),
            srt_path=Path(srt_file),
            output_dir=self.step_dirs[5],
            output_format='mp3'
        )
        
        # Count clips
        clips = list(self.step_dirs[5].glob('original_clip_*.mp3'))
        print(f"✓ Created {len(clips)} audio clips")
        
        return {
            'clips_dir': str(self.step_dirs[5])
        }
    
    def step6_tts_generation(
        self,
        original_clips_dir: str,
        original_srt: str,
        translated_srt: str
    ) -> dict:
        """Step 6: Generate TTS for translated text with retry mechanism."""
        self.logger.info("Step 6: TTS generation")
        print(f"\n🎶 Generating voice-cloned translated audio...")
        print(f"⚙️  Using IndexTTS2 for voice cloning")
        
        # Load SRT files
        from utils.srt_handler import load_srt
        original_entries = load_srt(original_srt)
        translated_entries = load_srt(translated_srt)
        
        total_clips = len(original_entries)
        print(f"📄 Processing {total_clips} clips...")
        
        # Initialize TTS generator based on mode
        if self.tts_mode == "http":
            self.logger.info(f"Using HTTP TTS API at {self.tts_api_url}")
            tts_generator = HttpTTSGenerator(
                api_url=self.tts_api_url,
                default_format="mp3",
                default_bitrate="320k"
            )
        else:  # local mode
            if not INDEXTTS_AVAILABLE:
                raise RuntimeError(
                    "IndexTTS2 is not available. Either:\n"
                    "  1. Use --tts-mode http to call external TTS service, or\n"
                    "  2. Install IndexTTS2 in the container"
                )
            self.logger.info("Using local IndexTTS2 model")
            tts_generator = IndexTTS2Generator(use_fp16=True)
        
        # Setup
        original_clips_dir = Path(original_clips_dir)
        output_dir = self.step_dirs[6]
        max_retries = 3
        retry_queue = []  # List of (clip_number, retry_count) tuples
        
        succeeded = 0
        copied_events = 0
        
        # First pass: process all clips
        for i, (orig_entry, trans_entry) in enumerate(zip(original_entries, translated_entries), start=1):
            original_clip = original_clips_dir / f"original_clip_{i}.mp3"
            translated_clip = output_dir / f"translated_clip_{i}.mp3"
            
            if not original_clip.exists():
                self.logger.warning(f"Original clip {i} not found, skipping")
                continue
            
            # Check if event
            if tts_generator.is_event_text(trans_entry.text):
                success = tts_generator.copy_event_clip(original_clip, translated_clip)
                if success:
                    copied_events += 1
                else:
                    retry_queue.append((i, 1))
                    self.logger.warning(f"Clip {i} (event) failed to copy, adding to retry queue (attempt 1/{max_retries})")
            else:
                success = tts_generator.generate_single_clip(
                    reference_audio_path=original_clip,
                    translated_text=trans_entry.text,
                    output_path=translated_clip,
                    match_duration=True,
                    time_stretch_tool='ffmpeg'
                )
                if success:
                    succeeded += 1
                else:
                    retry_queue.append((i, 1))
                    self.logger.warning(f"Clip {i} failed, adding to retry queue (attempt 1/{max_retries})")
            
            # Log progress every 5 clips
            if i % 5 == 0 or i == total_clips:
                progress_pct = (i / total_clips) * 100
                print(f"  Progress: {i}/{total_clips} clips ({progress_pct:.1f}%) - {succeeded} generated, {copied_events} events")
                self.logger.info(f"Progress: {i}/{total_clips} clips processed")
        
        # Retry rounds
        retry_round = 1
        while retry_queue:
            print(f"\n🔄 Retry round {retry_round}, {len(retry_queue)} clips in queue...")
            self.logger.info(f"Starting retry round {retry_round}, queue size: {len(retry_queue)}")
            current_queue = retry_queue[:]
            retry_queue = []
            
            for clip_number, retry_count in current_queue:
                i = clip_number
                orig_entry = original_entries[i - 1]
                trans_entry = translated_entries[i - 1]
                original_clip = original_clips_dir / f"original_clip_{i}.mp3"
                translated_clip = output_dir / f"translated_clip_{i}.mp3"
                
                self.logger.info(f"Retrying clip {i} (attempt {retry_count + 1}/{max_retries})...")
                
                # Retry generation
                if tts_generator.is_event_text(trans_entry.text):
                    success = tts_generator.copy_event_clip(original_clip, translated_clip)
                else:
                    success = tts_generator.generate_single_clip(
                        reference_audio_path=original_clip,
                        translated_text=trans_entry.text,
                        output_path=translated_clip,
                        match_duration=True,
                        time_stretch_tool='ffmpeg'
                    )
                
                if success:
                    self.logger.info(f"✓ Clip {i} succeeded on retry attempt {retry_count + 1}")
                    if tts_generator.is_event_text(trans_entry.text):
                        copied_events += 1
                    else:
                        succeeded += 1
                elif retry_count + 1 < max_retries:
                    retry_queue.append((i, retry_count + 1))
                    self.logger.warning(f"✗ Clip {i} failed again (attempt {retry_count + 1}/{max_retries}), will retry")
                else:
                    self.logger.error(f"✗ Clip {i} failed {max_retries} times, giving up")
            
            retry_round += 1
        
        # Final statistics
        failed = total_clips - succeeded - copied_events
        print(f"\n✓ TTS generation complete:")
        print(f"  - {succeeded} clips generated")
        print(f"  - {copied_events} events copied")
        print(f"  - {failed} failed")
        self.logger.info(f"TTS generation completed: {succeeded} succeeded, {copied_events} events copied, {failed} failed")
        
        return {
            'translated_clips_dir': str(output_dir)
        }
    
    def step7_audio_stitching(
        self,
        translated_clips_dir: str,
        no_vocals_file: str,
        srt_file: str,
        non_speech_file: str
    ) -> dict:
        """Step 7: Stitch translated clips with background."""
        self.logger.info("Step 7: Audio stitching")
        print(f"\n🎵 Stitching translated audio with background...")
        print(f"⚙️  Combining clips with original background audio (and non-speech vocals)")
        
        # Increase fade durations to prevent audio pops/clicks
        stitcher = AudioStitcher(
            fade_in_ms=100,  # 100ms fade-in to prevent audio pops
            fade_out_ms=100,  # 100ms fade-out to prevent audio pops
            time_stretch_tool='ffmpeg'
        )
        print("⏳ Merging all audio components...")
        result = stitcher.process({
            'translated_clips_dir': translated_clips_dir,
            'no_vocals_file': no_vocals_file,
            'srt_file': srt_file,
            'output_dir': str(self.step_dirs[7]),
            'non_speech_file': non_speech_file
        })
        
        print(f"✓ Complete audio track: {Path(result['audio_translated_full']).name}")
        
        return {
            'audio_translated_full': result['audio_translated_full']
        }
    
    def step8_video_assembly(self, video_file: str, audio_file: str, srt_file: str) -> dict:
        """Step 8: Combine video with translated audio."""
        self.logger.info("Step 8: Final video assembly")
        print(f"\n🎬 Assembling final translated video...")
        print(f"⚙️  Merging video with translated audio track")
        
        assembler = VideoAssembler()
        print("⏳ Creating final video file...")
        result = assembler.process({
            'video_file': video_file,
            'audio_file': audio_file,
            'output_dir': str(self.step_dirs[8]),
            'srt_file': srt_file
        })
        
        print(f"✓ Final video: {Path(result['video_translated']).name}")
        
        return {
            'video_translated': result['video_translated']
        }
    
    def cleanup_intermediate_files(self):
        """Remove intermediate step directories."""
        for step_num in range(1, 8):
            try:
                shutil.rmtree(self.step_dirs[step_num])
                self.logger.info(f"Removed step {step_num} intermediate files")
            except Exception as e:
                self.logger.warning(f"Failed to remove step {step_num} files: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VideoTranslator - Translate video content with voice cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python translate_video.py input.mp4 --target-lang English --output-dir output/
  
  # With terminology file
  python translate_video.py input.mp4 --target-lang Spanish --output-dir output/ --terminology terms.json
  
  # With cleanup of intermediate files
  python translate_video.py input.mp4 --target-lang Spanish --output-dir output/ --cleanup
  
  # Verbose mode
  python translate_video.py input.mp4 --target-lang Chinese --output-dir output/ --verbose
        """
    )
    
    parser.add_argument("input_video",
                        help="Path to input video file")
    parser.add_argument("--target-lang", "-t",
                        required=True,
                        help="Target language for translation (e.g., English, Spanish, Chinese)")
    parser.add_argument("--output-dir", "-o",
                        default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--terminology", "-T",
                        help="Path to terminology JSON file (optional)")
    parser.add_argument("--speech-sep-url",
                        default="http://127.0.0.1:5000",
                        help="Speech separation service URL (default: http://127.0.0.1:5000)")
    parser.add_argument("--target-srt",
                        help="Path to existing target language subtitle file (skips transcription/translation)")
    parser.add_argument("--source-srt",
                        help="Path to existing source language subtitle file (skips transcription, performs translation)")
    parser.add_argument("--whisper-mode",
                        choices=["local", "openai"],
                        default="openai",
                        help="Transcription mode: 'local' (GPU) or 'openai' (API) (default: openai)")
    parser.add_argument("--tts-mode",
                        choices=["local", "http"],
                        default="http",
                        help="TTS generation mode: 'local' uses IndexTTS2 in container, 'http' calls external API (default: http)")
    parser.add_argument("--tts-api-url",
                        default="http://localhost:7860",
                        help="TTS API URL when using --tts-mode=http (default: http://localhost:7860)")
    parser.add_argument("--start-step", "-s",
                        type=int,
                        default=1,
                        help="Step to start from (1-8). Default: 1")
    parser.add_argument("--source-lang", "-S",
                        help="Source language of the video (optional, improves transcription accuracy)")
    parser.add_argument("--force-transcribe",
                        action="store_true",
                        help="Force transcription even if embedded subtitles are present")
    parser.add_argument("--cleanup", "-c",
                        action="store_true",
                        help="Remove intermediate files after completion")
    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate input video
    input_video = Path(args.input_video)
    if not input_video.exists():
        print(f"❌ Error: Input video not found: {input_video}")
        sys.exit(1)
        
    # Validate start step
    if args.start_step < 1 or args.start_step > 8:
        print(f"❌ Error: Start step must be between 1 and 8. Got: {args.start_step}")
        sys.exit(1)
    
    # Validate terminology file if provided
    terminology_file = None
    if args.terminology:
        terminology_file = Path(args.terminology)
        if not terminology_file.exists():
            print(f"❌ Error: Terminology file not found: {terminology_file}")
            sys.exit(1)
    
    # Run pipeline
    pipeline = VideoTranslationPipeline(
        target_language=args.target_lang,
        output_dir=Path(args.output_dir),
        terminology_file=terminology_file,
        keep_intermediate=not args.cleanup,
        verbose=args.verbose,
        speech_sep_url=args.speech_sep_url,
        whisper_mode=args.whisper_mode,
        tts_mode=args.tts_mode,
        tts_api_url=args.tts_api_url,
        source_language=args.source_lang,
        target_srt_path=args.target_srt,
        source_srt_path=args.source_srt,
        force_transcribe=args.force_transcribe
    )
    
    try:
        result = pipeline.run(input_video, start_step=args.start_step)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Translation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
