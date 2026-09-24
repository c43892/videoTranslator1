import hashlib
import json
from dataclasses import asdict
from . import media
from .domain import Segment, Stems, NeedsReview, SCHEMA_VERSION
from .segmentation import parse_words, segment_dialogue, SEGMENTATION_VERSION
from .punctuation import apply_punctuation
from .references import segment_reference

PIPELINE_VERSION = '2026-09-24.legacy-alignment-v1'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]

class Pipeline:
    def __init__(self, config, storage, separator, transcriber, translator, synthesizer, punctuator=None):
        self.config, self.storage = config, storage
        self.separator, self.transcriber = separator, transcriber
        self.translator, self.synthesizer = translator, synthesizer
        self.punctuator = punctuator

    def run(self, job, report, check_cancel):
        storage, config = self.storage, self.config
        fingerprint = digest({'version': PIPELINE_VERSION, 'config': config.pipeline_config(),
                              'target': job.target_language, 'terminology': job.terminology,
                              'input': job.input_key})
        prefix = f'jobs/{job.id}/runs/{fingerprint}'
        manifest_key = f'jobs/{job.id}/manifest.json'
        manifest = storage.read_json(manifest_key) if storage.exists(manifest_key) else {}
        if manifest.get('fingerprint') != fingerprint:
            manifest = {'schema_version': SCHEMA_VERSION, 'pipeline_version': PIPELINE_VERSION,
                        'fingerprint': fingerprint, 'config': config.pipeline_config(),
                        'providers': {'separation': config.separation_label(),
                                      'transcription': config.transcription_model(), 'diarization': 'disabled',
                                      'translation': config.deepseek_model,
                                      'synthesis': 'IndexTTS2 v2.0.0 / 830f6f8'},
                        'completed': {}, 'segments': [], 'warnings': []}
        def save():
            storage.write_json(manifest_key, manifest)
        def stage(name, progress):
            check_cancel()
            report(name, progress)
        def remote_checkpoint(value):
            manifest['remote_separation'] = value
            save()
        def save_segments():
            manifest['segments'] = [s.to_dict() for s in segments]
            save()
        def preserve_original(segment):
            segment.render_status = 'original'
            segment.aligned_audio = ''

        stage('prepare', 5)
        video = storage.path(job.input_key)
        metadata = media.probe(video)
        duration = float(metadata['format']['duration'])
        if duration <= 0 or duration > config.max_video_seconds:
            raise ValueError('Video duration exceeds the configured limit')
        if not any(s['codec_type'] == 'video' for s in metadata['streams']) or not any(s['codec_type'] == 'audio' for s in metadata['streams']):
            raise ValueError('The file must contain both video and audio')
        manifest['duration'] = duration
        manifest['stem_layout'] = ('dialogue + combined background; effects is a silent compatibility track'
                                  if config.separation_provider == 'demucs' else 'dialogue + music + effects')
        audio_key = f'{prefix}/source.flac'
        if not manifest['completed'].get('prepare') or not storage.exists(audio_key):
            media.extract_audio(video, storage.path(audio_key))
            manifest['completed']['prepare'] = True
            save()

        stage('separate', 12)
        stem_keys = {name: f'{prefix}/stems/{name}.wav' for name in ('dialogue', 'music', 'effects')}
        if not manifest['completed'].get('separate') or not all(storage.exists(k) for k in stem_keys.values()):
            stems = self.separator.separate(audio_key, prefix, remote_checkpoint, manifest.get('remote_separation'), check_cancel)
            for name, key in stem_keys.items():
                check_cancel()
                media.canonical_audio(storage.path(getattr(stems, name)), storage.path(key))
                actual = float(media.probe(storage.path(key))['format']['duration'])
                if abs(actual-duration) > 0.2:
                    raise ValueError(f'{name} stem is not aligned with the source duration')
            manifest['completed']['separate'] = True
            manifest['stems'] = stem_keys
            save()
        stems = Stems(**stem_keys)

        stage('transcribe', 25)
        transcript_key = f'{prefix}/transcript.json'
        cached_transcript = storage.read_json(transcript_key) if storage.exists(transcript_key) else None
        if (cached_transcript is None or (config.transcription_provider == 'scribe'
                and cached_transcript.get('turn_detection') != 'boundary_only')):
            storage.write_json(transcript_key, self.transcriber.transcribe(stems.dialogue))
        transcript = storage.read_json(transcript_key)
        manifest['source_language'] = transcript.get('language_code', '')
        manifest['providers']['turn_detection'] = transcript.get('turn_detection', 'unavailable')
        manifest['transcript'] = transcript_key

        stage('segment', 35)
        # Keep upstream separation/ASR caches; changed segment IDs must never
        # reuse old translations, clips, synthesis or user edits.
        prefix += '/' + SEGMENTATION_VERSION
        words = parse_words(transcript)
        punctuation_warnings = []
        if transcript.get('text', '').strip() and not words:
            raise NeedsReview('Transcription contains text but no usable timed words')
        if self.punctuator and words and transcript.get('provider') != 'elevenlabs':
            punctuation_key = f'{prefix}/punctuation.json'
            if not storage.exists(punctuation_key):
                text = self.punctuator.restore(words)
                apply_punctuation(words, text)
                storage.write_json(punctuation_key, {'text': text, 'warnings': getattr(self.punctuator, 'warnings', [])})
            punctuation = storage.read_json(punctuation_key)
            words = apply_punctuation(words, punctuation['text'])
            punctuation_warnings = punctuation.get('warnings', [])
            manifest['punctuation'] = punctuation_key
        previous_segmentation = manifest.get('segmentation_version')
        manifest['segmentation_version'] = SEGMENTATION_VERSION
        events = transcript.get('audio_events', [])
        manifest['audio_events'] = events
        manifest['turn_boundaries'] = transcript.get('turn_boundaries', [])
        segments = segment_dialogue(words, config.emotion_reference_max_seconds, audio_events=events,
                                    turn_boundaries=manifest['turn_boundaries'],
                                    utterances=transcript.get('utterances', []))
        manifest['warnings'] = list(punctuation_warnings)
        if not segments:
            manifest['warnings'].append('未识别到可替换的对白，保留原音轨。')
        for segment in segments:
            check_cancel()
            segment.original_audio = f'{prefix}/original/{segment.id}.wav'
            segment.emotion_reference = segment.original_audio
            if segment.end <= segment.start:
                continue
            if segment.end > duration + 0.05:
                segment.flags.append('timestamp_outside_video')
                continue
            if not storage.exists(segment.original_audio):
                media.clip(storage.path(stems.dialogue), storage.path(segment.original_audio), segment.start, segment.end)
            reference = segment_reference(segment, storage, prefix, config.emotion_reference_max_seconds)
            segment.speaker_reference = segment.emotion_reference = reference

        overrides_key = f'jobs/{job.id}/overrides.json'
        overrides = storage.read_json(overrides_key) if storage.exists(overrides_key) else {}
        if overrides and overrides.get('segmentation_version') != SEGMENTATION_VERSION:
            # Legacy overrides are safe only when they were made against the
            # same segmentation already recorded in this manifest.
            if previous_segmentation != SEGMENTATION_VERSION:
                storage.write_json(f'{prefix}/superseded-overrides.json', overrides)
                overrides = {}
            else:
                overrides['segmentation_version'] = SEGMENTATION_VERSION
            storage.write_json(overrides_key, overrides)
        manifest.pop('speakers', None)
        manifest['providers']['diarization'] = 'disabled'
        save_segments()

        stage('translate', 45)
        translated_key = f'{prefix}/translations.json'
        translations = storage.read_json(translated_key) if storage.exists(translated_key) else {}
        # Process complete context batches and checkpoint after every accepted response.
        for start in range(0, len(segments), 16):
            check_cancel()
            batch = segments[start:start+16]
            if any(s.id not in translations for s in batch):
                translations.update(self.translator.translate(batch, job.target_language, job.terminology))
                storage.write_json(translated_key, translations)
        for segment in segments:
            segment.translation = overrides.get('translations', {}).get(segment.id, translations[segment.id])
        for start in range(0, len(segments), 100):
            batch = segments[start:start+100]
            counts = self.synthesizer.tokenize([s.translation for s in batch])
            if len(counts) != len(batch):
                raise ValueError('TTS tokenizer returned mismatched results')
            for segment, count in zip(batch, counts):
                if count > config.tts_max_text_tokens:
                    segment.notes.append(f'synthesis_internal_chunks:{count}_tokens')
        save_segments()

        stage('synthesize', 55)
        for index, segment in enumerate(segments):
            check_cancel()
            if segment.flags:
                preserve_original(segment)
                save_segments()
                continue
            cache_id = digest({'text': segment.translation, 'speaker': segment.speaker_reference,
                               'emotion': segment.emotion_reference, 'fingerprint': fingerprint,
                               **({'revision': overrides['synthesis_revisions'][segment.id]}
                                  if segment.id in overrides.get('synthesis_revisions', {}) else {})})
            segment.synthesized_audio = f'{prefix}/synthesized/{segment.id}-{cache_id}.wav'
            segment.aligned_audio = f'{prefix}/aligned/{segment.id}-{cache_id}.wav'
            if not storage.exists(segment.synthesized_audio):
                try:
                    self.synthesizer.synthesize(segment, segment.synthesized_audio)
                except NeedsReview as exc:
                    segment.flags.append(str(exc))
                    preserve_original(segment)
                    save_segments()
                    continue
            try:
                media.align(storage.path(segment.synthesized_audio), storage.path(segment.aligned_audio),
                            segment.end-segment.start, config.max_speedup)
            except NeedsReview as exc:
                segment.flags.append(str(exc))
                preserve_original(segment)
            else:
                segment.render_status = 'translated'
            save_segments()
            report('synthesize', 55 + round(25 * (index+1)/max(1, len(segments))))

        manifest['completion_policy'] = 'preserve_original_on_segment_review'
        manifest['voice_reference_policy'] = 'current_original_segment_for_both_prompts'
        manifest['dialogue_policy'] = 'legacy_background_plus_dubbed_and_explicit_fallback_only'
        manifest['splice_policy'] = 'legacy_adaptive_100ms_v1'
        manifest['warnings'].extend(
            f'{s.id}（{s.start:.2f}–{s.end:.2f} 秒）未替换，已保留原对白：' + '; '.join(s.flags)
            for s in segments if s.render_status == 'original')
        save_segments()
        stage('mix', 83)
        if segments and not any(s.render_status == 'translated' for s in segments):
            raise NeedsReview('未生成可用的译制对白：所有片段均需检查，未输出仅含原对白的成品。')
        dialogue_key, mixed_key = f'{prefix}/dialogue-translated.wav', f'{prefix}/mixed.wav'
        if segments:
            media.dialogue_timeline(segments, storage, storage.path(dialogue_key), duration,
                                   original=storage.path(stems.dialogue), preserve_intervals=events)
        else:
            # No recognized text: retain the original speech instead of silence.
            media.canonical_audio(storage.path(stems.dialogue), storage.path(dialogue_key))
        media.mix(storage.path(dialogue_key), storage.path(stems.music), storage.path(stems.effects), storage.path(mixed_key), duration)
        stage('assemble', 92)
        video_key = f'{prefix}/translated.mp4'
        outputs = {'video': video_key, 'audio': mixed_key}
        for field, label in [('source_text', 'original'), ('translation', 'translated')]:
            for extension in ('srt', 'vtt'):
                key = f'{prefix}/{label}.{extension}'
                storage.path(key).write_text(media.subtitles(segments, field, extension == 'vtt'), encoding='utf-8')
                outputs[f'{label}_{extension}'] = key
        media.assemble(video, storage.path(mixed_key), storage.path(video_key), metadata,
                       subtitle_path=storage.path(outputs['translated_srt'])
                       if any(s.translation.strip() for s in segments) else None,
                       subtitle_language=job.target_language)
        outputs['manifest'] = manifest_key
        manifest['outputs'] = outputs
        manifest['completed']['assemble'] = True
        save_segments()
        check_cancel()
        report('complete', 100)
        return outputs
