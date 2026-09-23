"""Revoice cached translated segments using only each segment's own source audio."""
import copy
import json
import sys
import time
import uuid
from sqlalchemy import text
import soundfile as sf
from videotranslator import media
from videotranslator.config import Settings
from videotranslator.db import engine, Session, Job
from videotranslator.domain import Segment, NeedsReview
from videotranslator.providers import IndexSynthesizer
from videotranslator.references import segment_reference
from videotranslator.storage import LocalStorage
from videotranslator.tasks import EXECUTION_LOCK_ID


def repair(job_id):
    job_id=str(uuid.UUID(job_id))
    storage=LocalStorage('/data'); config=Settings()
    prefix=f'jobs/{job_id}/corrections/segment-reference-v1'
    original_key=f'{prefix}/original-manifest.json'
    original=(storage.read_json(original_key) if storage.exists(original_key)
              else storage.read_json(f'jobs/{job_id}/manifest.json'))
    if not original.get('completed',{}).get('assemble'):
        raise ValueError('Completed task required')
    storage.write_json(original_key,original)
    manifest=copy.deepcopy(original)
    synth=IndexSynthesizer(config)
    segments=[Segment(**s) for s in manifest['segments']]
    with Session() as db:
        target_language=db.get(Job,job_id).target_language
    attempts=[];started=time.monotonic()
    with engine.connect() as lock:
        if not lock.execute(text('SELECT pg_try_advisory_lock(:id)'),{'id':EXECUTION_LOCK_ID}).scalar():
            raise RuntimeError('Another task is processing; retry correction after it finishes')
        lock.commit()
        try:
            processing = sorted(segments, key=lambda s:s.render_status != 'original')
            for index,s in enumerate(processing):
                s.speaker_id='';s.flags=[];s.notes=[];s.render_status='pending'
                if not s.source_text.strip() or not s.translation.strip():
                    s.flags=['no_recognized_text'];s.render_status='original'
                else:
                    reference=segment_reference(s,storage,prefix,config.emotion_reference_max_seconds)
                    s.speaker_reference=s.emotion_reference=reference
                    s.synthesized_audio=f'{prefix}/synthesized/{s.id}.wav'
                    s.aligned_audio=f'{prefix}/aligned/{s.id}.wav'
                    try:
                        if not storage.exists(s.synthesized_audio):
                            synth.synthesize(s,s.synthesized_audio)
                        media.align(storage.path(s.synthesized_audio),storage.path(s.aligned_audio),s.end-s.start,config.max_speedup)
                        actual=sf.info(storage.path(s.aligned_audio)).duration
                        if abs(actual-(s.end-s.start))>1/48000:
                            raise ValueError('Aligned duration mismatch')
                        s.render_status='translated'
                    except NeedsReview as exc:
                        s.flags=[str(exc)];s.render_status='original';s.aligned_audio=''
                attempts.append({'id':s.id,'status':s.render_status,'flags':s.flags})
                storage.write_json(f'{prefix}/progress.json',{'completed':index+1,'total':len(segments),
                    'elapsed_seconds':round(time.monotonic()-started),'attempts':attempts})
                print(json.dumps({'completed':index+1,'total':len(segments),'segment':s.id,'status':s.render_status,'flags':s.flags}),flush=True)
            if not any(s.render_status=='translated' for s in segments):
                raise RuntimeError('No usable synthesis generated')
            print('Mixing corrected dialogue',flush=True)
            stems=manifest['stems'];duration=manifest['duration']
            dialogue,mixed,video=[storage.path(f'{prefix}/{name}') for name in ['dialogue.wav','mixed.wav','translated.mp4']]
            media.dialogue_timeline(segments,storage,dialogue,duration,original=storage.path(stems['dialogue']))
            media.mix(dialogue,storage.path(stems['music']),storage.path(stems['effects']),mixed,duration)
            source=storage.path(original['outputs']['video'])
            media.assemble(source,mixed,video,media.probe(source),subtitle_path=storage.path(original['outputs']['translated_srt']),subtitle_language=target_language)
            media.ffmpeg(['-xerror','-i',video,'-f','null','-'])
            manifest['segments']=[s.to_dict() for s in segments]
            manifest.pop('speakers',None)
            manifest['voice_reference_policy']='current_original_segment_for_both_prompts'
            manifest['previous_punctuation_diagnostics']=[w for w in original.get('warnings',[]) if not w.startswith('seg-')]
            manifest['warnings']=[f'{s.id}（{s.start:.2f}–{s.end:.2f} 秒）无法合成，已保留原声：'+'; '.join(s.flags)
                                  for s in segments if s.render_status=='original']
            manifest['outputs'].update(video=f'{prefix}/translated.mp4',audio=f'{prefix}/mixed.wav',manifest=f'{prefix}/manifest.json')
            storage.write_json(f'{prefix}/manifest.json',manifest)
            evidence={'job_id':job_id,'attempts':attempts,'warnings':manifest['warnings'],'outputs':manifest['outputs'],
                      'previous_outputs':original['outputs'],'additional_charge_cents':0}
            storage.write_json(f'{prefix}/evidence.json',evidence)
            return evidence
        finally:
            lock.execute(text('SELECT pg_advisory_unlock(:id)'),{'id':EXECUTION_LOCK_ID});lock.commit()


def activate(job_id):
    job_id=str(uuid.UUID(job_id));storage=LocalStorage('/data')
    prefix=f'jobs/{job_id}/corrections/segment-reference-v1'
    manifest=storage.read_json(f'{prefix}/manifest.json');evidence=storage.read_json(f'{prefix}/evidence.json')
    with Session.begin() as db:
        job=db.get(Job,job_id)
        if not job or job.status not in ('completed','completed_with_warnings') or job.outputs not in (evidence['previous_outputs'],manifest['outputs']):
            raise ValueError('Task changed since correction')
        storage.write_json(f'jobs/{job_id}/manifest.json',manifest)
        job.outputs=manifest['outputs'];job.status='completed_with_warnings' if manifest['warnings'] else 'completed'
    return {'activated':True,'job_id':job_id}


if __name__=='__main__':
    result=activate(sys.argv[1]) if '--activate' in sys.argv else repair(sys.argv[1])
    print(json.dumps(result,ensure_ascii=False),flush=True)
