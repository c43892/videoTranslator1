"""Re-segment a completed task, keeping earlier artifacts and billing untouched."""
import copy
import json
import sys
import uuid
from sqlalchemy import text
from videotranslator.config import settings
from videotranslator.db import Session, Job, engine
from videotranslator.storage import LocalStorage
from videotranslator.pipeline import Pipeline
from videotranslator.providers import DeepSeekTranslator, IndexSynthesizer
from videotranslator.scribe import create_transcriber
from videotranslator.tasks import EXECUTION_LOCK_ID
from videotranslator import media
import soundfile as sf
import numpy as np


def repair(job_id, activate=False):
    job_id=str(uuid.UUID(job_id));cfg=settings();store=LocalStorage(cfg.storage_root)
    prefix=f'jobs/{job_id}/corrections/turn-boundaries-v1'
    root_manifest=f'jobs/{job_id}/manifest.json'
    correction_manifest=f'{prefix}/manifest.json'
    with Session() as db:
        job=db.get(Job,job_id)
        if not job or job.status not in ('completed','completed_with_warnings'):
            raise ValueError('Completed task required')
        db.expunge(job)
    if activate:
        evidence=store.read_json(prefix+'/evidence.json')
        manifest=store.read_json(correction_manifest)
        with Session.begin() as db:
            current=db.get(Job,job_id)
            if current.outputs not in (evidence['previous_outputs'],manifest['outputs']):
                raise ValueError('Task changed during correction')
            store.write_json(root_manifest,manifest)
            current.outputs=manifest['outputs']
            current.status='completed_with_warnings' if manifest['warnings'] else 'completed'
        return {'activated':True}

    class CorrectionStorage(LocalStorage):
        def path(self,key):
            if key==root_manifest: key=correction_manifest
            elif key.endswith('/transcript.json'): key=prefix+'/transcript.json'
            return super().path(key)
    class CachedSeparation:
        def separate(self,*args): raise RuntimeError('Expected existing separation cache')

    original=store.read_json(root_manifest)
    if not store.exists(correction_manifest):
        store.write_json(prefix+'/original-manifest.json',original)
        store.write_json(correction_manifest,copy.deepcopy(original))
    storage=CorrectionStorage(cfg.storage_root)
    def report(stage,progress):
        print(json.dumps({'stage':stage,'progress':progress}),flush=True)
    with engine.connect() as lock:
        if not lock.execute(text('SELECT pg_try_advisory_lock(:id)'),{'id':EXECUTION_LOCK_ID}).scalar():
            raise RuntimeError('Another task is processing')
        lock.commit()
        try:
            output=Pipeline(cfg,storage,CachedSeparation(),create_transcriber(cfg,storage),
                DeepSeekTranslator(cfg),IndexSynthesizer(cfg)).run(job,report,lambda:None)
            manifest=store.read_json(correction_manifest)
            manifest['transcript']=prefix+'/transcript.json'
            output['manifest']=correction_manifest
            manifest['outputs']=output
            boundaries=manifest['turn_boundaries']
            segments=manifest['segments']
            assert all(not any(s['start'] < t < s['end'] for t in boundaries) for s in segments)
            assert all(not s['speaker_id'] for s in segments)
            for s in segments:
                assert s['speaker_reference']==s['emotion_reference']
                assert s['speaker_reference']==s['original_audio'] or '/references/'+s['id']+'-' in s['speaker_reference']
                if s['render_status']=='translated':
                    assert sf.info(store.path(s['aligned_audio'])).frames==round((s['end']-s['start'])*48000)
                # Short conditioning can only pad the current source with zeros.
                reference,sr=sf.read(store.path(s['speaker_reference']))
                source,_=sf.read(store.path(s['original_audio']))
                if len(source)<=15*sr:
                    assert np.array_equal(source,reference[:len(source)])
                    assert not np.any(reference[len(source):])
            media.ffmpeg(['-xerror','-i',store.path(output['video']),'-f','null','-'])
            store.write_json(correction_manifest,manifest)
            evidence={'job_id':job_id,'previous_outputs':job.outputs,'outputs':output,
                'warnings':manifest['warnings'],'additional_charge_cents':0,
                'turn_boundaries':boundaries,'attempts':[{'id':s['id'],'start':s['start'],'end':s['end'],
                    'source_text':s['source_text'],'status':s['render_status']} for s in segments]}
            store.write_json(prefix+'/evidence.json',evidence)
            return evidence
        finally:
            lock.execute(text('SELECT pg_advisory_unlock(:id)'),{'id':EXECUTION_LOCK_ID});lock.commit()


if __name__=='__main__':
    print(json.dumps(repair(sys.argv[1],'--activate' in sys.argv),ensure_ascii=False),flush=True)
