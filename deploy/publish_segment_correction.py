"""Publish an independently validated correction without creating/debiting a task."""
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from videotranslator.adapters.docker_engine import validate_output
from videotranslator.adapters.ffmpeg import ffprobe_json
from videotranslator.adapters.local_storage import LocalObjectStorage
from videotranslator.config import settings_from_env
from videotranslator.docstore import SQLiteStore
from videotranslator.domain.models import Job


def publish(job_id, correction='segment-reference-v1'):
    if correction not in ('segment-reference-v1', 'turn-boundaries-v1'):
        raise ValueError('Unknown correction version')
    settings=settings_from_env()
    if settings.profile!='local-full' or settings.engine_backend!='docker':
        raise ValueError('Trusted local engine required')
    store=SQLiteStore(settings.store_path);storage=LocalObjectStorage(settings.local_storage_dir)
    with store.transaction() as tx:
        job=tx.get(Job,job_id)
        if not job or job.status!='succeeded' or job.media_type!='video' or job.assets_deleted_at:
            raise ValueError('Completed video task required')
    engine_id=str(uuid.UUID(job.backend_job_id))
    base=f'/data/jobs/{engine_id}/corrections/{correction}'
    key=str(Path(job.output_object_key).parent/f'result-{correction}.mp4')
    if key==job.output_object_key:
        print(json.dumps({'already_published':True}));return
    with tempfile.TemporaryDirectory(prefix='vt-segment-correction-') as tmp:
        root=Path(tmp)
        def copy(remote,local):
            subprocess.run([settings.docker_command,'cp',f'{settings.engine_container}:{remote}',str(local)],check=True,capture_output=True)
        copy(base+'/evidence.json',root/'evidence.json')
        evidence=json.loads((root/'evidence.json').read_text())
        if evidence['job_id']!=engine_id or not evidence['attempts']:
            raise ValueError('Correction identity mismatch')
        video=evidence['outputs']['video']
        if not video.startswith(f'jobs/{engine_id}/'):
            raise ValueError('Unexpected video path')
        copy('/data/'+video,root/'result.mp4')
        subtitle=evidence['outputs']['translated_vtt']
        if not subtitle.startswith(f'jobs/{engine_id}/'):
            raise ValueError('Unexpected subtitle path')
        copy('/data/'+subtitle,root/'result.vtt')
        validate_output(root/'result.mp4',job.duration_ms)
        streams=ffprobe_json(root/'result.mp4')['streams']
        if not any(s.get('codec_name')=='h264' and s.get('pix_fmt')=='yuv420p' for s in streams):
            raise ValueError('Browser-compatible video required')
        if not any(s.get('codec_name')=='mov_text' for s in streams):
            raise ValueError('Switchable subtitle stream missing')
        evidence.update(studio_job_id=job_id,previous_output_key=job.output_object_key,new_output_key=key)
        (root/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False))
        storage.upload(root/'result.mp4',key)
        storage.upload(root/'result.vtt',key+'.vtt')
        storage.upload(root/'evidence.json',key+'.evidence.json')
    with store.transaction() as tx:
        current=tx.get(Job,job_id)
        if current.status!='succeeded' or current.output_object_key!=job.output_object_key or current.assets_deleted_at:
            raise ValueError('Task changed during correction')
        current.output_object_key=key;current.warnings=evidence['warnings'];tx.put(current,job_id)
    print(json.dumps({'published':True,'job_id':job_id,'additional_charge_cents':0,'warnings':evidence['warnings']}))


if __name__=='__main__':publish(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'segment-reference-v1')
