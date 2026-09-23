"""Publish an already-remixed local artifact without changing task fees.

Run inside the Studio container after engine-patches/remix_completed_dialogue.py.
Old output bytes are retained; the evidence sidecar records the previous key.
"""
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from videotranslator.adapters.docker_engine import validate_output
from videotranslator.adapters.local_storage import LocalObjectStorage
from videotranslator.config import settings_from_env
from videotranslator.docstore import SQLiteStore
from videotranslator.domain.models import Job


def publish(job_id):
    settings = settings_from_env()
    if settings.profile != 'local-full' or settings.engine_backend != 'docker':
        raise ValueError('This repair tool is only for the trusted local engine')
    store = SQLiteStore(settings.store_path)
    storage = LocalObjectStorage(settings.local_storage_dir)
    with store.transaction() as tx:
        job = tx.get(Job, job_id)
        if not job or job.status != 'succeeded' or not job.output_object_key:
            raise ValueError('A completed task with an output is required')
    engine_id = str(uuid.UUID(job.backend_job_id))
    base = f'/data/jobs/{engine_id}/corrections/dialogue-v2'
    if job.media_type != 'video':
        raise ValueError('This publication command expects an MP4 task')
    key = str(Path(job.output_object_key).parent / 'result-dialogue-v2.mp4')
    with tempfile.TemporaryDirectory(prefix='vt-remix-') as tmp:
        result, evidence_path = Path(tmp)/'result.mp4', Path(tmp)/'evidence.json'
        for source, destination in [('translated.mp4',result),('evidence.json',evidence_path)]:
            subprocess.run([settings.docker_command,'cp',f'{settings.engine_container}:{base}/{source}',str(destination)],check=True,capture_output=True)
        validate_output(result,job.duration_ms)
        evidence = json.loads(evidence_path.read_text())
        if evidence['job_id'] != engine_id or evidence['new_uncovered_rms'] != 0:
            raise ValueError('Correction evidence mismatch')
        evidence.update(studio_job_id=job_id,previous_output_key=job.output_object_key,new_output_key=key)
        evidence_path.write_text(json.dumps(evidence,indent=2))
        if job.output_object_key == key:
            print(json.dumps({'job_id':job_id,'already_published':True}))
            return
        storage.upload(result,key)
        storage.upload(evidence_path,key+'.evidence.json')
    with store.transaction() as tx:
        current = tx.get(Job,job_id)
        if current.status != 'succeeded' or current.output_object_key != job.output_object_key:
            raise ValueError('Task changed before publication')
        current.output_object_key = key
        tx.put(current,job_id)
    print(json.dumps({'job_id':job_id,'published':True,'additional_charge_cents':0}))


if __name__ == '__main__':
    publish(sys.argv[1])
