"""Publish a validated tempo correction to the existing Studio task without charging."""
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
        raise ValueError('Trusted local engine required')
    store = SQLiteStore(settings.store_path)
    storage = LocalObjectStorage(settings.local_storage_dir)
    with store.transaction() as tx:
        job = tx.get(Job,job_id)
        if not job or job.status != 'succeeded' or job.media_type != 'video':
            raise ValueError('Completed video task required')
    engine_id = str(uuid.UUID(job.backend_job_id))
    base = f'/data/jobs/{engine_id}/corrections/tempo-v1'
    key = str(Path(job.output_object_key).parent/'result-tempo-v1.mp4')
    with tempfile.TemporaryDirectory(prefix='vt-tempo-') as tmp:
        result,evidence_file = Path(tmp)/'result.mp4',Path(tmp)/'evidence.json'
        for name,destination in [('translated.mp4',result),('evidence.json',evidence_file)]:
            subprocess.run([settings.docker_command,'cp',f'{settings.engine_container}:{base}/{name}',str(destination)],check=True,capture_output=True)
        evidence = json.loads(evidence_file.read_text())
        if evidence['job_id'] != engine_id or not evidence['repaired']:
            raise ValueError('Correction identity mismatch')
        validate_output(result,job.duration_ms)
        if job.output_object_key == key:
            print(json.dumps({'already_published':True})); return
        evidence.update(studio_job_id=job_id,previous_output_key=job.output_object_key,new_output_key=key)
        evidence_file.write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
        storage.upload(result,key); storage.upload(evidence_file,key+'.evidence.json')
    with store.transaction() as tx:
        current = tx.get(Job,job_id)
        if current.status != 'succeeded' or current.output_object_key != job.output_object_key:
            raise ValueError('Task changed during correction')
        current.output_object_key = key
        current.warnings = evidence['warnings']
        tx.put(current,job_id)
    print(json.dumps({'job_id':job_id,'published':True,'warnings':evidence['warnings'],'additional_charge_cents':0}))


if __name__ == '__main__':
    publish(sys.argv[1])
