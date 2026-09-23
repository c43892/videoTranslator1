"""Internal command contract executed in the existing GPU engine's API container.

No public endpoint or user session is added. Docker access is server-side only.
Engine jobs use its existing durable queue and deployment-wide GPU lock.
"""
import json
import os
import secrets
import sys
import time
import uuid
from pathlib import Path


def main():
    import httpx
    from sqlalchemy import text
    from videotranslator.config import settings
    from videotranslator.db import Session, User, Job
    from videotranslator.storage import LocalStorage

    request = json.load(sys.stdin)
    cfg = settings()
    storage = LocalStorage(cfg.storage_root)
    action = request['action']
    if action == 'health':
        ready = httpx.get(cfg.tts_url + '/health', timeout=10)
        ready.raise_for_status()
        print(json.dumps({'ready': not cfg.missing_keys() and ready.json().get('status') == 'ready',
                          'missing_keys': cfg.missing_keys(), 'tts': ready.json().get('status')}))
        return
    job_id = str(uuid.UUID(request['id']))
    user_id = str(uuid.uuid5(uuid.NAMESPACE_URL, 'videotranslator-studio-internal-engine'))
    with Session.begin() as db:
        # Serializes duplicate submission/recovery and cancellation for this job.
        if db.bind.dialect.name == 'postgresql':
            db.execute(text('SELECT pg_advisory_xact_lock(:id)'),
                       {'id': uuid.UUID(job_id).int % (2**63 - 1)})
        job = db.get(Job, job_id)
        if job and job.user_id != user_id:
            raise ValueError('Engine job ownership mismatch')
        if action == 'submit' and not job:
            spec = request['spec']
            if spec['target_language'] not in ('en', 'zh'):
                raise ValueError('Unsupported target language')
            source = Path('/tmp') / ('studio-' + job_id + '.mp4')
            if not source.is_file():
                raise ValueError('Engine input is missing')
            if not db.get(User, user_id):
                # A non-interactive principal: no usable password or login session.
                from videotranslator.api import password_hash
                db.add(User(id=user_id, email='studio-engine@internal.invalid',
                            password_hash=password_hash(secrets.token_urlsafe(64))))
                db.flush()
            key = f'jobs/{job_id}/input.mp4'
            destination = storage.path(key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(source, destination)
            source.unlink()
            storage.write_json(f'jobs/{job_id}/studio-spec.json', spec)
            job = Job(id=job_id, user_id=user_id, filename=spec['job_id'] + '.mp4',
                      input_key=key, target_language=spec['target_language'], status='queued')
            db.add(job)
            db.flush()
        if not job:
            print(json.dumps({'status': 'not_found'}))
            return
        if action == 'cancel':
            if job.status == 'running':
                job.status = 'cancel_requested'
            elif job.status in ('queued', 'waiting_configuration'):
                job.status = 'cancelled'
            job.updated = time.time()
        spec_key = f'jobs/{job_id}/studio-spec.json'
        spec = storage.read_json(spec_key) if storage.exists(spec_key) else {}
        if action == 'submit' and spec != request['spec']:
            raise ValueError('Idempotency key reused with a different spec')
        manifest_key = f'jobs/{job_id}/manifest.json'
        manifest = storage.read_json(manifest_key) if storage.exists(manifest_key) else {}
        print(json.dumps({'id': job.id, 'status': job.status, 'stage': job.stage,
            'progress': job.progress, 'error': job.error, 'spec': spec,
            'outputs': {k: str(storage.path(v)) for k, v in (job.outputs or {}).items()},
            'warnings': manifest.get('warnings', [])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
