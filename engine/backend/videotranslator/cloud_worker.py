"""Cloud CPU engine worker with a supervised child and renewable health lease.

PostgreSQL retains queued and running work. Demucs and IndexTTS2 are delegated
through the reverse GPU broker; all other pipeline stages run here. A crashed
attempt fails closed, and a user retry gets a new identity.
"""
import os
import signal
import subprocess
import sys
import time

import httpx
from sqlalchemy import select, text

from .cloud_control import CloudExecution, ACTIVE, health_supervised, HEARTBEAT_LEASE_SECONDS
from .runtime_health import ProcessActivity, ActivityWatchdog
from .config import settings
from .db import engine, Session, Job
from .domain import Cancelled
from .storage import LocalStorage

LOCK_ID = 24299694758483269
LOCAL_PROVIDER = 'local'
AZURE_PROVIDER = 'azure_t4'


def provider_name():
    value = os.getenv('GPU_PROVIDER', AZURE_PROVIDER)
    if value not in (LOCAL_PROVIDER, AZURE_PROVIDER):
        raise RuntimeError('GPU_PROVIDER must be local or azure_t4')
    return value


def local_gpu_online(db):
    return bool(db.execute(text("""SELECT EXISTS (
        SELECT 1 FROM gpu_workers
        WHERE ready = 1 AND last_seen > EXTRACT(EPOCH FROM NOW()) - 30
    )""")).scalar())


def next_execution(db, provider):
    active = (Job.status.in_(ACTIVE), CloudExecution.deadline > time.time())
    assigned = db.scalar(select(CloudExecution.id).join(Job, Job.id == CloudExecution.generation)
        .where(*active, CloudExecution.gpu_provider == provider)
        .order_by(CloudExecution.created).limit(1))
    if assigned:
        return assigned
    local_online = local_gpu_online(db)
    if (provider == LOCAL_PROVIDER) != local_online:
        return None
    return db.scalar(select(CloudExecution.id).join(Job, Job.id == CloudExecution.generation)
        .where(*active, Job.status == 'queued', CloudExecution.gpu_provider == '')
        .order_by(CloudExecution.created).limit(1))


def update(identifier, *, renew_lease=False, **fields):
    with Session.begin() as db:
        execution = db.get(CloudExecution, identifier)
        job = db.get(Job, execution.generation, with_for_update=True)
        if job.status not in ACTIVE or time.time() >= execution.deadline:
            raise Cancelled()
        if job.status == 'cancel_requested':
            raise Cancelled()
        if renew_lease and health_supervised(execution):
            execution.deadline = time.time() + HEARTBEAT_LEASE_SECONDS
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated = time.time()


def finish(identifier, status, error=''):
    with Session.begin() as db:
        execution = db.get(CloudExecution, identifier)
        job = db.get(Job, execution.generation, with_for_update=True)
        if job.status in ACTIVE:
            job.status, job.error = status, error
            job.updated = execution.finished = time.time()


def run_pipeline(identifier):
    from .demucs import create_separator
    from .pipeline import Pipeline
    from .providers import DeepSeekTranslator, IndexSynthesizer
    from .punctuation import DeepSeekPunctuator
    from .scribe import create_transcriber
    cfg = settings()
    storage = LocalStorage(cfg.storage_root)
    with Session() as db:
        execution = db.get(CloudExecution, identifier)
        job = db.get(Job, execution.generation)
    def check():
        update(identifier)
    def report(stage, progress):
        update(identifier, stage=stage, progress=progress, heartbeat=time.time())
    pipeline = Pipeline(cfg, storage, create_separator(cfg, storage),
        create_transcriber(cfg, storage, check_cancel=check), DeepSeekTranslator(cfg),
        IndexSynthesizer(cfg), DeepSeekPunctuator(cfg))
    outputs = pipeline.run(job, report, check)
    warnings = storage.read_json(outputs['manifest']).get('warnings', [])
    with Session.begin() as db:
        execution = db.get(CloudExecution, identifier)
        job = db.get(Job, execution.generation, with_for_update=True)
        if job.status != 'running' or time.time() >= execution.deadline:
            raise Cancelled()
        job.outputs, job.progress, job.error = outputs, 100, ''
        job.status = 'completed_with_warnings' if warnings else 'completed'
        job.updated = execution.finished = time.time()


def stop_child(child):
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def process(identifier, lock, provider=None):
    provider = provider or provider_name()
    cfg = settings()
    child = None
    activity = None
    watchdog = ActivityWatchdog(clock=time.monotonic)
    next_probe, service_signature = 0, None
    try:
        with Session.begin() as db:
            execution = db.get(CloudExecution, identifier)
            job = db.get(Job, execution.generation, with_for_update=True)
            if execution.gpu_provider and execution.gpu_provider != provider:
                return
            if job.status == 'cancel_requested':
                job.status = 'cancelled'
                execution.finished = time.time()
                return
            if job.status != 'queued':
                # An old process released/lost its execution lock. Never race a
                # second writer against stale GPU requests in the same namespace.
                if job.status in ACTIVE:
                    job.status, job.error = 'failed', 'Worker interrupted; retry as a new attempt'
                    execution.finished = time.time()
                return
            if cfg.missing_keys() or time.time() >= execution.deadline:
                job.status, job.error = 'failed', 'Configuration unavailable or deadline exceeded'
                execution.finished = time.time()
                return
            execution.gpu_provider = provider
            job.status, job.stage = 'provisioning', 'starting_gpu'
            execution.started = time.time()
            supervised = health_supervised(execution)
            if supervised:
                execution.deadline = time.time() + HEARTBEAT_LEASE_SECONDS
        while True:
            # The lock must stay alive; loss terminates the child before retry.
            lock.execute(text('SELECT 1'))
            lock.commit()
            with Session() as db:
                execution = db.get(CloudExecution, identifier)
                current = db.get(Job, execution.generation)
                state = current.status
            if state == 'cancel_requested':
                stop_child(child)
                finish(identifier, 'cancelled')
                return
            if state not in ACTIVE:
                stop_child(child)
                return
            if time.time() >= execution.deadline:
                stop_child(child)
                finish(identifier, 'failed', 'Worker heartbeat expired' if supervised else 'Execution deadline exceeded')
                return
            ready = False
            if not child or (supervised and time.monotonic() >= next_probe):
                next_probe = time.monotonic() + 10
                try:
                    response = httpx.get(cfg.tts_url.rstrip('/') + '/health', timeout=5)
                    ready = response.status_code == 200
                    payload = response.json() if hasattr(response, 'json') else {}
                    if payload.get('status') == 'error':
                        finish(identifier, 'failed', 'GPU service reported an error')
                        return
                    service_signature = (payload.get('status'), payload.get('activity_seq'))
                except (httpx.HTTPError, ValueError):
                    pass  # No successful probe must manufacture progress.
                if supervised and not watchdog.observe(
                        (current.stage, current.progress, service_signature),
                        active=activity.sample() if activity else False):
                    finish(identifier, 'failed', 'No step progress, CPU/I/O or GPU activity for 15 minutes')
                    return
            if child:
                if child.poll() is not None:
                    finish(identifier, 'failed', 'Engine process exited before publishing a result')
                    return
                update(identifier, renew_lease=supervised, heartbeat=time.time())
            else:
                update(identifier, renew_lease=supervised, heartbeat=time.time())
                if ready:
                    update(identifier, status='running', stage='prepare', heartbeat=time.time())
                    child = subprocess.Popen([sys.executable, '-m', 'videotranslator.cloud_worker', '--execute', identifier],
                                             start_new_session=True, env=os.environ | {
                                                 'CLOUD_HEALTH_EXECUTION': '1' if supervised else '0',
                                                 'CLOUD_ENGINE_GENERATION': execution.generation})
                    if supervised:
                        activity = ProcessActivity(child.pid)
            time.sleep(2)
    except BaseException:
        stop_child(child)
        try:
            finish(identifier, 'failed', 'Worker interrupted')
        except Exception:
            pass  # Durable deadline still excludes abandoned work from the scaler.
        raise
    finally:
        stop_child(child)


def main():
    if engine.dialect.name != 'postgresql':
        raise RuntimeError('Cloud worker requires PostgreSQL locking')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    provider = provider_name()
    while True:
        try:
            if provider == LOCAL_PROVIDER:
                with Session() as db:
                    if not local_gpu_online(db):
                        time.sleep(5)
                        continue
            with engine.connect() as lock:
                acquired = lock.execute(text('SELECT pg_try_advisory_lock(:id)'), {'id': LOCK_ID}).scalar()
                lock.commit()
                if acquired:
                    try:
                        with Session() as db:
                            identifier = next_execution(db, provider)
                        if identifier:
                            process(identifier, lock, provider)
                    finally:
                        lock.execute(text('SELECT pg_advisory_unlock(:id)'), {'id': LOCK_ID})
                        lock.commit()
        except Exception as exc:
            # Deliberately exclude raw connection strings and provider errors.
            print('Cloud worker unavailable:', type(exc).__name__, flush=True)
        time.sleep(5)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--execute':
        run_pipeline(sys.argv[2])
    else:
        main()
