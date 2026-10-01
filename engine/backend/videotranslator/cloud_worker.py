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
from sqlalchemy import func, select, text

from .cloud_control import CloudExecution, ACTIVE, health_supervised, HEARTBEAT_LEASE_SECONDS
from .runtime_health import ProcessActivity, ActivityWatchdog
from .config import settings
from .db import engine, Session, Job
from .domain import Cancelled
from .gpu_scheduler import (AZURE_PROVIDER, LOCAL_PROVIDER, always_available_providers,
                            poll_seconds, provider_capacities, provider_lock_id,
                            provider_priority, provider_enabled)
from .storage import LocalStorage
from .gpu_registry import GpuProvider, registered


def provider_name():
    value = os.getenv('GPU_PROVIDER', LOCAL_PROVIDER)
    import re
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,31}', value):
        raise RuntimeError('GPU_PROVIDER must be a valid provider ID')
    return value


def local_gpu_online(db):
    from .gpu_broker import GpuWorker
    return bool(db.scalar(select(GpuWorker.id).where(GpuWorker.ready == 1,
        GpuWorker.provider_id == '', GpuWorker.last_seen > time.time() - 30).limit(1)))


def provider_online(db, provider):
    if not provider_enabled(provider, db):
        return False
    record = db.get(GpuProvider, provider) if db is not None else None
    if record:
        return bool(record.ready and record.last_seen > time.time() - 30)
    if provider in always_available_providers():
        return True
    if provider == LOCAL_PROVIDER:
        return local_gpu_online(db)
    if db.bind.dialect.name != 'postgresql':
        return False
    return bool(db.execute(text("""SELECT EXISTS (
        SELECT 1 FROM gpu_provider_workers
        WHERE provider = :provider
          AND ready = 1
          AND last_seen > EXTRACT(EPOCH FROM NOW()) - 30
    )"""), {'provider': provider}).scalar())


def active_provider_count(db, provider):
    active = (Job.status.in_(ACTIVE), CloudExecution.deadline > time.time())
    return int(db.scalar(select(func.count()).select_from(CloudExecution)
        .join(Job, Job.id == CloudExecution.generation)
        .where(*active, CloudExecution.gpu_provider == provider)) or 0)


def provider_available(db, provider):
    return (provider_online(db, provider)
            and active_provider_count(db, provider) < provider_capacities(db)[provider])


def record_provider_heartbeat(db, provider):
    if os.getenv('GPU_DISPATCH_REGISTERED', 'false').lower() == 'true':
        return  # Reverse identities are owned and heartbeated by GPU agents.
    record = db.get(GpuProvider, provider)
    if record and record.transport != 'direct':
        raise RuntimeError('Direct worker ID conflicts with a registered GPU agent')
    kind = os.getenv('GPU_PROVIDER_TYPE', 't4' if provider == AZURE_PROVIDER else 'local')
    if kind not in ('local', 'cloud', 't4'):
        raise RuntimeError('GPU_PROVIDER_TYPE must be local, cloud, or t4')
    if record is None:
        record = GpuProvider(id=provider, provider_type=kind, transport='direct', owner='')
        db.add(record)
    if record.provider_type != kind:
        raise RuntimeError('Provider type cannot change while registered')
    record.last_seen, record.ready = time.time(), 1
    db.commit()


def next_execution(db, provider):
    active = (Job.status.in_(ACTIVE), CloudExecution.deadline > time.time())
    assigned = db.scalar(select(CloudExecution.id).join(Job, Job.id == CloudExecution.generation)
        .where(*active, CloudExecution.gpu_provider == provider)
        .order_by(CloudExecution.created).limit(1))
    if assigned:
        return assigned
    priority = provider_priority(db)
    if not provider_available(db, provider):
        return None
    for preferred in priority[:priority.index(provider)]:
        if provider_available(db, preferred):
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
    from .providers import create_translator, IndexSynthesizer
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
        create_transcriber(cfg, storage, check_cancel=check), create_translator(cfg),
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
            if not execution.gpu_provider and not provider_enabled(provider, db):
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
            if not execution.gpu_provider and db.bind.dialect.name == 'postgresql':
                db.execute(text('SELECT pg_advisory_xact_lock(7182953)'))
                if next_execution(db, provider) != identifier:
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
            if os.getenv('GPU_DISPATCH_REGISTERED', 'false').lower() != 'true':
                with Session() as db:
                    record_provider_heartbeat(db, provider)
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
                    health_url = cfg.tts_url.rstrip('/') + '/health'
                    if os.getenv('GPU_DISPATCH_REGISTERED', 'false').lower() == 'true':
                        health_url += '?provider_id=' + provider
                    response = httpx.get(health_url, timeout=5)
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


def provider_loop(provider):
    if engine.dialect.name != 'postgresql':
        raise RuntimeError('Cloud worker requires PostgreSQL locking')
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    interval = poll_seconds()
    lock_id = provider_lock_id(provider)
    while True:
        try:
            with Session() as db:
                record_provider_heartbeat(db, provider)
            with engine.connect() as lock:
                acquired = lock.execute(text('SELECT pg_try_advisory_lock(:id)'), {'id': lock_id}).scalar()
                lock.commit()
                if acquired:
                    try:
                        with Session() as db:
                            identifier = next_execution(db, provider)
                        if identifier:
                            process(identifier, lock, provider)
                    finally:
                        lock.execute(text('SELECT pg_advisory_unlock(:id)'), {'id': lock_id})
                        lock.commit()
        except Exception as exc:
            # Deliberately exclude raw connection strings and provider errors.
            print('Cloud worker unavailable:', type(exc).__name__, flush=True)
        time.sleep(interval)


def dispatch_registered():
    """One supervised CPU process per registered reverse GPU, not one local slot."""
    children = {}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        while True:
            with Session() as db:
                records = registered(db)
                ids = {p.id for p in records if p.transport == 'reverse' and
                       (p.last_seen > time.time() - 30 or active_provider_count(db, p.id))}
                # Legacy agents remain usable during a rolling upgrade.
                if local_gpu_online(db):
                    ids.add(LOCAL_PROVIDER)
            for identifier in ids:
                child = children.get(identifier)
                if child is None or child.poll() is not None:
                    children[identifier] = subprocess.Popen(
                        [sys.executable, '-m', 'videotranslator.cloud_worker', '--provider-loop', identifier],
                        start_new_session=True, env=os.environ | {'GPU_PROVIDER': identifier})
            for identifier in set(children) - ids:
                stop_child(children.pop(identifier))
            time.sleep(poll_seconds())
    finally:
        for child in children.values():
            stop_child(child)


def main():
    if os.getenv('GPU_DISPATCH_REGISTERED', 'false').lower() == 'true':
        dispatch_registered()
    else:
        provider_loop(provider_name())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--execute':
        run_pipeline(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == '--provider-loop':
        provider_loop(provider_name())
    else:
        main()
