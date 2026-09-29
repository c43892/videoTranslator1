"""Private cloud control API. CPU-only; durable jobs are the GPU scaling signal.

The legacy local API and Celery worker are deliberately independent of this entrypoint.
Run one control process on the CPU host; the CPU engine and GPU broker share /data.
"""
import base64
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import String, Float, JSON, ForeignKey, select, text
from sqlalchemy.orm import Mapped, mapped_column

from .config import settings
from .db import Base, Job, User, Session, engine, init_db
from .storage import LocalStorage

ACTIVE = ('queued', 'provisioning', 'running', 'cancel_requested')
DONE = ('completed', 'completed_with_warnings')
HEARTBEAT_LEASE_SECONDS = 180
STARTUP_LEASE_SECONDS = 900


def health_supervised(execution):
    return execution.spec.get('processing_profile') == 'health-v1'


class CloudExecution(Base):
    __tablename__ = 'cloud_executions'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    generation: Mapped[str] = mapped_column(ForeignKey('jobs.id'), unique=True)
    spec: Mapped[dict] = mapped_column(JSON)
    created: Mapped[float] = mapped_column(Float)
    deadline: Mapped[float] = mapped_column(Float)
    started: Mapped[float] = mapped_column(Float, default=0)
    finished: Mapped[float] = mapped_column(Float, default=0)
    gpu_provider: Mapped[str] = mapped_column(String(32), default='')


class Spec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: int = Field(ge=2, le=2)
    job_id: str = Field(min_length=1, max_length=100)
    attempt_number: int = Field(ge=1, le=100)
    input_uri: str = Field(min_length=7, max_length=1500)
    output_uri: str = Field(min_length=7, max_length=1500)
    duration_ms: int = Field(gt=0, le=7200000)
    target_language: str = Field(pattern='^(en|zh)$')
    source_language: str | None
    processing_profile: str = Field(max_length=100)
    duration_policy_version: str = Field(max_length=100)
    max_runtime_seconds: int = Field(gt=0, le=86400)


def authorize(request: Request):
    token = os.environ.get('ENGINE_CONTROL_TOKEN', '')
    if len(token) < 32:
        raise HTTPException(503, 'Control authentication is not configured')
    if not hmac.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + token):
        raise HTTPException(401, 'Unauthorized')


def initialize():
    init_db()
    if engine.dialect.name == 'postgresql':
        legacy_reverse = os.getenv('REVERSE_GPU_ENABLED', 'false').lower() == 'true'
        mode = os.getenv('GPU_PROVIDER_MODE') or ('local_only' if legacy_reverse else 'azure_t4')
        if mode not in ('azure_t4', 'local_only', 'hybrid'):
            raise RuntimeError('GPU_PROVIDER_MODE must be azure_t4, local_only, or hybrid')
        active = """j.status IN ('queued','provisioning','running','cancel_requested')
                AND e.deadline > EXTRACT(EPOCH FROM NOW())"""
        if mode == 'azure_t4':
            predicate = active
        elif mode == 'local_only':
            predicate = 'FALSE'
        else:
            predicate = active + """ AND (
                e.gpu_provider = 'azure_t4'
                OR (e.gpu_provider = '' AND j.status = 'queued' AND NOT EXISTS (
                    SELECT 1 FROM gpu_workers w
                    WHERE w.ready = 1
                      AND w.last_seen > EXTRACT(EPOCH FROM NOW()) - 30
                )))"""
        with engine.begin() as db:
            db.execute(text("""CREATE TABLE IF NOT EXISTS gpu_workers (
                id VARCHAR(64) PRIMARY KEY,
                last_seen DOUBLE PRECISION NOT NULL DEFAULT 0,
                ready INTEGER NOT NULL DEFAULT 0,
                activity_seq INTEGER NOT NULL DEFAULT 0
            )"""))
            db.execute(text("""ALTER TABLE cloud_executions
                ADD COLUMN IF NOT EXISTS gpu_provider VARCHAR(32) NOT NULL DEFAULT ''"""))
            db.execute(text(f"""CREATE OR REPLACE VIEW gpu_runnable_work AS
                SELECT e.id FROM cloud_executions e JOIN jobs j ON j.id=e.generation
                WHERE {predicate}"""))


@asynccontextmanager
async def lifespan(app):
    initialize()
    yield


app = FastAPI(title='Private engine control', lifespan=lifespan, dependencies=[Depends(authorize)])


def get_pair(db, identifier):
    execution = db.get(CloudExecution, str(identifier))
    if not execution:
        raise HTTPException(404, 'Unknown job')
    return execution, db.get(Job, execution.generation)


def status(execution, job):
    # For health-v1, deadline is a renewable worker lease, not a total runtime cap.
    expired = job.status in ACTIVE and time.time() >= execution.deadline
    warnings = []
    if job.status in DONE and (job.outputs or {}).get('manifest'):
        storage = LocalStorage(settings().storage_root)
        manifest = storage.path(job.outputs['manifest'])
        if manifest.is_relative_to(storage.path(f'jobs/{execution.generation}')) and manifest.is_file():
            warnings = json.loads(manifest.read_text()).get('warnings', [])
    return dict(id=execution.id, generation=execution.generation, gpu_provider=execution.gpu_provider,
                spec=execution.spec, warnings=warnings,
                status='failed' if expired else job.status, progress=job.progress,
                stage=job.stage, error=(('Worker heartbeat expired' if execution.started else 'GPU startup unavailable')
                    if health_supervised(execution) else 'Execution deadline exceeded') if expired else job.error,
                outputs=job.outputs or {}, runtime_seconds=(
                    max(0, (execution.finished or time.time()) - execution.started) if execution.started else 0))


@app.get('/health')
def health():
    with Session() as db:
        db.execute(text('SELECT 1'))
    return {'ready': True}


@app.get('/v1/jobs/{identifier}')
def get_status(identifier: uuid.UUID):
    with Session() as db:
        return status(*get_pair(db, identifier))


@app.put('/v1/jobs/{identifier}')
async def submit(identifier: uuid.UUID, request: Request):
    try:
        raw = request.headers.get('x-job-spec', '')
        if len(raw) > 12000:
            raise ValueError('Oversized specification')
        spec = Spec.model_validate_json(base64.b64decode(raw, validate=True)).model_dump()
        for uri in (spec['input_uri'], spec['output_uri']):
            if not uri.startswith('obj://') or any(p in ('', '.', '..') for p in uri[6:].split('/')) or '\\' in uri:
                raise ValueError('Invalid object URI')
        length = int(request.headers.get('content-length', '0'))
        if not 0 < length <= settings().max_upload_mb * 1024**2:
            raise ValueError('Invalid content length')
    except (ValueError, TypeError):
        raise HTTPException(400, 'Invalid job specification or content length')
    identifier = str(identifier)
    with Session() as db:
        existing = db.get(CloudExecution, identifier)
        if existing:
            if existing.spec != spec:
                raise HTTPException(409, 'Idempotency conflict')
            return status(existing, db.get(Job, existing.generation))
    # Explicit validation gate: jobs cannot incur GPU charges until configured.
    if os.getenv('CLOUD_ACCEPT_JOBS', 'false').lower() != 'true':
        raise HTTPException(503, 'Job admission is disabled')
    if settings().missing_keys():
        raise HTTPException(503, 'Provider configuration is incomplete')
    generation = str(uuid.uuid4())
    storage = LocalStorage(settings().storage_root)
    target = storage.path(f'jobs/{generation}/input.mp4')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.upload')
    received = 0
    try:
        with temporary.open('wb') as stream:
            async for chunk in request.stream():
                received += len(chunk)
                if received > length:
                    raise HTTPException(413, 'Input exceeds declared length')
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if received != length:
            raise HTTPException(400, 'Incomplete input')
        now = time.time()
        with Session.begin() as db:
            # Serialize all admissions, including cumulative validation budget.
            if engine.dialect.name == 'postgresql':
                db.execute(text('SELECT pg_advisory_xact_lock(7182952)'))
            existing = db.get(CloudExecution, identifier)
            if existing:
                if existing.spec != spec:
                    raise HTTPException(409, 'Idempotency conflict')
                return status(existing, db.get(Job, existing.generation))
            max_jobs = int(os.getenv('CLOUD_VALIDATION_MAX_JOBS', '1'))
            # A positive cap is for attended validation. Zero selects normal
            # operation, where the web control plane enforces durable cost budgets.
            if max_jobs > 0 and len(db.scalars(select(CloudExecution.id)).all()) >= max_jobs:
                raise HTTPException(409, 'Validation job limit reached')
            if not db.get(User, 'cloud-control'):
                db.add(User(id='cloud-control', email='cloud-control@internal.invalid', password_hash='disabled'))
                db.flush()
            temporary.replace(target)
            job = Job(id=generation, user_id='cloud-control', filename='input.mp4',
                      input_key=f'jobs/{generation}/input.mp4', target_language=spec['target_language'],
                      status='queued', stage='queued', outputs={})
            db.add(job)
            db.flush()
            execution = CloudExecution(id=identifier, generation=generation, spec=spec, created=now,
                deadline=now + (STARTUP_LEASE_SECONDS if spec['processing_profile'] == 'health-v1' else
                    min(spec['max_runtime_seconds'], int(os.getenv('CLOUD_JOB_MAX_SECONDS', '1800')))),
                started=0, finished=0, gpu_provider='')
            db.add(execution)
            db.flush()
            return status(execution, job)
    finally:
        temporary.unlink(missing_ok=True)


@app.delete('/v1/jobs/{identifier}')
def cancel(identifier: uuid.UUID):
    with Session.begin() as db:
        execution, job = get_pair(db, identifier)
        db.refresh(job, with_for_update=True)
        if job.status in ACTIVE:
            job.status = 'cancel_requested' if execution.started else 'cancelled'
            job.updated = time.time()
        return status(execution, job)


@app.get('/v1/jobs/{identifier}/artifacts/{kind}')
def artifact(identifier: uuid.UUID, kind: str):
    if kind not in ('audio', 'video', 'translated_vtt'):
        raise HTTPException(404, 'Unknown artifact')
    with Session() as db:
        execution, job = get_pair(db, identifier)
        if job.status not in DONE:
            raise HTTPException(409, 'Result is not complete')
        key = (job.outputs or {}).get(kind)
        if not key:
            raise HTTPException(404, 'Artifact is absent')
        storage = LocalStorage(settings().storage_root)
        path = storage.path(key)
        if not path.is_relative_to(storage.path(f'jobs/{execution.generation}')):
            raise HTTPException(500, 'Invalid artifact reference')
        if not path.is_file():
            raise HTTPException(404, 'Artifact is absent')
        return FileResponse(path)
