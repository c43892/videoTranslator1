"""Durable CPU-side broker for outbound-only Demucs and IndexTTS2 agents.

The private engine talks to the familiar TTS HTTP interface. Home agents only
see tasks leased to their own credential and transfer bounded media through
this broker; they never receive database, payment, or cloud account secrets.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import PurePosixPath

import soundfile as sf
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import JSON, Float, Integer, String, select
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, Job, Session, init_db
from .storage import LocalStorage

WORKER_PREFIX = '/api/v1/gpu-workers'
LEASE_SECONDS = 45
PRESENCE_SECONDS = 30
MAX_TRANSFER = 4 * 1024**3  # Two hours of 44.1 kHz stereo float WAV is about 2.5 GiB.


class GpuTask(Base):
    __tablename__ = 'gpu_tasks'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    generation: Mapped[str] = mapped_column(String(36), default='', index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    received: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[str] = mapped_column(String(80), default='')
    worker_id: Mapped[str] = mapped_column(String(64), default='')
    lease_hash: Mapped[str] = mapped_column(String(64), default='')
    lease_until: Mapped[float] = mapped_column(Float, default=0)
    created: Mapped[float] = mapped_column(Float)
    deadline: Mapped[float] = mapped_column(Float)


class GpuWorker(Base):
    __tablename__ = 'gpu_workers'
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_seen: Mapped[float] = mapped_column(Float, default=0)
    ready: Mapped[int] = mapped_column(Integer, default=0)
    activity_seq: Mapped[int] = mapped_column(Integer, default=0)


class Texts(BaseModel):
    texts: list[str] = Field(max_length=5000)


class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    speaker_audio: str
    emotion_audio: str
    output: str


class Separation(BaseModel):
    audio: str
    prefix: str
    model: str = Field(default='htdemucs', pattern='^(htdemucs|htdemucs_ft)$')
    segment: float = Field(default=5, ge=1, le=7.8)


class ActiveLease(BaseModel):
    task_id: str
    lease_token: str


class Heartbeat(BaseModel):
    ready: bool = False
    gpu_active: bool = False
    active: list[ActiveLease] = Field(default_factory=list, max_length=1)


class TokenResult(BaseModel):
    task_id: str
    lease_token: str
    counts: list[int] = Field(max_length=5000)


class AgentFailure(BaseModel):
    task_id: str
    lease_token: str
    code: str = Field(pattern='^(gpu_unavailable|gpu_failed|invalid_media|cancelled)$')


def storage():
    return LocalStorage(os.environ.get('STORAGE_ROOT', '/data'))


def media_key(value: str, *, suffix: str = '') -> str:
    path = PurePosixPath(value)
    if (not value.startswith('jobs/') or path.is_absolute() or
            any(part in ('', '.', '..') for part in value.split('/')) or '\\' in value or
            (suffix and path.suffix != suffix)):
        raise HTTPException(422, 'Invalid media key')
    storage().path(value)  # Resolve within the configured storage root.
    return value


def internal(authorization: str = Header(default='')):
    token = os.environ.get('GPU_BROKER_INTERNAL_TOKEN', '')
    if len(token) < 32 or not hmac.compare_digest(authorization, 'Bearer ' + token):
        raise HTTPException(401, 'Unauthorized engine')


def worker(authorization: str = Header(default='')) -> str:
    supplied = authorization.removeprefix('Bearer ') if authorization.startswith('Bearer ') else ''
    allowed = [part.strip() for part in os.environ.get('GPU_WORKER_TOKENS', '').split(',') if part.strip()]
    if not supplied or not any(len(part) >= 32 and hmac.compare_digest(supplied, part) for part in allowed):
        raise HTTPException(401, 'Unauthorized GPU agent')
    return hashlib.sha256(supplied.encode()).hexdigest()[:48]


def lease_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _sweep(db, now: float):
    for task in db.scalars(select(GpuTask).where(GpuTask.status.in_(('queued', 'leased')))):
        job = db.get(Job, task.generation) if task.generation else None
        if task.generation and (job is None or job.status != 'running'):
            task.status, task.error = 'cancelled', 'engine_job_inactive'
            task.worker_id, task.lease_hash, task.lease_until = '', '', 0
        elif now >= task.deadline:
            task.status, task.error = 'failed', 'gpu_task_timeout'
            task.worker_id, task.lease_hash, task.lease_until = '', '', 0
        elif task.status == 'leased' and now >= task.lease_until:
            task.status, task.worker_id, task.lease_hash, task.lease_until = 'queued', '', '', 0


def _verify(db, task_id: str, worker_id: str, lease_token: str) -> GpuTask:
    task = db.get(GpuTask, task_id, with_for_update=True)
    if (not task or task.status != 'leased' or task.worker_id != worker_id or
            time.time() >= min(task.lease_until, task.deadline) or
            not hmac.compare_digest(task.lease_hash, lease_hash(lease_token))):
        raise HTTPException(409, 'GPU task lease is no longer valid')
    return task


def _create(kind: str, payload: dict, seconds: int, generation: str = '') -> GpuTask:
    now = time.time()
    with Session.begin() as db:
        if generation:
            try:
                generation = str(uuid.UUID(generation))
            except ValueError as exc:
                raise HTTPException(422, 'Invalid engine generation') from exc
            job = db.get(Job, generation)
            if not job or job.status != 'running':
                raise HTTPException(409, 'Engine job is not active')
        task = GpuTask(id=str(uuid.uuid4()), generation=generation, kind=kind, status='queued', payload=payload,
                       result={}, received=[], created=now, deadline=now + seconds)
        db.add(task)
    return task


def _view(task: GpuTask) -> dict:
    result = {'id': task.id, 'status': 'running' if task.status == 'leased' else task.status}
    if task.status == 'completed':
        result.update(task.result or {})
    elif task.status == 'failed':
        result['error'] = task.error or 'gpu_failed'
    return result


async def _wait(task_id: str, seconds: int, request: Request):
    limit = time.monotonic() + seconds
    while time.monotonic() < limit:
        if await request.is_disconnected():
            with Session.begin() as db:
                task = db.get(GpuTask, task_id, with_for_update=True)
                if task and task.status in ('queued', 'leased'):
                    task.status, task.error = 'cancelled', 'engine_disconnected'
            raise HTTPException(499, 'Engine disconnected')
        with Session.begin() as db:
            _sweep(db, time.time())
            task = db.get(GpuTask, task_id)
            if task.status == 'completed':
                return task.result
            if task.status in ('failed', 'cancelled'):
                raise HTTPException(503, task.error or 'GPU task failed')
        await asyncio.sleep(1)
    raise HTTPException(504, 'GPU task timed out')


@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(title='Private reverse GPU broker', lifespan=lifespan)


@app.get('/health')
def health():
    now = time.time()
    with Session.begin() as db:
        workers = list(db.scalars(select(GpuWorker).where(GpuWorker.last_seen > now - PRESENCE_SECONDS)))
    if not any(w.ready for w in workers):
        raise HTTPException(503, 'No GPU agent online')
    return {'status': 'ready', 'workers_online': sum(bool(w.ready) for w in workers),
            'activity_seq': sum(w.activity_seq for w in workers)}


@app.post('/tokenize', dependencies=[Depends(internal)])
async def tokenize(body: Texts, request: Request, x_engine_generation: str = Header(default='')):
    task = _create('tokenize', {'texts': body.texts, 'inputs': {}, 'outputs': {}}, 600, x_engine_generation)
    return await _wait(task.id, 600, request)


@app.post('/synthesize', dependencies=[Depends(internal)])
async def synthesize(body: Speech, request: Request, x_engine_generation: str = Header(default='')):
    speaker = media_key(body.speaker_audio, suffix='.wav')
    emotion = media_key(body.emotion_audio, suffix='.wav')
    output = media_key(body.output, suffix='.wav')
    if not storage().exists(speaker) or not storage().exists(emotion):
        raise HTTPException(422, 'Speech reference is unavailable')
    task = _create('synthesize', {'text': body.text,
        'inputs': {'speaker': speaker, 'emotion': emotion}, 'outputs': {'audio': output}}, 1800,
        x_engine_generation)
    return await _wait(task.id, 1800, request)


@app.post('/separations', dependencies=[Depends(internal)])
def separate(body: Separation, x_engine_generation: str = Header(default='')):
    audio = media_key(body.audio)
    prefix = media_key(body.prefix)
    if not storage().exists(audio):
        raise HTTPException(422, 'Source audio is unavailable')
    outputs = {name: f'{prefix}/{name}.wav' for name in ('dialogue', 'music', 'effects')}
    task = _create('separate', {'model': body.model, 'segment': body.segment,
        'inputs': {'audio': audio}, 'outputs': outputs}, 3600, x_engine_generation)
    return _view(task)


@app.get('/separations/{task_id}', dependencies=[Depends(internal)])
def separation_status(task_id: str):
    with Session.begin() as db:
        _sweep(db, time.time())
        task = db.get(GpuTask, task_id)
        if not task or task.kind != 'separate':
            raise HTTPException(404)
        return _view(task)


@app.delete('/separations/{task_id}', dependencies=[Depends(internal)])
def cancel_separation(task_id: str):
    with Session.begin() as db:
        task = db.get(GpuTask, task_id, with_for_update=True)
        if not task or task.kind != 'separate':
            raise HTTPException(404)
        if task.status in ('queued', 'leased'):
            task.status, task.error = 'cancelled', 'cancelled'
        return _view(task)


@app.post(WORKER_PREFIX + '/heartbeat')
def heartbeat(body: Heartbeat, worker_id=Depends(worker)):
    now, cancelled = time.time(), []
    with Session.begin() as db:
        _sweep(db, now)
        host = db.get(GpuWorker, worker_id, with_for_update=True)
        if host is None:
            host = GpuWorker(id=worker_id)
            db.add(host)
        host.last_seen, host.ready = now, int(body.ready)
        if body.gpu_active:
            host.activity_seq += 1
        for entry in body.active:
            try:
                task = _verify(db, entry.task_id, worker_id, entry.lease_token)
            except HTTPException:
                cancelled.append(entry.task_id)
            else:
                task.lease_until = min(now + LEASE_SECONDS, task.deadline)
    return {'cancelled': cancelled}


@app.post(WORKER_PREFIX + '/claim')
def claim(worker_id=Depends(worker)):
    now = time.time()
    with Session.begin() as db:
        _sweep(db, now)
        host = db.get(GpuWorker, worker_id, with_for_update=True)
        if not host or not host.ready or now - host.last_seen >= PRESENCE_SECONDS:
            raise HTTPException(409, 'GPU agent must report ready before claiming')
        active = db.scalar(select(GpuTask.id).where(GpuTask.status == 'leased', GpuTask.worker_id == worker_id).limit(1))
        if active:
            return {'task': None}
        task = db.scalar(select(GpuTask).where(GpuTask.status == 'queued')
                         .order_by(GpuTask.created, GpuTask.id).with_for_update(skip_locked=True).limit(1))
        if not task:
            return {'task': None}
        token = secrets.token_urlsafe(32)
        task.status, task.worker_id = 'leased', worker_id
        task.lease_hash, task.lease_until = lease_hash(token), min(now + LEASE_SECONDS, task.deadline)
        return {'task': {'id': task.id, 'kind': task.kind, 'payload': {
            k: v for k, v in task.payload.items() if k not in ('inputs', 'outputs')},
            'inputs': list(task.payload.get('inputs', {})),
            'outputs': list(task.payload.get('outputs', {})),
            'lease_token': token}}


@app.get(WORKER_PREFIX + '/tasks/{task_id}/inputs/{name}')
def task_input(task_id: str, name: str, x_gpu_lease: str = Header(default=''), worker_id=Depends(worker)):
    with Session.begin() as db:
        task = _verify(db, task_id, worker_id, x_gpu_lease)
        key = task.payload.get('inputs', {}).get(name)
        if not key:
            raise HTTPException(404)
    path = storage().path(key)
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.put(WORKER_PREFIX + '/tasks/{task_id}/outputs/{name}')
async def task_output(task_id: str, name: str, request: Request,
                      x_gpu_lease: str = Header(default=''), worker_id=Depends(worker)):
    with Session.begin() as db:
        task = _verify(db, task_id, worker_id, x_gpu_lease)
        key = task.payload.get('outputs', {}).get(name)
        if not key:
            raise HTTPException(404)
    raw_length = request.headers.get('content-length', '')
    if not raw_length.isdigit() or not 0 < int(raw_length) <= MAX_TRANSFER:
        raise HTTPException(413, 'Invalid GPU output size')
    target = storage().path(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.' + task_id + '.partial')
    try:
        size = 0
        with temporary.open('wb') as stream:
            async for chunk in request.stream():
                size += len(chunk)
                if size > int(raw_length):
                    raise HTTPException(413, 'GPU output exceeds declared size')
                stream.write(chunk)
        if size != int(raw_length):
            raise HTTPException(422, 'GPU output is incomplete')
        try:
            info = sf.info(str(temporary))
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError('Empty audio')
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(422, 'Invalid GPU audio output') from exc
        with Session.begin() as db:
            task = _verify(db, task_id, worker_id, x_gpu_lease)
            temporary.replace(target)
            received = set(task.received or []) | {name}
            task.received = sorted(received)
            if received == set(task.payload['outputs']):
                task.status = 'completed'
                task.result = ({'stems': task.payload['outputs']} if task.kind == 'separate'
                               else {'output': task.payload['outputs']['audio']})
        return {'accepted': True, 'complete': task.status == 'completed'}
    finally:
        temporary.unlink(missing_ok=True)


@app.post(WORKER_PREFIX + '/result')
def token_result(body: TokenResult, worker_id=Depends(worker)):
    with Session.begin() as db:
        task = _verify(db, body.task_id, worker_id, body.lease_token)
        if task.kind != 'tokenize' or len(body.counts) != len(task.payload['texts']) or any(x < 0 for x in body.counts):
            raise HTTPException(422, 'Invalid tokenizer result')
        task.result, task.status = {'counts': body.counts}, 'completed'
    return {'accepted': True}


@app.post(WORKER_PREFIX + '/fail')
def fail_task(body: AgentFailure, worker_id=Depends(worker)):
    with Session.begin() as db:
        task = _verify(db, body.task_id, worker_id, body.lease_token)
        task.status, task.error = 'failed', body.code
    return {'accepted': True}
