import hashlib
import hmac
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool
from .config import settings
from .db import Session, User, LoginSession, Job, init_db
from .storage import LocalStorage
from . import media
from .youtube import youtube_url

cfg = settings()
storage = LocalStorage(cfg.storage_root)

@asynccontextmanager
async def lifespan(app):
    init_db()
    yield

app = FastAPI(title='VideoTranslator', lifespan=lifespan)

@app.middleware('http')
async def csrf_guard(request: Request, call_next):
    if request.method == 'POST' and request.url.path == '/api/jobs':
        content_length = request.headers.get('content-length')
        if content_length and content_length.isdigit() and int(content_length) > (cfg.max_upload_mb+1)*1024**2:
            return JSONResponse({'detail': '视频超过上传大小限制'}, status_code=413)
    if request.url.path.startswith('/api/') and request.method not in ('GET', 'HEAD', 'OPTIONS'):
        origin = request.headers.get('origin')
        if request.headers.get('x-requested-with') != 'VideoTranslator' or (origin and urlparse(origin).netloc != request.headers.get('host')):
            return JSONResponse({'detail': 'Invalid request origin'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response

def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()

def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 600000).hex()
    return salt + ':' + derived

def current_user(request: Request):
    token = request.cookies.get('vt_session')
    if not token:
        raise HTTPException(401, '请先登录')
    with Session() as db:
        login = db.get(LoginSession, token_hash(token))
        if not login or login.expires < time.time():
            raise HTTPException(401, '登录已过期')
        user = db.get(User, login.user_id)
        if not user:
            raise HTTPException(401, '账户不存在')
        return user

def owned_job(job_id, user):
    with Session() as db:
        job = db.get(Job, job_id)
        if not job or job.user_id != user.id:
            raise HTTPException(404, '任务不存在')
        return job

def job_json(job):
    return {name: getattr(job, name) for name in ('id', 'filename', 'target_language', 'status', 'stage',
                                                 'progress', 'error', 'created', 'updated')} | {'outputs': list(job.outputs or {})}

class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=10, max_length=128)

def set_session(response, user_id):
    token = secrets.token_urlsafe(32)
    with Session.begin() as db:
        db.add(LoginSession(token_hash=token_hash(token), user_id=user_id, expires=time.time()+7*86400))
    response.set_cookie('vt_session', token, max_age=7*86400, httponly=True, secure=cfg.cookie_secure, samesite='strict')

# Local guard supplements deployment-level rate limits.
_attempts = {}
def throttle(request):
    key = request.client.host if request.client else 'local'
    now = time.monotonic()
    _attempts[key] = [t for t in _attempts.get(key, []) if now-t < 60]
    if len(_attempts[key]) >= 20:
        raise HTTPException(429, '请求过于频繁，请稍后再试')
    _attempts[key].append(now)

@app.get('/api/health')
def health():
    with Session() as db:
        db.execute(select(1))
    return {'status': 'ok', 'registration': cfg.allow_registration}

@app.post('/api/auth/register')
def register(body: Credentials, request: Request, response: Response):
    throttle(request)
    if not cfg.allow_registration:
        raise HTTPException(403, '注册已关闭')
    email = body.email.strip().lower()
    if not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', email):
        raise HTTPException(422, '请输入有效邮箱')
    with Session.begin() as db:
        user = User(email=email, password_hash=password_hash(body.password))
        db.add(user)
        try:
            db.flush()
        except IntegrityError:
            raise HTTPException(409, '此邮箱已注册')
        user_id = user.id
    set_session(response, user_id)
    return {'email': email}

@app.post('/api/auth/login')
def login(body: Credentials, request: Request, response: Response):
    throttle(request)
    with Session() as db:
        user = db.scalar(select(User).where(User.email == body.email.strip().lower()))
        stored = user.password_hash if user else password_hash('invalid-password')
        valid = hmac.compare_digest(stored, password_hash(body.password, stored.split(':')[0]))
        if not user or not valid:
            raise HTTPException(401, '邮箱或密码不正确')
        set_session(response, user.id)
        return {'email': user.email}

@app.get('/api/auth/me')
def me(user=Depends(current_user)):
    return {'email': user.email}

@app.post('/api/auth/logout')
def logout(request: Request, response: Response):
    token = request.cookies.get('vt_session', '')
    with Session.begin() as db:
        login = db.get(LoginSession, token_hash(token))
        if login:
            db.delete(login)
    response.delete_cookie('vt_session')
    return {'ok': True}

@app.get('/api/services')
def services(user=Depends(current_user)):
    state = {'status': 'unreachable', 'detail': 'IndexTTS2 服务尚未启动'}
    try:
        result = httpx.get(cfg.tts_url + '/health', timeout=3)
        state = result.json()
    except (httpx.HTTPError, ValueError):
        pass
    return {'missing_keys': cfg.missing_keys(), 'tts': state,
            'providers': {'separation': cfg.separation_label(), 'transcription': cfg.transcription_model(),
                          'diarization': 'disabled',
                          'translation': cfg.deepseek_model, 'synthesis': 'IndexTTS2 · 本地 GPU'},
            'limits': {'max_upload_mb': cfg.max_upload_mb, 'max_video_seconds': cfg.max_video_seconds},
            'target_languages': [{'code': 'zh', 'name': '简体中文'}, {'code': 'en', 'name': 'English'}]}

@app.get('/api/jobs')
def jobs(user=Depends(current_user)):
    with Session() as db:
        return [job_json(j) for j in db.scalars(select(Job).where(Job.user_id == user.id).order_by(Job.created.desc()).limit(100))]

@app.post('/api/jobs', status_code=201)
async def create_job(file: UploadFile = File(...), target_language: str = Form(...),
                     terminology: str = Form(''), user=Depends(current_user)):
    if target_language not in ('en', 'zh'):
        raise HTTPException(422, 'IndexTTS2 第一版支持中文与英文配音')
    if len(terminology) > 10000:
        raise HTTPException(422, '术语表过长')
    filename = Path((file.filename or 'video').replace('\\', '/')).name[:200]
    suffix = Path(filename).suffix.lower()
    if suffix not in ('.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v', '.mpeg', '.mpg', '.ts'):
        raise HTTPException(422, '不支持此视频格式')
    with Session() as db:
        active = db.scalar(select(func.count()).select_from(Job).where(Job.user_id == user.id, Job.status.in_(['queued', 'running', 'waiting_configuration'])))
        if active >= 5:
            raise HTTPException(429, '请先完成或取消已有任务（最多 5 个待处理任务）')
    job_id = str(uuid.uuid4())
    key = f'jobs/{job_id}/input{suffix}'
    path = storage.path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        size = 0
        with path.open('wb') as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > cfg.max_upload_mb * 1024**2:
                    raise HTTPException(413, '视频超过上传大小限制')
                out.write(chunk)
        if not size:
            raise HTTPException(422, '视频为空')
        metadata = await run_in_threadpool(media.probe, path)
        duration = float(metadata['format']['duration'])
        if not 0 < duration <= cfg.max_video_seconds:
            raise HTTPException(422, '视频超过时长限制')
        types = {s['codec_type'] for s in metadata['streams']}
        if not {'audio', 'video'} <= types:
            raise HTTPException(422, '视频必须包含画面和音轨')
        with Session.begin() as db:
            job = Job(id=job_id, user_id=user.id, filename=filename, input_key=key,
                      target_language=target_language, terminology=terminology,
                      status='waiting_configuration' if cfg.missing_keys() else 'queued',
                      error=('等待填写 .env 中的 API key' if cfg.missing_keys() else ''))
            db.add(job)
            db.flush()
            result = job_json(job)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(422, '无法读取此视频，请检查文件格式或完整性')
    finally:
        await file.close()
    # Durable DB queue is picked up by the scheduler, even if broker publication fails.
    return result

class YouTubeInput(BaseModel):
    url: str = Field(min_length=1, max_length=2048)
    target_language: str = Field(pattern='^(en|zh)$')
    terminology: str = Field(default='', max_length=10000)

@app.post('/api/jobs/youtube', status_code=201)
def create_youtube_job(body: YouTubeInput, request: Request, user=Depends(current_user)):
    throttle(request)
    try:
        canonical = youtube_url(body.url)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    job_id = str(uuid.uuid4())
    source_key = f'jobs/{job_id}/source.json'
    try:
        with Session.begin() as db:
            active = db.scalar(select(func.count()).select_from(Job).where(Job.user_id == user.id,
                Job.status.in_(['queued', 'running', 'waiting_configuration', 'cancel_requested'])))
            if active >= 5:
                raise HTTPException(429, '请先完成或取消已有任务（最多 5 个待处理任务）')
            storage.write_json(source_key, {'kind':'youtube','url':canonical,'downloaded':False})
            job = Job(id=job_id, user_id=user.id, filename='YouTube - '+canonical.rsplit('=',1)[1]+'.mp4',
                input_key=f'jobs/{job_id}/input.mp4', target_language=body.target_language,
                terminology=body.terminology, stage='import',
                status='waiting_configuration' if cfg.missing_keys() else 'queued',
                error='等待填写 .env 中的 API key' if cfg.missing_keys() else '')
            db.add(job)
            db.flush()
            result = job_json(job)
        return result
    except Exception:
        storage.path(source_key).unlink(missing_ok=True)
        raise

@app.get('/api/jobs/{job_id}')
def job_details(job_id: str, user=Depends(current_user)):
    job = owned_job(job_id, user)
    key = f'jobs/{job.id}/manifest.json'
    manifest = storage.read_json(key) if storage.exists(key) else {}
    source_key = f'jobs/{job.id}/source.json'
    source = storage.read_json(source_key) if storage.exists(source_key) else {'kind':'upload'}
    return job_json(job) | {'segments': manifest.get('segments', []), 'warnings': manifest.get('warnings', []),
                            'speakers': manifest.get('speakers', {}), 'duration': manifest.get('duration'),
                            'source': source, 'original_ready': storage.exists(job.input_key)}

@app.post('/api/jobs/{job_id}/retry')
def retry(job_id: str, user=Depends(current_user)):
    owned_job(job_id, user)
    with Session.begin() as db:
        job = db.get(Job, job_id, with_for_update=True)
        if job.status not in ('failed', 'needs_review', 'cancelled', 'waiting_configuration', 'completed_with_warnings'):
            raise HTTPException(409, '任务当前不可重试')
        job.status, job.error, job.updated = 'queued', '', time.time()
    return {'ok': True}

@app.post('/api/jobs/{job_id}/cancel')
def cancel(job_id: str, user=Depends(current_user)):
    owned_job(job_id, user)
    with Session.begin() as db:
        job = db.get(Job, job_id, with_for_update=True)
        if job.status == 'running':
            job.status = 'cancel_requested'
        elif job.status in ('queued', 'waiting_configuration', 'failed', 'needs_review'):
            job.status = 'cancelled'
        else:
            raise HTTPException(409, '任务当前不可取消')
        job.error = ''
        job.updated = time.time()
    return {'ok': True}

class Revision(BaseModel):
    translations: dict[str, str] = Field(default_factory=dict)
    speaker_references: dict[str, str] = Field(default_factory=dict)

@app.patch('/api/jobs/{job_id}/segments')
def revise(job_id: str, body: Revision, user=Depends(current_user)):
    owned_job(job_id, user)
    with Session.begin() as db:
        job = db.get(Job, job_id, with_for_update=True)
        if job.status not in ('needs_review', 'failed', 'cancelled', 'completed_with_warnings'):
            raise HTTPException(409, '仅可修改已暂停的任务')
        key = f'jobs/{job.id}/manifest.json'
        if not storage.exists(key):
            raise HTTPException(409, '尚未生成对白片段')
        manifest = storage.read_json(key)
        segments = {s['id']: s for s in manifest.get('segments', [])}
        for sid, translation in body.translations.items():
            if sid not in segments or not translation.strip() or len(translation) > 5000:
                raise HTTPException(422, '无效的片段或译文')
        for speaker, sid in body.speaker_references.items():
            if sid not in segments or segments[sid]['speaker_id'] != speaker:
                raise HTTPException(422, '参考片段必须来自同一说话人')
        override_key = f'jobs/{job.id}/overrides.json'
        overrides = storage.read_json(override_key) if storage.exists(override_key) else {'translations': {}, 'speaker_references': {}}
        overrides.setdefault('translations', {}).update(body.translations)
        overrides.setdefault('speaker_references', {}).update(body.speaker_references)
        overrides['segmentation_version'] = manifest.get('segmentation_version')
        storage.write_json(override_key, overrides)
        for sid, translation in body.translations.items():
            segments[sid]['translation'] = translation
        manifest['segments'] = list(segments.values())
        storage.write_json(key, manifest)
    return {'ok': True}

@app.post('/api/jobs/{job_id}/segments/{segment_id}/retry')
def retry_segment(job_id: str, segment_id: str, user=Depends(current_user)):
    owned_job(job_id, user)
    with Session.begin() as db:
        job = db.get(Job, job_id, with_for_update=True)
        if job.status not in ('needs_review', 'failed', 'cancelled', 'completed_with_warnings'):
            raise HTTPException(409, '请等待当前处理完成')
        key = f'jobs/{job.id}/manifest.json'
        manifest = storage.read_json(key) if storage.exists(key) else {}
        if not any(s['id'] == segment_id for s in manifest.get('segments', [])):
            raise HTTPException(404, '片段不存在')
        key = f'jobs/{job.id}/overrides.json'
        overrides = storage.read_json(key) if storage.exists(key) else {}
        overrides.setdefault('synthesis_revisions', {})[segment_id] = str(uuid.uuid4())
        overrides['segmentation_version'] = manifest.get('segmentation_version')
        storage.write_json(key, overrides)
        job.status, job.error, job.updated = 'queued', '', time.time()
    return {'ok': True}

@app.get('/api/jobs/{job_id}/files/{kind}')
def download(job_id: str, kind: str, user=Depends(current_user)):
    job = owned_job(job_id, user)
    key = job.input_key if kind == 'original' else (job.outputs or {}).get(kind)
    if kind == 'manifest':
        key = f'jobs/{job.id}/manifest.json'
    if not key or not storage.exists(key):
        raise HTTPException(404, '文件尚未生成')
    path = storage.path(key)
    # Inline video and VTT support browser seeking and subtitle tracks.
    media_type = 'text/vtt' if path.suffix == '.vtt' else None
    return FileResponse(path, media_type=media_type, filename=path.name, content_disposition_type='inline')

@app.get('/api/jobs/{job_id}/segments/{segment_id}/audio/{kind}')
def segment_audio(job_id: str, segment_id: str, kind: str, user=Depends(current_user)):
    job = owned_job(job_id, user)
    manifest_key = f'jobs/{job.id}/manifest.json'
    if not storage.exists(manifest_key):
        raise HTTPException(404)
    if kind not in ('original_audio', 'speaker_reference', 'emotion_reference', 'synthesized_audio', 'aligned_audio'):
        raise HTTPException(404)
    segment = next((s for s in storage.read_json(manifest_key)['segments'] if s['id'] == segment_id), None)
    if not segment or not segment.get(kind) or not storage.exists(segment[kind]):
        raise HTTPException(404)
    return FileResponse(storage.path(segment[kind]), media_type='audio/wav')

# Keep the old bookmarked entry point on the current Firebase-backed Studio.
# API routes and internal worker communication remain on this engine origin.
if os.environ.get('STUDIO_PUBLIC_URL'):
    @app.get('/', include_in_schema=False)
    @app.get('/index.html', include_in_schema=False)
    def studio_entry():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(os.environ['STUDIO_PUBLIC_URL'], status_code=307,
                                headers={'Cache-Control': 'no-store'})

if Path('/app/static').exists():
    app.mount('/', StaticFiles(directory='/app/static', html=True), name='web')
