import threading
import time
import httpx
from celery import Celery
from sqlalchemy import select, text, case
from .config import settings
from .db import Session, Job, engine, init_db
from .domain import Cancelled, NeedsReview
from .storage import LocalStorage
from .providers import MVSeparator, create_translator, IndexSynthesizer
from .scribe import create_transcriber
from .pipeline import Pipeline
from .punctuation import DeepSeekPunctuator
from .youtube import prepare_input
from .demucs import create_separator

cfg = settings()
# Shared by all workers in this deployment, not keyed by job or user.
EXECUTION_LOCK_ID = 24299694758483269
LOCAL_EXECUTION_LOCK = threading.Lock()  # SQLite development/test fallback.
celery = Celery('videotranslator', broker=cfg.redis_url)
celery.conf.update(task_ignore_result=True, task_acks_late=True, worker_prefetch_multiplier=1,
                   broker_connection_retry_on_startup=True, broker_transport_options={'visibility_timeout': 90000},
                   beat_schedule={'recover-jobs': {'task': 'videotranslator.dispatch', 'schedule': 15.0}})

def next_job(db):
    statuses = ['queued', 'running', 'cancel_requested']
    if not cfg.missing_keys():
        statuses.append('waiting_configuration')
    # Finish/recover the in-flight job before admitting another input.
    return db.scalar(select(Job).where(Job.status.in_(statuses)).order_by(
        case((Job.status.in_(['running', 'cancel_requested']), 0), else_=1),
        Job.created, Job.id).limit(1))


def update(job_id, **fields):
    with Session.begin() as db:
        job = db.get(Job, job_id)
        if job:
            for name, value in fields.items():
                setattr(job, name, value)
            job.updated = time.time()

@celery.task(name='videotranslator.dispatch')
def dispatch():
    init_db()
    with Session() as db:
        job = next_job(db)
        if job and (job.status in ('queued', 'waiting_configuration') or time.time()-job.heartbeat > 300):
            # Duplicate deliveries are harmless: process() rechecks the queue head
            # while holding the deployment-wide execution lock.
            process.delay(job.id)

@celery.task(name='videotranslator.process')
def process(job_id):
    init_db()
    lock = engine.connect()
    acquired = False
    stop = threading.Event()
    pulse = None
    try:
        if engine.dialect.name == 'postgresql':
            acquired = bool(lock.execute(text('SELECT pg_try_advisory_lock(:id)'), {'id': EXECUTION_LOCK_ID}).scalar())
            if not acquired:
                return
            lock.commit()
        else:
            acquired = LOCAL_EXECUTION_LOCK.acquire(blocking=False)
            if not acquired:
                return
        with Session() as db:
            job = next_job(db)
            if not job or job.id != job_id:
                return
        if job.status == 'cancel_requested':
            update(job_id, status='cancelled', error='')
            return
        missing = cfg.missing_keys()
        if missing:
            update(job_id, status='waiting_configuration', error='Missing configuration: ' + ', '.join(missing))
            return
        try:
            ready = httpx.get(cfg.tts_url + '/health', timeout=5)
            if ready.status_code != 200:
                update(job_id, status='waiting_configuration', error='IndexTTS2 is downloading or loading; see service status.')
                return
        except httpx.HTTPError:
            update(job_id, status='waiting_configuration', error='IndexTTS2 service is not reachable.')
            return
        with Session.begin() as db:
            current = db.get(Job, job_id, with_for_update=True)
            if current.status in ('cancelled', 'cancel_requested'):
                current.status = 'cancelled'
                return
            if current.status not in ('queued', 'waiting_configuration', 'running'):
                return
            current.status, current.error, current.heartbeat = 'running', '', time.time()
        def heartbeat():
            while not stop.wait(15):
                update(job_id, heartbeat=time.time())
        pulse = threading.Thread(target=heartbeat, daemon=True)
        pulse.start()
        def check_cancel():
            with Session() as db:
                current = db.get(Job, job_id)
                if not current or current.status in ('cancel_requested', 'cancelled'):
                    raise Cancelled()
        def report(stage, progress):
            check_cancel()
            update(job_id, stage=stage, progress=progress, heartbeat=time.time())
        storage = LocalStorage(cfg.storage_root)
        source = prepare_input(job, cfg, storage, check_cancel, report)
        if source and source.get('title'):
            update(job_id, filename=source['title']+'.mp4')
        pipeline = Pipeline(cfg, storage, create_separator(cfg, storage), create_transcriber(cfg, storage, check_cancel=check_cancel),
                            create_translator(cfg), IndexSynthesizer(cfg), DeepSeekPunctuator(cfg))
        outputs = pipeline.run(job, report, check_cancel)
        with Session.begin() as db:
            current = db.get(Job, job_id, with_for_update=True)
            if current.status == 'cancel_requested':
                current.status = 'cancelled'
            else:
                warnings = storage.read_json(outputs['manifest']).get('warnings', [])
                current.status = 'completed_with_warnings' if warnings else 'completed'
                current.progress, current.outputs, current.error = 100, outputs, ''
            current.updated = time.time()
    except Cancelled:
        update(job_id, status='cancelled', error='')
    except NeedsReview as exc:
        update(job_id, status='needs_review', error=str(exc))
    except Exception as exc:
        # Never persist authorization headers, raw requests, or provider URLs.
        message = f'{type(exc).__name__}: {str(exc)[:1200]}'
        for secret in (cfg.mvsep_api_key, cfg.openai_api_key, cfg.elevenlabs_api_key, cfg.deepseek_api_key):
            if secret:
                message = message.replace(secret, '[redacted]')
        update(job_id, status='failed', error=message)
    finally:
        stop.set()
        if pulse:
            pulse.join(timeout=2)
        if acquired and engine.dialect.name == 'postgresql':
            try:
                lock.execute(text('SELECT pg_advisory_unlock(:id)'), {'id': EXECUTION_LOCK_ID})
                lock.commit()
            except Exception:
                pass
        elif acquired:
            LOCAL_EXECUTION_LOCK.release()
        lock.close()
