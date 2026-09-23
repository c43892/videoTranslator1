from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import pytest

from videotranslator.adapters.docker_engine import DockerEngineBackend, validate_output
from videotranslator.domain.enums import BackendError, FailureClass, JobStatus
from videotranslator.domain.models import Job, JobOutbox, BackendStatus
from .conftest import NOW, give_balance, make_job_via_inspection


def spec_and_outbox(container, user):
    give_balance(container, user.user_id, 1000)
    make_job_via_inspection(container)
    with container.store.transaction() as tx:
        from videotranslator.domain.models import JobSpec
        row = tx.query(JobOutbox)[0]
        return JobSpec.from_json_dict(row.jobspec), row.outbox_id


def test_submit_replay_recovers_without_uploading_or_relaunching(container, user, monkeypatch):
    spec, key = spec_and_outbox(container, user)
    backend = DockerEngineBackend(container.storage)
    monkeypatch.setattr(backend, '_call', lambda action, identifier: {
        'status':'running', 'spec':spec.to_json_dict()})
    monkeypatch.setattr(backend, '_command', lambda *a, **kw: pytest.fail('must not copy or create'))
    ref = backend.submit(spec, key)
    assert ref.backend_job_id == DockerEngineBackend.engine_id(key)
    with pytest.raises(BackendError):
        backend.submit(replace(spec, target_language='different'), key)


def test_fast_completion_between_polls_is_not_stuck(container, user, monkeypatch):
    spec, key = spec_and_outbox(container, user)
    container.dispatcher.dispatch_due_jobs(now=NOW)
    output = spec.output_uri.removeprefix('obj://')
    container.storage.put_bytes(output, b'test')
    monkeypatch.setattr(container.job_backend, 'get_status', lambda _:
        BackendStatus(state='succeeded', output_object_key=output, warnings=['segment retained']))
    container.reconciler.reconcile_once(now=NOW + 1)
    with container.store.transaction() as tx:
        job = tx.get(Job, spec.job_id)
        assert job.status == JobStatus.SUCCEEDED and job.warnings == ['segment retained']


def test_transient_submit_is_claimed_again_with_same_idempotency_key(container, user, monkeypatch):
    spec, key = spec_and_outbox(container, user)
    original = container.job_backend.submit
    attempts = []
    def submit(spec, idempotency_key):
        attempts.append(idempotency_key)
        if len(attempts) == 1:
            raise BackendError('temporary Docker outage', failure_class=FailureClass.RETRYABLE)
        return original(spec, idempotency_key)
    monkeypatch.setattr(container.job_backend, 'submit', submit)
    container.dispatcher.dispatch_due_jobs(now=NOW)
    container.dispatcher.dispatch_due_jobs(now=NOW + 1800_000)
    assert attempts == [key, key]
    assert len(container.job_backend.submitted_specs) == 1


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg required')
def test_validation_rejects_placeholder_wrong_duration_and_missing_audio(tmp_path):
    placeholder = tmp_path/'fake.mp4'
    placeholder.write_bytes(b'fake-result')
    with pytest.raises(Exception):
        validate_output(placeholder, 1000)
    silent = tmp_path/'silent.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=s=32x32:d=1',str(silent)],check=True)
    with pytest.raises(ValueError, match='streams'):
        validate_output(silent,1000)
    audio = tmp_path/'audio.mp3'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=duration=1',str(audio)],check=True)
    validate_output(audio, 1000, audio=True)
    with pytest.raises(ValueError, match='duration'):
        validate_output(audio, 10000, audio=True)


def test_status_mapping_preserves_warnings_and_cancellation(container, monkeypatch):
    backend = DockerEngineBackend(container.storage)
    identifier = backend.engine_id('task')
    monkeypatch.setattr(backend, '_call', lambda *a, **kw: {'status':'completed_with_warnings','warnings':['kept original']})
    monkeypatch.setattr(backend, '_collect', lambda _: 'result.mp4')
    status = backend.get_status(identifier)
    assert status.state == 'succeeded' and status.warnings == ['kept original']
    monkeypatch.setattr(backend, '_call', lambda *a, **kw: {'status':'cancel_requested','stage':'synthesize','progress':60})
    assert backend.cancel(identifier)
    assert backend.get_status(identifier).state == 'running'


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg required')
def test_collection_keeps_embedded_captions_and_publishes_browser_track(tmp_path, monkeypatch):
    from videotranslator.adapters.local_storage import LocalObjectStorage
    from videotranslator.adapters.ffmpeg import ffprobe_json
    storage = LocalObjectStorage(tmp_path/'objects')
    backend = DockerEngineBackend(storage)
    srt, vtt, video = [tmp_path/name for name in ('translated.srt','translated.vtt','source.mp4')]
    srt.write_text('1\n00:00:00,250 --> 00:00:01,750\n你好，世界。\n', encoding='utf-8')
    vtt.write_text('WEBVTT\n\n00:00:00.250 --> 00:00:01.750\n你好，世界。\n', encoding='utf-8')
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=s=32x32:d=2',
                    '-f','lavfi','-i','sine=duration=2','-i',str(srt),'-map','0:v','-map','1:a',
                    '-map','2:s','-c:v','libx264','-c:a','aac','-c:s','mov_text',
                    '-metadata:s:s:0','language=zho',str(video)], check=True)
    def copy(args, **kwargs):
        assert args[0] == 'cp'
        shutil.copyfile(vtt if args[1].endswith('.vtt') else video, args[2])
    monkeypatch.setattr(backend, '_command', copy)
    key = backend._collect({'id':'engine-one', 'spec':{'output_uri':'obj://result.mp4','duration_ms':2000},
        'outputs':{'video':'/data/jobs/engine-one/translated.mp4','translated_vtt':'/data/jobs/engine-one/translated.vtt'}})
    subtitles = [s for s in ffprobe_json(storage.local_path(key))['streams'] if s['codec_type'] == 'subtitle']
    assert len(subtitles) == 1 and subtitles[0]['codec_name'] == 'mov_text'
    assert subtitles[0]['tags']['language'] == 'zho'
    assert storage.local_path(key+'.vtt').read_text(encoding='utf-8') == vtt.read_text(encoding='utf-8')
    extracted = subprocess.check_output(['ffmpeg','-v','error','-i',str(storage.local_path(key)),'-map','0:s','-f','srt','-']).decode()
    assert '你好，世界。' in extracted and '00:00:00,250 --> 00:00:01,750' in extracted
