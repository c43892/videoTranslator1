import base64
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from videotranslator.cloud_control import app, CloudExecution
from videotranslator.config import settings
from videotranslator.db import Base, engine, Session, Job
from videotranslator.storage import LocalStorage

TOKEN = 'test-private-control-token-32-characters'


@pytest.fixture
def control(monkeypatch):
    monkeypatch.setenv('ENGINE_CONTROL_TOKEN', TOKEN)
    monkeypatch.setenv('CLOUD_ACCEPT_JOBS', 'true')
    monkeypatch.setenv('CLOUD_VALIDATION_MAX_JOBS', '1')
    monkeypatch.setattr(settings(), 'openai_api_key', 'test-placeholder')
    monkeypatch.setattr(settings(), 'deepseek_api_key', 'test-placeholder')
    Base.metadata.drop_all(engine)
    with TestClient(app, headers={'Authorization': 'Bearer ' + TOKEN}) as client:
        yield client


def submit(control, identifier=None, **overrides):
    identifier = identifier or str(uuid.uuid4())
    spec = dict(schema_version=2, job_id='job_example', attempt_number=1,
        input_uri='obj://inputs/users/u/input.mp4', output_uri='obj://users/u/result.mp4',
        duration_ms=1000, target_language='zh', source_language='en',
        processing_profile='default', duration_policy_version='v1', max_runtime_seconds=600)
    spec.update(overrides)
    response = control.put('/v1/jobs/' + identifier, content=b'video',
        headers={'X-Job-Spec': base64.b64encode(json.dumps(spec).encode()).decode()})
    return identifier, response


def test_auth_health_and_idempotent_admission(control):
    assert control.get('/health', headers={'Authorization': 'Bearer wrong'}).status_code == 401
    assert control.get('/health').json() == {'ready': True}
    identifier, response = submit(control)
    assert response.status_code == 200, response.text
    assert response.json()['status'] == 'queued'
    generation = response.json()['generation']
    assert submit(control, identifier)[1].json()['generation'] == generation
    assert submit(control, identifier, duration_ms=2000)[1].status_code == 409
    assert submit(control)[1].status_code == 409  # Durable validation budget guard.
    with Session() as db:
        assert len(db.scalars(select(CloudExecution)).all()) == 1
    assert LocalStorage(settings().storage_root).path(f'jobs/{generation}/input.mp4').read_bytes() == b'video'


def test_cancel_before_start_never_needs_gpu(control):
    identifier, response = submit(control)
    assert control.delete('/v1/jobs/' + identifier).json()['status'] == 'cancelled'
    assert control.get('/v1/jobs/' + identifier + '/artifacts/video').status_code == 409


def test_zero_validation_cap_allows_normal_budgeted_operation(control, monkeypatch):
    monkeypatch.setenv('CLOUD_VALIDATION_MAX_JOBS', '0')
    assert submit(control)[1].status_code == 200
    assert submit(control)[1].status_code == 200


def test_deadline_and_artifact_path_fencing(control):
    identifier, response = submit(control)
    with Session.begin() as db:
        execution = db.get(CloudExecution, identifier)
        execution.deadline = time.time() - 1
    assert control.get('/v1/jobs/' + identifier).json()['status'] == 'failed'
    with Session.begin() as db:
        execution = db.get(CloudExecution, identifier)
        job = db.get(Job, execution.generation)
        job.status = 'completed'
        job.outputs = {'video': 'jobs/some-other-job/result.mp4'}
    assert control.get('/v1/jobs/' + identifier + '/artifacts/video').status_code == 500
    assert control.get('/v1/jobs/' + identifier + '/artifacts/manifest').status_code == 404


def test_disabled_admission_and_invalid_paths(control, monkeypatch):
    assert submit(control, input_uri='obj://inputs/../secret')[1].status_code == 400
    monkeypatch.setenv('CLOUD_ACCEPT_JOBS', 'false')
    assert submit(control)[1].status_code == 503


def test_restarted_worker_never_reuses_inflight_generation(control, monkeypatch):
    from videotranslator import cloud_worker
    identifier, response = submit(control)
    with Session.begin() as db:
        execution = db.get(CloudExecution, identifier)
        db.get(Job, execution.generation).status = 'running'
        execution.started = time.time()
    monkeypatch.setattr(cloud_worker.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not relaunch old work'))
    cloud_worker.process(identifier, None)
    result = control.get('/v1/jobs/' + identifier).json()
    assert result['status'] == 'failed'
    assert result['generation'] == response.json()['generation']
    assert 'Worker interrupted' in result['error']


def test_worker_cancel_during_model_loading_does_not_start_pipeline(control, monkeypatch):
    from videotranslator import cloud_worker
    identifier, _ = submit(control)
    class Lock:
        def execute(self, *_):
            return None
        def commit(self):
            pass
    class NotReady:
        status_code = 503
    monkeypatch.setattr(cloud_worker.httpx, 'get', lambda *a, **k: NotReady())
    monkeypatch.setattr(cloud_worker.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not start after cancellation'))
    monkeypatch.setattr(cloud_worker.time, 'sleep', lambda *_: control.delete('/v1/jobs/' + identifier))
    cloud_worker.process(identifier, Lock())
    result = control.get('/v1/jobs/' + identifier).json()
    assert result['status'] == 'cancelled' and result['outputs'] == {}
