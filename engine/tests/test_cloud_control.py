import base64
import json
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from videotranslator.cloud_control import app, CloudExecution
from videotranslator.config import settings
from videotranslator.db import Base, engine, Session, Job
from videotranslator.storage import LocalStorage


def test_reverse_gpu_engine_worker_declares_local_provider():
    compose = (Path(__file__).parents[2] / 'deploy' / 'azure-jp' / 'compose.cpu.yml').read_text()
    worker = compose.split('\n  engine-worker:\n', 1)[1].split('\n  postgres:\n', 1)[0]
    assert 'GPU_PROVIDER: local' in worker


@pytest.mark.parametrize('mode,expected', [('local_only', 'WHERE FALSE'),
                                          ('azure_t4', "WHERE j.status IN"),
                                          ('hybrid', "e.gpu_provider = 'azure_t4'")])
def test_scaler_view_selects_provider_mode(monkeypatch, mode, expected):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from videotranslator import cloud_control
    statements = []
    class Connection:
        def execute(self, statement):
            statements.append(str(statement))
    @contextmanager
    def begin():
        yield Connection()
    monkeypatch.setenv('GPU_PROVIDER_MODE', mode)
    monkeypatch.setenv('GPU_AZURE_T4_ENABLED', 'true')
    monkeypatch.setattr(cloud_control, 'init_db', lambda: None)
    monkeypatch.setattr(cloud_control, 'engine', SimpleNamespace(
        dialect=SimpleNamespace(name='postgresql'), begin=begin))
    cloud_control.initialize()
    assert expected in statements[-1]


def test_local_provider_has_priority_and_azure_keeps_assigned_work(control, monkeypatch):
    from videotranslator import cloud_worker
    identifier, response = submit(control)
    generation = response.json()['generation']
    online = [True]
    monkeypatch.setenv('GPU_AZURE_T4_ENABLED', 'true')
    monkeypatch.setattr(cloud_worker, 'local_gpu_online', lambda _db: online[0])
    with Session() as db:
        assert cloud_worker.next_execution(db, 'local') == identifier
        assert cloud_worker.next_execution(db, 'azure_t4') is None
    online[0] = False
    with Session() as db:
        assert cloud_worker.next_execution(db, 'local') is None
        assert cloud_worker.next_execution(db, 'azure_t4') == identifier
    with Session.begin() as db:
        execution = db.get(CloudExecution, identifier)
        execution.gpu_provider = 'azure_t4'
        db.get(Job, generation).status = 'provisioning'
    online[0] = True
    with Session() as db:
        assert cloud_worker.next_execution(db, 'local') is None
        assert cloud_worker.next_execution(db, 'azure_t4') == identifier


def test_busy_local_spills_to_t4_and_both_busy_leave_fifo_queued(control, monkeypatch):
    from videotranslator import cloud_worker
    monkeypatch.setenv('CLOUD_VALIDATION_MAX_JOBS', '0')
    monkeypatch.setenv('GPU_AZURE_T4_ENABLED', 'true')
    monkeypatch.setattr(cloud_worker, 'local_gpu_online', lambda _db: True)
    first, first_response = submit(control)
    second, second_response = submit(control)
    third, _ = submit(control)
    with Session.begin() as db:
        first_execution = db.get(CloudExecution, first)
        first_execution.gpu_provider = 'local'
        db.get(Job, first_response.json()['generation']).status = 'running'
    with Session() as db:
        assert cloud_worker.provider_available(db, 'local') is False
        assert cloud_worker.next_execution(db, 'azure_t4') == second
    with Session.begin() as db:
        second_execution = db.get(CloudExecution, second)
        second_execution.gpu_provider = 'azure_t4'
        db.get(Job, second_response.json()['generation']).status = 'running'
    with Session() as db:
        assert cloud_worker.provider_available(db, 'local') is False
        assert cloud_worker.provider_available(db, 'azure_t4') is False
        assert db.get(CloudExecution, third).gpu_provider == ''


def test_provider_order_lock_and_poll_contract(monkeypatch):
    from videotranslator.gpu_scheduler import poll_seconds, provider_lock_id, provider_priority
    monkeypatch.delenv('GPU_PROVIDER_PRIORITY', raising=False)
    monkeypatch.delenv('GPU_SCHEDULER_POLL_SECONDS', raising=False)
    assert provider_priority() == ('local', 'azure_t4')
    assert provider_lock_id('local') != provider_lock_id('azure_t4')
    assert poll_seconds() == 10
    template = json.loads((Path(__file__).parents[2] / 'deploy' / 'azure-jp' / 'gpu.template.json').read_text())
    assert template['properties']['template']['scale']['pollingInterval'] == 10


@pytest.mark.parametrize('local_online', [False, True])
def test_explicitly_disabled_t4_leaves_new_work_queued(control, monkeypatch, local_online):
    from videotranslator import cloud_worker
    from videotranslator.gpu_scheduler import provider_enabled
    monkeypatch.setenv('GPU_AZURE_T4_ENABLED', 'false')
    monkeypatch.setenv('CLOUD_VALIDATION_MAX_JOBS', '0')
    monkeypatch.setattr(cloud_worker, 'local_gpu_online', lambda _db: local_online)
    first, first_response = submit(control)
    second, _ = submit(control)
    if local_online:
        with Session.begin() as db:
            db.get(CloudExecution, first).gpu_provider = 'local'
            db.get(Job, first_response.json()['generation']).status = 'running'
    assert provider_enabled('local') is True
    assert provider_enabled('azure_t4') is False
    with Session() as db:
        assert cloud_worker.provider_available(db, 'azure_t4') is False
        assert cloud_worker.next_execution(db, 'azure_t4') is None
        assert db.get(CloudExecution, second).gpu_provider == ''


def test_disabling_t4_preserves_assigned_attempts(control, monkeypatch):
    from videotranslator import cloud_worker
    monkeypatch.setenv('GPU_AZURE_T4_ENABLED', 'false')
    identifier, response = submit(control)
    with Session.begin() as db:
        db.get(CloudExecution, identifier).gpu_provider = 'azure_t4'
        db.get(Job, response.json()['generation']).status = 'running'
    with Session() as db:
        assert cloud_worker.next_execution(db, 'azure_t4') == identifier


@pytest.mark.parametrize('mode', ['azure_t4', 'hybrid'])
def test_disabled_t4_scaler_only_sees_previously_assigned_work(monkeypatch, mode):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from videotranslator import cloud_control
    statements = []
    class Connection:
        def execute(self, statement):
            statements.append(str(statement))
    @contextmanager
    def begin():
        yield Connection()
    monkeypatch.setenv('GPU_AZURE_T4_ENABLED', 'false')
    monkeypatch.setenv('GPU_PROVIDER_MODE', mode)
    monkeypatch.setattr(cloud_control, 'init_db', lambda: None)
    monkeypatch.setattr(cloud_control, 'engine', SimpleNamespace(
        dialect=SimpleNamespace(name='postgresql'), begin=begin))
    cloud_control.initialize()
    assert "AND e.gpu_provider = 'azure_t4'" in statements[-1]
    assert "e.gpu_provider = ''" not in statements[-1]


def test_standalone_local_worker_is_not_t4_and_does_not_require_reverse_agent(monkeypatch):
    from videotranslator import cloud_worker
    monkeypatch.delenv('GPU_PROVIDER', raising=False)
    monkeypatch.setenv('GPU_AZURE_T4_ENABLED', 'false')
    monkeypatch.setenv('GPU_PROVIDER_ALWAYS_AVAILABLE', 'local')
    monkeypatch.setattr(cloud_worker, 'local_gpu_online', lambda _db: pytest.fail('direct GPU needs no reverse agent'))
    assert cloud_worker.provider_name() == 'local'
    assert cloud_worker.provider_available(None, 'azure_t4') is False
    assert cloud_worker.provider_online(None, 'local') is True
    compose = (Path(__file__).parents[2] / 'compose.local-gpu.yml').read_text()
    worker = compose.split('\n  engine-worker:\n', 1)[1].split('\n  tts:\n', 1)[0]
    assert 'GPU_PROVIDER: local' in worker
    assert 'GPU_PROVIDER_ALWAYS_AVAILABLE: local' in worker


@pytest.mark.parametrize('available', ['local', 'local_second', 'other_gpu', None])
def test_t4_enabled_by_default_waits_for_every_other_provider(control, monkeypatch, available):
    from videotranslator import cloud_worker
    from videotranslator.gpu_scheduler import provider_enabled, provider_priority
    monkeypatch.delenv('GPU_AZURE_T4_ENABLED', raising=False)
    monkeypatch.setenv('GPU_PROVIDER_PRIORITY', 'local,azure_t4,local_second,other_gpu')
    order = provider_priority()
    assert order == ('local', 'local_second', 'other_gpu', 'azure_t4')
    assert provider_enabled('azure_t4') is True
    identifier, _ = submit(control)
    checked = []
    def provider_available(db, provider):
        checked.append(provider)
        return provider == 'azure_t4' or provider == available
    monkeypatch.setattr(cloud_worker, 'provider_available', provider_available)
    with Session() as db:
        assert cloud_worker.next_execution(db, 'azure_t4') == (identifier if available is None else None)
    if available is None:
        assert checked == ['azure_t4', 'local', 'local_second', 'other_gpu']


@pytest.mark.parametrize('mode', [None, 'hybrid', 'azure_t4'])
def test_scaler_checks_additional_providers_before_t4(monkeypatch, mode):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from videotranslator import cloud_control
    statements = []
    class Connection:
        def execute(self, statement):
            statements.append(str(statement))
    @contextmanager
    def begin():
        yield Connection()
    monkeypatch.delenv('GPU_AZURE_T4_ENABLED', raising=False)
    monkeypatch.delenv('GPU_PROVIDER_MODE', raising=False)
    if mode:
        monkeypatch.setenv('GPU_PROVIDER_MODE', mode)
    monkeypatch.delenv('REVERSE_GPU_ENABLED', raising=False)
    monkeypatch.setenv('GPU_PROVIDER_PRIORITY', 'local,azure_t4,local_second,other_gpu')
    monkeypatch.setattr(cloud_control, 'init_db', lambda: None)
    monkeypatch.setattr(cloud_control, 'engine', SimpleNamespace(
        dialect=SimpleNamespace(name='postgresql'), begin=begin))
    cloud_control.initialize()
    query = statements[-1]
    assert "pw.provider = 'local_second'" in query and "pw.provider = 'other_gpu'" in query
    assert "e.gpu_provider = 'azure_t4'" in query

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


def test_health_lease_renews_past_old_runtime_limit_and_cannot_resurrect(control, monkeypatch):
    from videotranslator import cloud_worker
    from videotranslator.domain import Cancelled
    identifier, _ = submit(control, processing_profile='health-v1', max_runtime_seconds=1)
    future = time.time() + 7200
    with Session.begin() as db:
        execution = db.get(CloudExecution, identifier)
        execution.started = future - 7100
        execution.deadline = future + 60
        db.get(Job, execution.generation).status = 'running'
    monkeypatch.setattr(cloud_worker.time, 'time', lambda: future)
    cloud_worker.update(identifier, renew_lease=True)
    with Session() as db:
        assert db.get(CloudExecution, identifier).deadline == future + 180
    assert control.get('/v1/jobs/' + identifier).json()['status'] == 'running'
    with Session.begin() as db:
        db.get(CloudExecution, identifier).deadline = future - 1
    with pytest.raises(Cancelled):
        cloud_worker.update(identifier, renew_lease=True)
    assert control.get('/v1/jobs/' + identifier).json()['error'] == 'Worker heartbeat expired'


@pytest.mark.parametrize('healthy', [True, False])
def test_supervisor_runs_beyond_cap_but_stops_stalled_work(control, monkeypatch, healthy):
    from videotranslator import cloud_worker
    identifier, _ = submit(control, processing_profile='health-v1', max_runtime_seconds=1)
    clock = [time.time()]
    start = clock[0]
    monkeypatch.setattr(cloud_worker.time, 'time', lambda: clock[0])
    monkeypatch.setattr(cloud_worker.time, 'monotonic', lambda: clock[0])
    class Lock:
        def execute(self, *_): pass
        def commit(self): pass
    class Child:
        pid = 42
        def poll(self): return None
    class Response:
        status_code = 200
        def json(self):
            return {'status': 'ready', 'activity_seq': int(clock[0]) if healthy else 0}
    monkeypatch.setattr(cloud_worker.httpx, 'get', lambda *a, **k: Response())
    monkeypatch.setattr(cloud_worker.subprocess, 'Popen', lambda *a, **k: Child())
    monkeypatch.setattr(cloud_worker.ProcessActivity, 'sample', lambda *_: False)
    stopped = []
    monkeypatch.setattr(cloud_worker, 'stop_child', lambda child: stopped.append(child))
    def advance(_):
        clock[0] += 60
        if clock[0] - start >= 7200:
            with Session.begin() as db:
                execution = db.get(CloudExecution, identifier)
                db.get(Job, execution.generation).status = 'completed'
                execution.finished = clock[0]
    monkeypatch.setattr(cloud_worker.time, 'sleep', advance)
    cloud_worker.process(identifier, Lock())
    with Session() as db:
        execution = db.get(CloudExecution, identifier)
        job = db.get(Job, execution.generation)
        if healthy:
            assert job.status == 'completed' and clock[0] - start >= 7200
        else:
            assert job.status == 'failed' and 'No step progress' in job.error
            assert clock[0] - start < 1800
    assert stopped
