"""Actual HTTP registration, scheduling and provider-pinned broker protocol."""
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from videotranslator.db import Base, engine, Session, Job, User
from videotranslator.cloud_control import CloudExecution
from videotranslator.gpu_broker import app, GpuTask, _create
from videotranslator.gpu_registry import GpuProvider, provider_status
from videotranslator.gpu_scheduler import provider_priority
from videotranslator.cloud_worker import next_execution, record_provider_heartbeat


TOKENS = ['gpu-provider-' + str(i) + '-' + 'x' * 40 for i in range(6)]


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setenv('GPU_WORKER_TOKENS', ','.join(TOKENS))
    monkeypatch.setenv('GPU_BROKER_INTERNAL_TOKEN', 'i' * 40)
    monkeypatch.setenv('GPU_PROVIDER_ALWAYS_AVAILABLE', '')
    monkeypatch.delenv('GPU_AZURE_T4_ENABLED', raising=False)
    Base.metadata.drop_all(engine)
    with TestClient(app) as client:
        yield client


def request(client, token, path, body=None):
    return client.post('/api/v1/gpu-workers/' + path,
                       headers={'Authorization': 'Bearer ' + token}, json=body)


def register(client, token, identifier, kind):
    body = {'provider_id': identifier, 'provider_type': kind}
    response = request(client, token, 'register', body)
    assert response.status_code == 200, response.text
    assert response.json() == {'id': identifier, 'type': kind, 'capacity': 1}
    assert request(client, token, 'heartbeat', body | {'ready': True}).status_code == 200


def enqueue():
    identifier, generation = str(uuid.uuid4()), str(uuid.uuid4())
    with Session.begin() as db:
        if not db.get(User, 'registry-test'):
            db.add(User(id='registry-test', email='registry@example.invalid', password_hash='disabled'))
            db.flush()
        db.add(Job(id=generation, user_id='registry-test', filename='test.mp4',
                   input_key=f'jobs/{generation}/input.mp4', target_language='zh', status='queued'))
        db.flush()
        db.add(CloudExecution(id=identifier, generation=generation, spec={},
                             created=time.time(), deadline=time.time() + 300, gpu_provider=''))
    return identifier, generation


def assign(identifier, provider):
    with Session.begin() as db:
        assert next_execution(db, provider) == identifier
        execution = db.get(CloudExecution, identifier)
        execution.gpu_provider = provider
        db.get(Job, execution.generation).status = 'running'


def test_multiple_local_cloud_and_t4_instances_schedule_independently(registry):
    # Register T4 first; type, not spelling or registration order, makes it last.
    providers = [('fallback_one', 't4'), ('local_one', 'local'), ('local_two', 'local'),
                 ('cloud_one', 'cloud'), ('fallback_two', 't4')]
    for token, (identifier, kind) in zip(TOKENS, providers):
        register(registry, token, identifier, kind)
    with Session() as db:
        order = provider_priority(db)
        assert order.index('fallback_one') > order.index('local_two')
        assert order.index('fallback_two') > order.index('cloud_one')
    jobs = [enqueue() for _ in range(6)]
    for i, provider in enumerate(('cloud_one', 'local_one', 'local_two', 'fallback_one', 'fallback_two')):
        if i < 3:
            with Session() as db:
                assert next_execution(db, 'fallback_one') is None
        assign(jobs[i][0], provider)
    with Session() as db:
        assert next_execution(db, 'fallback_two') == jobs[4][0]  # Sticky assigned job.
        assert db.get(CloudExecution, jobs[5][0]).gpu_provider == ''
        statuses = provider_status(db)
        assert len(statuses) == 5 and all(p['online'] and p['busy'] for p in statuses)


def test_tasks_are_pinned_and_wrong_provider_cannot_claim_or_publish(registry):
    register(registry, TOKENS[0], 'gpu_a', 'local')
    register(registry, TOKENS[1], 'gpu_b', 'local')
    first, first_gen = enqueue()
    second, second_gen = enqueue()
    assign(first, 'gpu_a')
    assign(second, 'gpu_b')
    tasks = [_create('tokenize', {'texts': ['hello'], 'inputs': {}, 'outputs': {}}, 60, gen)
             for gen in (first_gen, second_gen)]
    claimed = [request(registry, token, 'claim').json()['task'] for token in TOKENS[:2]]
    assert [t['id'] for t in claimed] == [t.id for t in tasks]
    assert request(registry, TOKENS[0], 'claim').json()['task'] is None
    assert request(registry, TOKENS[1], 'result', {'task_id': claimed[0]['id'],
        'lease_token': claimed[0]['lease_token'], 'counts': [1]}).status_code == 409
    for token, task in zip(TOKENS, claimed):
        assert request(registry, token, 'result', {'task_id': task['id'],
            'lease_token': task['lease_token'], 'counts': [1]}).status_code == 200


def test_offline_provider_skipped_without_moving_pinned_audio(registry):
    register(registry, TOKENS[0], 'gpu_a', 'local')
    register(registry, TOKENS[1], 'gpu_b', 'cloud')
    register(registry, TOKENS[2], 'gpu_t', 't4')
    identifier, gen = enqueue()
    with Session.begin() as db:
        db.get(GpuProvider, 'gpu_a').last_seen = time.time() - 100
    with Session() as db:
        assert next_execution(db, 'gpu_t') is None
    assign(identifier, 'gpu_b')
    task = _create('tokenize', {'texts': ['hello'], 'inputs': {}, 'outputs': {}}, 60, gen)
    leased = request(registry, TOKENS[1], 'claim').json()['task']
    with Session.begin() as db:
        db.get(GpuTask, task.id).lease_until = time.time() - 1
    assert request(registry, TOKENS[2], 'claim').json()['task'] is None
    reclaimed = request(registry, TOKENS[1], 'claim').json()['task']
    assert reclaimed['id'] == task.id and reclaimed['lease_token'] != leased['lease_token']


def test_identity_auth_conflict_type_and_single_credential_binding(registry):
    assert request(registry, 'wrong', 'register', {'provider_id': 'gpu_a', 'provider_type': 'local'}).status_code == 401
    register(registry, TOKENS[0], 'gpu_a', 'local')
    for token, body in ((TOKENS[1], {'provider_id': 'gpu_a', 'provider_type': 'local'}),
                        (TOKENS[0], {'provider_id': 'gpu_b', 'provider_type': 'local'}),
                        (TOKENS[0], {'provider_id': 'gpu_a', 'provider_type': 't4'})):
        assert request(registry, token, 'register', body).status_code == 409
    assert request(registry, TOKENS[1], 'register', {'provider_id': '../bad', 'provider_type': 'local'}).status_code == 422
    assert request(registry, TOKENS[1], 'register', {'provider_id': 'gpu_b', 'provider_type': 'unknown'}).status_code == 422


def test_direct_provider_registers_custom_id_and_type(registry, monkeypatch):
    monkeypatch.setenv('GPU_PROVIDER_TYPE', 'cloud')
    with Session() as db:
        record_provider_heartbeat(db, 'direct_cloud_gpu')
    identifier, _ = enqueue()
    with Session() as db:
        assert next_execution(db, 'direct_cloud_gpu') == identifier
        record = db.get(GpuProvider, 'direct_cloud_gpu')
        assert record.provider_type == 'cloud' and record.transport == 'direct'
    # Outbound clients cannot take an internal direct worker's ID.
    assert request(registry, TOKENS[0], 'register', {
        'provider_id': 'direct_cloud_gpu', 'provider_type': 'cloud'}).status_code == 409


def test_broker_health_is_specific_to_assigned_provider(registry):
    register(registry, TOKENS[0], 'gpu_a', 'local')
    assert registry.get('/health?provider_id=gpu_a').status_code == 200
    assert registry.get('/health?provider_id=gpu_b').status_code == 503


def test_dispatcher_starts_separate_supervisors_for_all_registered_types(registry, monkeypatch):
    from videotranslator import cloud_worker
    register(registry, TOKENS[0], 'gpu_a', 'local')
    register(registry, TOKENS[1], 'gpu_b', 'local')
    register(registry, TOKENS[2], 'gpu_c', 'cloud')
    register(registry, TOKENS[3], 'gpu_t', 't4')
    monkeypatch.setenv('GPU_DISPATCH_REGISTERED', 'true')
    started, stopped = [], []
    class Child:
        def __init__(self, command, **kwargs):
            started.append((command[-1], kwargs['env']['GPU_PROVIDER']))
        def poll(self):
            return None
    monkeypatch.setattr(cloud_worker.subprocess, 'Popen', Child)
    monkeypatch.setattr(cloud_worker, 'stop_child', lambda child: stopped.append(child))
    def stop(_):
        raise KeyboardInterrupt()
    monkeypatch.setattr(cloud_worker.time, 'sleep', stop)
    with pytest.raises(KeyboardInterrupt):
        cloud_worker.dispatch_registered()
    assert set(started) == {(p, p) for p in ('gpu_a', 'gpu_b', 'gpu_c', 'gpu_t')}
    assert len(stopped) == 4
