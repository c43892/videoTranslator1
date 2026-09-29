import io
import time
import uuid

import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

from videotranslator.db import Base, engine, Session, Job, User
from videotranslator.gpu_broker import app, GpuTask, storage

INTERNAL = 'i' * 40
FIRST = 'a' * 40
SECOND = 'b' * 40


def wav():
    data = io.BytesIO()
    sf.write(data, np.zeros(1600, dtype='float32'), 16000, format='WAV')
    return data.getvalue()


@pytest.fixture
def broker(monkeypatch):
    monkeypatch.setenv('GPU_BROKER_INTERNAL_TOKEN', INTERNAL)
    monkeypatch.setenv('GPU_WORKER_TOKENS', FIRST + ',' + SECOND)
    Base.metadata.drop_all(engine)
    with TestClient(app) as client:
        yield client


def call(broker, method, path, token, **kwargs):
    return broker.request(method, path, headers={'Authorization': 'Bearer ' + token,
                                                  **kwargs.pop('headers', {})}, **kwargs)


def register(broker, token):
    response = call(broker, 'POST', '/api/v1/gpu-workers/heartbeat', token,
                    json={'ready': True, 'gpu_active': False, 'active': []})
    assert response.status_code == 200, response.text


def claim(broker, token):
    response = call(broker, 'POST', '/api/v1/gpu-workers/claim', token)
    assert response.status_code == 200, response.text
    return response.json()['task']


def test_two_hosts_register_and_tokenize_with_distinct_leases(broker):
    assert call(broker, 'POST', '/tokenize', FIRST, json={'texts': []}).status_code == 401
    assert call(broker, 'POST', '/api/v1/gpu-workers/heartbeat', 'wrong',
                json={'ready': True}).status_code == 401
    register(broker, FIRST)
    register(broker, SECOND)
    assert broker.get('/health').json()['workers_online'] == 2
    with Session.begin() as db:
        db.add(GpuTask(id='token-task', kind='tokenize', status='queued',
                       payload={'texts': ['hello'], 'inputs': {}, 'outputs': {}},
                       result={}, received=[], created=time.time(), deadline=time.time() + 60))
    task = claim(broker, FIRST)
    assert task['id'] == 'token-task'
    assert claim(broker, FIRST) is None
    assert claim(broker, SECOND) is None
    wrong = call(broker, 'POST', '/api/v1/gpu-workers/result', SECOND,
                 json={'task_id': task['id'], 'lease_token': task['lease_token'], 'counts': [1]})
    assert wrong.status_code == 409
    correct = call(broker, 'POST', '/api/v1/gpu-workers/result', FIRST,
                   json={'task_id': task['id'], 'lease_token': task['lease_token'], 'counts': [1]})
    assert correct.status_code == 200
    assert call(broker, 'POST', '/api/v1/gpu-workers/result', FIRST,
                json={'task_id': task['id'], 'lease_token': task['lease_token'], 'counts': [1]}).status_code == 409


def test_media_transfer_is_fenced_and_validated(broker):
    store = storage()
    for name in ('speaker', 'emotion'):
        path = store.path(f'jobs/example/{name}.wav')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav())
    register(broker, FIRST)
    register(broker, SECOND)
    # Create directly so the test controls the asynchronous engine request.
    with Session.begin() as db:
        db.add(GpuTask(id='speech-task', kind='synthesize', status='queued',
                       payload={'text': 'hello', 'inputs': {'speaker': 'jobs/example/speaker.wav',
                           'emotion': 'jobs/example/emotion.wav'},
                           'outputs': {'audio': 'jobs/example/output.wav'}},
                       result={}, received=[], created=time.time(), deadline=time.time() + 60))
    task = claim(broker, FIRST)
    base = '/api/v1/gpu-workers/tasks/' + task['id']
    headers = {'X-Gpu-Lease': task['lease_token']}
    assert call(broker, 'GET', base + '/inputs/speaker', FIRST, headers=headers).content == wav()
    assert call(broker, 'GET', base + '/inputs/speaker', SECOND, headers=headers).status_code == 409
    assert call(broker, 'PUT', base + '/outputs/audio', FIRST, headers=headers,
                content=b'not audio').status_code == 422
    assert not store.exists('jobs/example/output.wav')
    good = call(broker, 'PUT', base + '/outputs/audio', FIRST, headers=headers, content=wav())
    assert good.status_code == 200, good.text
    assert good.json()['complete'] is True
    assert store.exists('jobs/example/output.wav')


def test_expired_lease_can_be_reclaimed_without_stale_publication(broker):
    register(broker, FIRST)
    register(broker, SECOND)
    with Session.begin() as db:
        db.add(GpuTask(id='expired-task', kind='tokenize', status='queued',
                       payload={'texts': ['a'], 'inputs': {}, 'outputs': {}},
                       result={}, received=[], created=time.time(), deadline=time.time() + 60))
    first = claim(broker, FIRST)
    with Session.begin() as db:
        db.get(GpuTask, first['id']).lease_until = time.time() - 1
    second = claim(broker, SECOND)
    assert second['id'] == first['id']
    assert second['lease_token'] != first['lease_token']
    assert call(broker, 'POST', '/api/v1/gpu-workers/result', FIRST,
                json={'task_id': first['id'], 'lease_token': first['lease_token'],
                      'counts': [1]}).status_code == 409
    assert call(broker, 'POST', '/api/v1/gpu-workers/result', SECOND,
                json={'task_id': second['id'], 'lease_token': second['lease_token'],
                      'counts': [1]}).status_code == 200


def test_cancelled_video_revokes_its_gpu_lease(broker):
    generation = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    with Session.begin() as db:
        db.add(User(id=user_id, email='gpu-test@example.invalid', password_hash='test'))
        db.add(Job(id=generation, user_id=user_id, filename='test.mp4',
                   input_key='jobs/test/input.mp4', target_language='zh', status='running'))
    register(broker, FIRST)
    with Session.begin() as db:
        db.add(GpuTask(id='video-task', generation=generation, kind='tokenize', status='queued',
                       payload={'texts': ['hello'], 'inputs': {}, 'outputs': {}},
                       result={}, received=[], created=time.time(), deadline=time.time() + 60))
    task = claim(broker, FIRST)
    with Session.begin() as db:
        db.get(Job, generation).status = 'cancel_requested'
    response = call(broker, 'POST', '/api/v1/gpu-workers/heartbeat', FIRST,
                    json={'ready': True, 'active': [{'task_id': task['id'],
                        'lease_token': task['lease_token']}]})
    assert response.json()['cancelled'] == [task['id']]
    assert call(broker, 'POST', '/api/v1/gpu-workers/result', FIRST,
                json={'task_id': task['id'], 'lease_token': task['lease_token'],
                      'counts': [1]}).status_code == 409
