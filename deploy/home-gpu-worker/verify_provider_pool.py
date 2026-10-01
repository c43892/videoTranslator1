"""Exercise real PostgreSQL and App APIs in an isolated temporary database.

Run in the local engine container; never submits GPU work to cloud services.
"""
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import create_engine, text, select
from sqlalchemy.engine import make_url


def main():
    source = make_url(os.environ['DATABASE_URL'])
    name = 'vt_provider_check_' + uuid.uuid4().hex[:16]
    admin = create_engine(source, isolation_level='AUTOCOMMIT')
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    os.environ['DATABASE_URL'] = source.set(database=name).render_as_string(hide_password=False)
    os.environ['STORAGE_ROOT'] = tempfile.mkdtemp(prefix='vt-provider-check-')
    os.environ['ENGINE_CONTROL_TOKEN'] = 'control-' + 'x' * 40
    os.environ['GPU_WORKER_TOKENS'] = ','.join('agent-' + str(i) + '-' + 'x' * 40 for i in range(5))
    os.environ['GPU_PROVIDER_MODE'] = 'hybrid'
    os.environ['GPU_PROVIDER_ALWAYS_AVAILABLE'] = ''
    os.environ['GPU_DISPATCH_REGISTERED'] = 'true'
    os.environ['GPU_AZURE_T4_ENABLED'] = 'true'
    try:
        from fastapi.testclient import TestClient
        from videotranslator.db import Base, engine, Session, User, Job
        from videotranslator.cloud_control import app as control_app, CloudExecution
        from videotranslator.gpu_broker import app as broker_app, _create
        from videotranslator.cloud_worker import next_execution
        Base.metadata.create_all(engine)
        # Verify an additive migration from the previous worker/task schema.
        with engine.begin() as connection:
            connection.execute(text('ALTER TABLE gpu_workers DROP COLUMN provider_id'))
            connection.execute(text('ALTER TABLE gpu_tasks DROP COLUMN provider_id'))
        tokens = os.environ['GPU_WORKER_TOKENS'].split(',')
        headers = {'Authorization': 'Bearer ' + os.environ['ENGINE_CONTROL_TOKEN']}
        with TestClient(control_app, headers=headers) as control, TestClient(broker_app) as broker:
            def competing_registration(i):
                return broker.post('/api/v1/gpu-workers/register',
                    headers={'Authorization': 'Bearer ' + tokens[i]},
                    json={'provider_id': 'race_check', 'provider_type': 'local'}).status_code
            with ThreadPoolExecutor(max_workers=2) as pool:
                assert sorted(pool.map(competing_registration, (0, 1))) == [200, 409]
            from videotranslator.gpu_broker import GpuWorker
            from videotranslator.gpu_registry import GpuProvider
            with Session.begin() as db:
                for worker in db.scalars(select(GpuWorker).where(GpuWorker.provider_id == 'race_check')):
                    db.delete(worker)
                db.delete(db.get(GpuProvider, 'race_check'))
            for i, (identifier, kind) in enumerate((('gpu_a', 'local'), ('gpu_b', 'local'), ('gpu_c', 'cloud'),
                                                   ('gpu_t1', 't4'), ('gpu_t2', 't4'))):
                auth = {'Authorization': 'Bearer ' + tokens[i]}
                body = {'provider_id': identifier, 'provider_type': kind}
                response = broker.post('/api/v1/gpu-workers/register', headers=auth, json=body)
                assert response.status_code == 200, response.text
                assert broker.post('/api/v1/gpu-workers/heartbeat', headers=auth,
                                   json=body | {'ready': True}).status_code == 200
            jobs = []
            with Session.begin() as db:
                db.add(User(id='verification', email='verification@example.invalid', password_hash='disabled'))
                db.flush()
                for _ in range(6):
                    generation, identifier = str(uuid.uuid4()), str(uuid.uuid4())
                    db.add(Job(id=generation, user_id='verification', filename='test.mp4', input_key='test.mp4',
                               target_language='zh', status='queued'))
                    db.flush()
                    db.add(CloudExecution(id=identifier, generation=generation, spec={}, created=time.time(),
                                          deadline=time.time() + 300, gpu_provider=''))
                    jobs.append((identifier, generation))
            with Session() as db:
                assert db.execute(text('SELECT COUNT(*) FROM gpu_runnable_work')).scalar() == 0
                assert next_execution(db, 'gpu_t1') is None
            for i, provider in enumerate(('gpu_a', 'gpu_b', 'gpu_c', 'gpu_t1', 'gpu_t2')):
                with Session.begin() as db:
                    assert next_execution(db, provider) == jobs[i][0]
                    db.get(CloudExecution, jobs[i][0]).gpu_provider = provider
                    db.get(Job, jobs[i][1]).status = 'running'
                task = _create('tokenize', {'texts': ['hello'], 'inputs': {}, 'outputs': {}}, 60, jobs[i][1])
                wrong = {'Authorization': 'Bearer ' + tokens[(i + 1) % 5]}
                assert broker.post('/api/v1/gpu-workers/claim', headers=wrong).json()['task'] is None
                auth = {'Authorization': 'Bearer ' + tokens[i]}
                leased = broker.post('/api/v1/gpu-workers/claim', headers=auth).json()['task']
                assert leased['id'] == task.id
                result = broker.post('/api/v1/gpu-workers/result', headers=auth, json={
                    'task_id': task.id, 'lease_token': leased['lease_token'], 'counts': [1]})
                assert result.status_code == 200
            with Session() as db:
                assert db.execute(text('SELECT COUNT(*) FROM gpu_runnable_work')).scalar() == 1
            result = control.get('/v1/gpu-providers')
            assert result.status_code == 200 and len(result.json()['providers']) == 5
            assert all(p['online'] and p['busy'] for p in result.json()['providers'])
            print('POSTGRES_PROVIDER_POOL_CHECK_PASSED: additive migration, 5 registrations, '
                  'concurrent identity admission, independent assignments, pinned tasks, T4 fallback and scaler')
        engine.dispose()
    finally:
        # Exact, freshly generated database only; production data is untouched.
        assert name.startswith('vt_provider_check_') and len(name) == 34
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


if __name__ == '__main__':
    main()
