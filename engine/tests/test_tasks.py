import threading
from types import SimpleNamespace

import pytest
from videotranslator import tasks
from videotranslator.config import Settings
from videotranslator.db import Session, Job, User
from videotranslator.domain import Cancelled
from videotranslator.storage import LocalStorage


@pytest.fixture
def queue(client, monkeypatch, tmp_path):
    cfg=Settings(_env_file=None,storage_root=str(tmp_path),mvsep_api_key='test',
                 openai_api_key='test',deepseek_api_key='test')
    monkeypatch.setattr(tasks,'cfg',cfg)
    monkeypatch.setattr(tasks.httpx,'get',lambda *a,**k:SimpleNamespace(status_code=200))
    monkeypatch.setattr(tasks,'prepare_input',lambda *a,**k:None)
    with Session.begin() as db:
        db.add_all([User(id='one',email='one@test.local',password_hash='unused'),
                    User(id='two',email='two@test.local',password_hash='unused')])
        db.flush()
        db.add_all([Job(id='upload',user_id='one',filename='upload.mp4',input_key='upload.mp4',target_language='zh',created=1),
                    Job(id='youtube',user_id='two',filename='youtube.mp4',input_key='youtube.mp4',target_language='zh',created=2)])
    store=LocalStorage(str(tmp_path))
    store.write_json('manifest.json',{'warnings':[]})
    return cfg


def statuses():
    with Session() as db:
        return {j.id:j.status for j in db.query(Job).all()}


def test_fifo_across_users_and_sources_with_concurrent_deliveries(queue,monkeypatch):
    entered,finish=threading.Event(),threading.Event()
    calls=[]
    class Pipeline:
        def __init__(self,*a):pass
        def run(self,job,*a):
            calls.append(job.id)
            if job.id=='upload':
                entered.set()
                assert finish.wait(5)
            return {'manifest':'manifest.json'}
    monkeypatch.setattr(tasks,'Pipeline',Pipeline)
    dispatched=[]
    monkeypatch.setattr(tasks.process,'delay',dispatched.append)
    tasks.dispatch.run()
    assert dispatched==['upload']
    tasks.process.run('youtube')  # Out-of-order broker message cannot jump the queue.
    assert calls==[]
    worker=threading.Thread(target=tasks.process.run,args=('upload',))
    worker.start()
    try:
        assert entered.wait(5)
        tasks.process.run('youtube')
        tasks.process.run('upload')  # Duplicate delivery cannot run concurrently either.
        tasks.dispatch.run()
        assert calls==['upload']
        assert statuses()=={'upload':'running','youtube':'queued'}
        assert dispatched==['upload']
    finally:
        finish.set()
        worker.join(5)
    assert not worker.is_alive()
    tasks.dispatch.run()
    assert dispatched==['upload','youtube']
    tasks.process.run('youtube')
    assert calls==['upload','youtube']
    assert set(statuses().values())=={'completed'}


@pytest.mark.parametrize('failure,status',[(RuntimeError('test failure'),'failed'),(Cancelled(),'cancelled')])
def test_failed_or_cancelled_job_releases_queue(queue,monkeypatch,failure,status):
    class Pipeline:
        def __init__(self,*a):pass
        def run(self,job,*a):
            if job.id=='upload':raise failure
            return {'manifest':'manifest.json'}
    monkeypatch.setattr(tasks,'Pipeline',Pipeline)
    tasks.process.run('upload')
    assert statuses()['upload']==status
    tasks.process.run('youtube')
    assert statuses()['youtube']=='completed'


def test_recovery_takes_priority_and_configuration_waits(queue,monkeypatch):
    tasks.update('youtube',status='running',heartbeat=0)
    dispatched=[]
    monkeypatch.setattr(tasks.process,'delay',dispatched.append)
    tasks.dispatch.run()
    assert dispatched==['youtube']
    tasks.update('youtube',status='cancel_requested')
    tasks.process.run('youtube')
    assert statuses()['youtube']=='cancelled'
    tasks.update('upload',status='waiting_configuration')
    tasks.dispatch.run()
    assert dispatched[-1]=='upload'
    queue.openai_api_key=''
    with Session() as db:
        assert tasks.next_job(db) is None
