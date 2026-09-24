import importlib.util
import json
from pathlib import Path
import threading
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from videotranslator.config import Settings
from videotranslator.demucs import DemucsSeparator, create_separator
from videotranslator.providers import MVSeparator
from videotranslator.domain import Cancelled, ProviderError
from videotranslator.storage import LocalStorage


def test_local_default_no_mvsep_key_and_distinct_cache():
    local=Settings(_env_file=None)
    remote=Settings(_env_file=None,separation_provider='mvsep')
    assert 'MVSEP_API_KEY' not in local.missing_keys()
    assert 'MVSEP_API_KEY' in remote.missing_keys()
    assert local.pipeline_config()!=remote.pipeline_config()
    assert isinstance(create_separator(local,None),DemucsSeparator)
    assert isinstance(create_separator(remote,None),MVSeparator)


@pytest.mark.parametrize('cancel',[False,True])
def test_adapter_outputs_and_cancellation(tmp_path,monkeypatch,cancel):
    from videotranslator import demucs
    storage=LocalStorage(str(tmp_path));calls=[]
    stems={k:'run/demucs/'+k+'.wav' for k in ('dialogue','music','effects')}
    for key in stems.values():
        p=storage.path(key);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    def handler(request):
        calls.append(request.method)
        if request.method=='POST':return httpx.Response(202,json={'id':'test','status':'running'})
        if request.method=='DELETE':return httpx.Response(200,json={'status':'cancelled'})
        return httpx.Response(200,json={'id':'test','status':'completed','stems':stems})
    real=httpx.Client
    monkeypatch.setattr(demucs.httpx,'Client',lambda **kw:real(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(demucs.time,'sleep',lambda _:None)
    def check():
        if cancel and calls:raise Cancelled()
    adapter=DemucsSeparator(Settings(_env_file=None),storage)
    if cancel:
        with pytest.raises(Cancelled):adapter.separate('in.wav','run',lambda _:None,None,check)
        assert calls==['POST','DELETE']
    else:
        assert adapter.separate('in.wav','run',lambda _:None,None,check).dialogue==stems['dialogue']


def test_gpu_service_holds_lock_until_cancelled_child_exits(tmp_path,monkeypatch):
    path=Path(__file__).parents[1]/'services/tts/demucs_service.py'
    spec=importlib.util.spec_from_file_location('demucs_service_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    entered=threading.Event();gpu=threading.Lock();events=[]
    class Process:
        pid=123;returncode=None
        def __init__(self,*a,**kw):
            assert not gpu.acquire(blocking=False)
            assert 'OPENAI_API_KEY' not in kw['env']
            events.append('child');entered.set()
        def poll(self):return self.returncode
        def wait(self,**kw):self.returncode=-15;events.append('exit')
    monkeypatch.setattr(mod.subprocess,'Popen',Process)
    monkeypatch.setattr(mod.os,'killpg',lambda *a:events.append('kill'))
    storage=LocalStorage(str(tmp_path));storage.path('in.wav').write_bytes(b'audio')
    app=FastAPI();mod.install(app,gpu,storage.path,lambda:events.append('unload'))
    with TestClient(app) as client:
        body={'audio':'in.wav','prefix':'run/demucs'}
        job=client.post('/separations',json=body).json()
        assert entered.wait(3)
        assert events[:2]==['unload','child']
        assert client.post('/separations',json=body).json()['id']==job['id']
        assert client.post('/separations',json={**body,'prefix':'another'}).status_code==409
        response=client.delete('/separations/'+job['id'])
        assert response.json()['status']=='cancelled'
        assert events[-2:]==['kill','exit']
        assert gpu.acquire(blocking=False)
        gpu.release()
