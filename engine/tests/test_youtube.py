import json
from pathlib import Path
from types import SimpleNamespace
import time
import subprocess

import pytest
from videotranslator.config import Settings
from videotranslator.domain import Cancelled, ProviderError
from videotranslator.storage import LocalStorage
from videotranslator import media, youtube

VIDEO = 'jNQXAC9IVRw'
URL = 'https://www.youtube.com/watch?v=' + VIDEO

@pytest.mark.parametrize('url', [URL+'&list=ignored&t=10', 'https://youtu.be/'+VIDEO+'?si=tracking',
    'https://m.youtube.com/shorts/'+VIDEO, 'https://www.youtube.com/embed/'+VIDEO])
def test_canonical_url(url):
    assert youtube.youtube_url(url) == URL

@pytest.mark.parametrize('url', ['http://localhost/video', 'file:///etc/passwd',
    'https://youtube.com.evil.test/watch?v='+VIDEO, 'https://youtube.com@localhost/watch?v='+VIDEO,
    'https://youtube.com/playlist?list=123', 'https://youtube.com/watch?v=invalid',
    URL+'&v='+VIDEO, 'https://youtube.com:8080/watch?v='+VIDEO])
def test_reject_untrusted_urls(url):
    with pytest.raises(ValueError):
        youtube.youtube_url(url)

@pytest.mark.parametrize('extra', [{'_type':'playlist'}, {'is_live':True}, {'duration':None},
    {'duration':7201}, {'duration':float('nan')}, {'availability':'private'}])
def test_reject_ineligible_metadata(extra):
    with pytest.raises(ProviderError):
        youtube.validate_info({'duration':19, **extra}, 7200)

def make_video(path):
    media.ffmpeg(['-f','lavfi','-i','color=s=160x90:d=1','-f','lavfi','-i','sine=duration=1',
        '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',path])

def test_download_publish_and_retry_reuses_valid_input(tmp_path,monkeypatch):
    config = Settings(_env_file=None)
    source = youtube.YouTubeSource(config)
    calls=[]
    def run(args,work,check,deadline):
        calls.append(args)
        if '--dump-single-json' in args:
            return json.dumps({'id':VIDEO,'title':'A / title','duration':1})
        make_video(work/'source.mp4')
        return ''
    monkeypatch.setattr(source,'_run',run)
    storage = LocalStorage(str(tmp_path))
    job=SimpleNamespace(id='test',input_key='jobs/test/input.mp4')
    storage.write_json('jobs/test/source.json',{'kind':'youtube','url':URL})
    result=youtube.prepare_input(job,config,storage,lambda:None,lambda *a:None,source)
    assert result['downloaded'] and result['title']=='A   title'
    assert media.probe(storage.path(job.input_key))['streams']
    assert not list((tmp_path/'jobs/test').glob('.youtube-*'))
    youtube.prepare_input(job,config,storage,lambda:None,lambda *a:None,source)
    assert len(calls)==2

def test_invalid_download_is_not_published(tmp_path,monkeypatch):
    source=youtube.YouTubeSource(Settings(_env_file=None))
    def run(args,work,*rest):
        if '--dump-single-json' in args:
            return json.dumps({'id':VIDEO,'duration':1})
        (work/'source.mp4').write_bytes(b'broken')
        return ''
    monkeypatch.setattr(source,'_run',run)
    with pytest.raises(Exception):
        source.download(URL,tmp_path/'input.mp4',lambda:None,lambda *a:None)
    assert list(tmp_path.iterdir())==[]

def test_cancel_kills_process_and_does_not_pass_credentials(tmp_path,monkeypatch):
    real_popen=subprocess.Popen
    children=[]
    def popen(args,**kwargs):
        assert 'OPENAI_API_KEY' not in kwargs['env']
        proc=real_popen(['python','-c','import time; time.sleep(60)'],**kwargs)
        children.append(proc)
        return proc
    monkeypatch.setattr(youtube.subprocess,'Popen',popen)
    def cancel():
        raise Cancelled()
    with pytest.raises(Cancelled):
        youtube.YouTubeSource(Settings(_env_file=None))._run([],tmp_path,cancel,time.monotonic()+10)
    assert children[0].poll() is not None

def test_youtube_api_auth_validation_quota_and_ownership(client):
    body={'url':URL,'target_language':'zh'}
    assert client.post('/api/jobs/youtube',json=body).status_code==401
    client.post('/api/auth/register',json={'email':'youtube@example.com','password':'long-password-123'})
    assert client.post('/api/jobs/youtube',json=body,headers={'X-Requested-With':''}).status_code==403
    assert client.post('/api/jobs/youtube',json={**body,'url':'http://localhost'}).status_code==422
    result=client.post('/api/jobs/youtube',json=body)
    assert result.status_code==201, result.text
    job_id=result.json()['id']
    detail=client.get('/api/jobs/'+job_id).json()
    assert detail['stage']=='import' and detail['original_ready'] is False
    assert detail['source']=={'kind':'youtube','url':URL,'downloaded':False}
    for _ in range(4):
        assert client.post('/api/jobs/youtube',json=body).status_code==201
    assert client.post('/api/jobs/youtube',json=body).status_code==429
    client.post('/api/jobs/'+job_id+'/cancel')
    assert client.post('/api/jobs/youtube',json=body).status_code==201
    client.post('/api/auth/logout')
    client.post('/api/auth/register',json={'email':'other@example.com','password':'long-password-123'})
    assert client.get('/api/jobs/'+job_id).status_code==404
