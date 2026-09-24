from pathlib import Path
import pytest
from videotranslator import media
from videotranslator.storage import LocalStorage
from videotranslator.providers import identify_stems, require_success
from videotranslator.domain import ProviderError
import httpx

def register(client, email='test@example.com'):
    return client.post('/api/auth/register',json={'email':email,'password':'a-long-test-password'})

def test_auth_and_csrf(client):
    assert client.get('/api/jobs').status_code==401
    assert client.post('/api/auth/register',headers={'X-Requested-With':''},json={'email':'a@b.com','password':'password12345'}).status_code==403
    assert register(client).status_code==200
    assert client.get('/api/auth/me').json()['email']=='test@example.com'
    assert client.post('/api/auth/logout').status_code==200
    assert client.get('/api/jobs').status_code==401
    assert client.post('/api/auth/login',json={'email':'test@example.com','password':'a-long-test-password'}).status_code==200

def test_upload_without_credentials_and_owner_isolation(client,tmp_path):
    register(client)
    video=tmp_path/'test.mp4'
    media.ffmpeg(['-f','lavfi','-i','color=s=160x90:d=2','-f','lavfi','-i','sine=duration=2',
                  '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',video])
    with video.open('rb') as f:
        result=client.post('/api/jobs',data={'target_language':'zh'},files={'file':('test.mp4',f,'video/mp4')})
    assert result.status_code==201, result.text
    job_id=result.json()['id']
    assert result.json()['status']=='waiting_configuration'
    assert client.get(f'/api/jobs/{job_id}/files/original',headers={'Range':'bytes=0-99'}).status_code==206
    assert client.post(f'/api/jobs/{job_id}/cancel').status_code==200
    assert client.post(f'/api/jobs/{job_id}/retry').status_code==200
    client.post('/api/auth/logout')
    register(client,'second@example.com')
    assert client.get(f'/api/jobs/{job_id}').status_code==404
    assert client.get(f'/api/jobs/{job_id}/files/original').status_code==404
    assert client.post(f'/api/jobs/{job_id}/retry').status_code==404

def test_invalid_video_rejected(client):
    register(client)
    response=client.post('/api/jobs',data={'target_language':'zh'},files={'file':('fake.mp4',b'not a video','video/mp4')})
    assert response.status_code==422

def test_storage_traversal_is_rejected(tmp_path):
    storage=LocalStorage(str(tmp_path))
    with pytest.raises(ValueError):
        storage.path('../outside')
    with pytest.raises(ValueError):
        storage.path('/etc/passwd')

def test_mvsep_requires_three_correct_stems():
    files=[{'type':name,'url':'https://de2.mvsep.com/'+name+'.wav'} for name in ('speech','music','sfx')]
    assert set(identify_stems(files))=={'dialogue','music','effects'}
    with pytest.raises(ProviderError):
        identify_stems(files[:2])

def test_provider_errors_do_not_echo_secrets():
    response=httpx.Response(401,text='secret-account-key')
    with pytest.raises(ProviderError) as exc:
        require_success(response,'Example')
    assert 'secret-account-key' not in str(exc.value)

def test_services_never_return_key_values(client):
    register(client)
    state=client.get('/api/services').json()
    assert state['missing_keys']==['OPENAI_API_KEY','DEEPSEEK_API_KEY']
    assert 'Demucs' in state['providers']['separation']

def test_partial_result_can_be_edited_and_one_segment_retried_with_owner_checks(client):
    from sqlalchemy import select
    from videotranslator.db import Session, User, Job
    from videotranslator.api import storage
    register(client)
    with Session.begin() as db:
        user=db.scalar(select(User).where(User.email=='test@example.com'))
        db.add(Job(id='partial',user_id=user.id,filename='test.mp4',input_key='input.mp4',
                   target_language='zh',status='completed_with_warnings'))
    storage.write_json('jobs/partial/manifest.json',{'segments':[
        {'id':'bad','speaker_id':'A','translation':'旧译文'},
        {'id':'good','speaker_id':'A','translation':'保留'}]})
    result=client.patch('/api/jobs/partial/segments',json={'translations':{'bad':'短译文'}})
    assert result.status_code==200
    assert client.post('/api/jobs/partial/segments/missing/retry').status_code==404
    assert client.post('/api/jobs/partial/segments/bad/retry').status_code==200
    overrides=storage.read_json('jobs/partial/overrides.json')
    assert overrides['translations']=={'bad':'短译文'}
    assert set(overrides['synthesis_revisions'])=={'bad'}
    assert client.get('/api/jobs/partial').json()['status']=='queued'
    assert client.post('/api/jobs/partial/segments/bad/retry').status_code==409
    client.post('/api/auth/logout')
    register(client,'outsider@example.com')
    assert client.post('/api/jobs/partial/segments/bad/retry').status_code==404
