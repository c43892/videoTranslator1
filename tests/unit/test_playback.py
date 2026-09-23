import time

from fastapi.testclient import TestClient

from videotranslator.adapters.local_storage import LocalObjectStorage
from videotranslator.api.app import create_app
from videotranslator.domain.enums import JobStatus
from videotranslator.domain.models import Job


def test_private_playback_requires_owned_url_and_supports_ranges(container, user, tmp_path):
    storage = LocalObjectStorage(tmp_path/'media')
    container.storage = storage
    container.jobs._storage = storage
    with container.store.transaction() as tx:
        for job_id in ('one', 'two'):
            key = f'users/u1/jobs/{job_id}/result.mp4'
            storage.put_bytes(key, b'0123456789')
            tx.insert(Job(job_id=job_id, owner_user_id='u1', status=JobStatus.SUCCEEDED,
                          output_object_key=key, media_type='video'), job_id)
    client = TestClient(create_app(container))
    assert client.get('/api/v1/jobs/one/result').status_code == 401
    assert client.get('/api/v1/jobs/one/result',headers={'Authorization':'Bearer fake:other'}).status_code == 403
    response = client.get('/api/v1/jobs/one/result',headers={'Authorization':'Bearer fake:u1'})
    url = response.json()['download_url']
    partial = client.get(url, headers={'Range':'bytes=2-5'})
    assert partial.status_code == 206 and partial.content == b'2345'
    assert partial.headers['content-type'] == 'video/mp4'
    assert partial.headers['content-disposition'].startswith('inline')
    assert client.get(url+'&download=true').headers['content-disposition'].startswith('attachment')
    assert client.get(url.replace('/one/', '/two/')).status_code == 403
    assert client.get(url+'bad').status_code == 403
    expired = int(time.time())-10
    key = 'users/u1/jobs/one/result.mp4'
    assert client.get(f'/api/v1/jobs/one/playback?expires={expired}&token={storage._token(key, expired)}').status_code == 403
    with container.store.transaction() as tx:
        job = tx.get(Job,'one'); job.status = JobStatus.FAILED; tx.put(job,'one')
    assert client.get(url).status_code == 404


def test_default_storage_signatures_cannot_be_forged_from_a_public_default(tmp_path):
    a,b = LocalObjectStorage(tmp_path/'a'),LocalObjectStorage(tmp_path/'b')
    expires = int(time.time())+60
    assert not a.verify_url('result.mp4',expires,b._token('result.mp4',expires))


def test_subtitles_are_private_scoped_to_asset_and_removed_with_video(container, user, tmp_path):
    storage = LocalObjectStorage(tmp_path/'media')
    container.storage = container.jobs._storage = storage
    key = 'users/u1/jobs/subs/result.mp4'
    captions = 'WEBVTT\n\n00:00:00.250 --> 00:00:01.750\n你好，世界。\n'
    storage.put_bytes(key, b'video')
    storage.put_bytes(key+'.vtt', captions.encode())
    with container.store.transaction() as tx:
        tx.insert(Job(job_id='subs', owner_user_id='u1', status=JobStatus.SUCCEEDED,
                      output_object_key=key, input_object_key='input.mp4', media_type='video', target_language='zh'), 'subs')
    client = TestClient(create_app(container))
    assert client.get('/api/v1/jobs/subs/result', headers={'Authorization':'Bearer fake:other'}).status_code == 403
    data = client.get('/api/v1/jobs/subs/result', headers={'Authorization':'Bearer fake:u1'}).json()
    url = data['subtitle_url']
    assert data['subtitle_language'] == 'zh'
    response = client.get(url)
    assert response.status_code == 200 and response.text == captions
    assert response.headers['content-type'].startswith('text/vtt')
    assert client.get(data['download_url']+'&asset=subtitles').status_code == 403
    assert client.get(url.replace('asset=subtitles', 'asset=media')).status_code == 403
    assert client.get(url.replace('asset=subtitles', 'asset=unknown')).status_code == 404
    expired = int(time.time())-10
    assert client.get(f'/api/v1/jobs/subs/playback?asset=subtitles&expires={expired}&token={storage._token(key+".vtt", expired)}').status_code == 403
    container.jobs.delete_assets('subs', 'u1', now=1)
    assert not storage.exists(key+'.vtt')
    assert client.get(url).status_code == 404
