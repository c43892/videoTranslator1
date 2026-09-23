import os

os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['STUDIO_PUBLIC_URL'] = 'http://localhost:8090/'

from fastapi.testclient import TestClient
from videotranslator.api import app


def test_old_bookmarks_enter_current_studio_without_forwarding_credentials():
    client = TestClient(app)
    for path in ('/', '/index.html?token=do-not-forward'):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 307
        assert response.headers['location'] == 'http://localhost:8090/'
        assert response.headers['cache-control'] == 'no-store'


def test_engine_api_keeps_its_auth_boundary():
    response = TestClient(app).get('/api/auth/me', follow_redirects=False)
    assert response.status_code == 401
    assert 'location' not in response.headers
