import os
import tempfile
os.environ['DATABASE_URL'] = 'sqlite:///' + tempfile.mktemp(suffix='.db')
os.environ['STORAGE_ROOT'] = tempfile.mkdtemp(prefix='vt-tests-')
os.environ['MVSEP_API_KEY'] = ''
os.environ['OPENAI_API_KEY'] = ''
os.environ['DEEPSEEK_API_KEY'] = ''

import pytest
from fastapi.testclient import TestClient
from videotranslator.api import app, _attempts
from videotranslator.db import Base, engine

@pytest.fixture
def client():
    Base.metadata.drop_all(engine)
    _attempts.clear()
    with TestClient(app, headers={'X-Requested-With': 'VideoTranslator'}) as client:
        yield client
