import shutil
import subprocess
import time
import uuid

import pytest

from videotranslator.adapters.local_jobs import LocalCpuInspectionBackend
from videotranslator.domain.models import InspectionSpec


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='ffmpeg required')
def test_real_inspection_subprocess_finishes_and_releases_slot(tmp_path, monkeypatch):
    monkeypatch.setenv('APP_PROFILE', 'local-ui')
    monkeypatch.setenv('LOCAL_STORAGE_DIR', str(tmp_path))
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=duration=1',
                    str(tmp_path / 'input.wav')], check=True)
    backend = LocalCpuInspectionBackend(tmp_path / 'queue.db')
    identifiers = [uuid.uuid4().hex, uuid.uuid4().hex]
    for identifier in identifiers:
        backend.submit(InspectionSpec(identifier, 1, 'obj://input.wav', 'obj://result.json'), identifier)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        states = [backend.get_status(identifier).state for identifier in identifiers]
        if all(state == 'succeeded' for state in states):
            break
        if 'failed' in states:
            pytest.fail(str([backend.get_status(identifier) for identifier in identifiers]))
        time.sleep(.1)
    assert states == ['succeeded', 'succeeded']
    assert backend.get_result(identifiers[0]).duration_ms == 1000
    assert not backend._queue._processes
