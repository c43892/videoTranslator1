"""Private agent health: readiness requires a recent acknowledged cloud heartbeat."""
import json
import os
from pathlib import Path
import time

STATUS_PATH = Path('/tmp/gpu-agent-status.json')


def write_status(provider_id, *, ready, registered, acknowledged_at=0, busy=False,
                 path=STATUS_PATH):
    state = dict(provider_id=provider_id, ready=ready, registered=registered,
                 acknowledged_at=acknowledged_at, busy=busy)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state), encoding='utf-8')
    os.replace(temporary, path)


def read_status(path=STATUS_PATH):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def healthy(state, now=None):
    age = (time.time() if now is None else now) - state.get('acknowledged_at', 0)
    return bool(state.get('registered') and state.get('ready') and 0 <= age < 30)


if __name__ == '__main__':
    state = read_status()
    print(json.dumps(state))
    raise SystemExit(0 if healthy(state) else 1)
