"""Resource exclusivity, failure handling, and cloud acknowledgement checks."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = load('local_service', ROOT / 'deploy/local_service.py')
status = load('runtime_status', ROOT / 'deploy/home-gpu-worker/runtime_status.py')


class RecordingDeployment(cli.Deployment):
    def __init__(self, fail_stop=False, fail_health=False):
        super().__init__()
        self.calls = []
        self.fail_stop = fail_stop
        self.fail_health = fail_health

    def configuration(self, mode):
        return {'services': {}, 'volumes': {'models': {'external': True, 'name': 'retained-models'}}}

    def container_ids(self, mode):
        return []

    def run(self, args, **kwargs):
        self.calls.append(tuple(args))
        return subprocess.CompletedProcess(args, 0, '', '')

    def stop_mode(self, mode):
        self.calls.append(('stop_mode', mode))
        if self.fail_stop:
            raise RuntimeError('Stop failed')

    def compose(self, mode, *args, **kwargs):
        self.calls.append((mode, *args))

    def wait_json(self, container, code, predicate, label):
        self.calls.append(('wait', label))
        if self.fail_health:
            raise RuntimeError('Health failed')
        return {'model': 'IndexTTS2 2.5.0', 'provider_id': 'gpu_test'}


class LocalDeploymentTests(unittest.TestCase):
    def test_concurrent_switch_is_rejected_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cli.Path, 'home', return_value=Path(directory)):
            with cli.mode_lock():
                with self.assertRaises(RuntimeError):
                    with cli.mode_lock():
                        self.fail('A concurrent switch acquired the lock')
            with cli.mode_lock():
                pass

    def test_modes_stop_opposite_before_gpu_and_wait_before_remaining_services(self):
        for mode, opposite in [('demo', 'provider'), ('provider', 'demo')]:
            d = RecordingDeployment()
            d.start(mode)
            stop = d.calls.index(('stop_mode', opposite))
            gpu = d.calls.index((mode, 'up', '-d', '--no-build', 'tts'))
            ready = d.calls.index(('wait', 'GPU'))
            rest = d.calls.index((mode, 'up', '-d', '--no-build'))
            self.assertLess(stop, gpu)
            self.assertLess(gpu, ready)
            self.assertLess(ready, rest)

    def test_failed_stop_never_starts_a_second_gpu(self):
        d = RecordingDeployment(fail_stop=True)
        with self.assertRaises(RuntimeError):
            d.start('provider')
        self.assertFalse(any('up' in call for call in d.calls))

    def test_failed_gpu_health_does_not_enable_web_or_claiming(self):
        d = RecordingDeployment(fail_health=True)
        with self.assertRaises(RuntimeError):
            d.start('demo')
        self.assertNotIn(('demo', 'up', '-d', '--no-build'), d.calls)

    def test_invalid_configuration_does_not_stop_current_mode(self):
        d = RecordingDeployment()
        with patch.object(d, 'configuration', side_effect=RuntimeError('Bad token')):
            with self.assertRaises(RuntimeError):
                d.start('provider')
        self.assertEqual(d.calls, [])

    def test_stop_needs_no_env_and_does_not_delete_volumes(self):
        d = cli.Deployment()
        calls = []
        with patch.object(d, 'container_ids', side_effect=[['id1', 'id2'], []]), \
                patch.object(d, 'run', side_effect=lambda args, **kw: calls.append(args)):
            d.stop_mode('provider')
        self.assertEqual(calls, [['stop', '--time', '30', 'id1', 'id2'], ['rm', 'id1', 'id2']])
        self.assertFalse(any('-v' in call for call in calls))

    def test_initialization_preserves_existing_keys_and_generates_private_demo_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['.env.local-gpu', '.env.home-gpu']:
                (root / (name + '.example')).write_text((ROOT / (name + '.example')).read_text())
            with patch.object(cli, 'ROOT', root):
                cli.initialize()
                original = (root / '.env.local-gpu').read_text()
                values = dict(line.split('=', 1) for line in original.splitlines() if line and not line.startswith('#'))
                self.assertGreaterEqual(len(values['ENGINE_CONTROL_TOKEN']), 32)
                self.assertNotEqual(values['ENGINE_CONTROL_TOKEN'], values['POSTGRES_PASSWORD'])
                cli.initialize()
                self.assertEqual(original, (root / '.env.local-gpu').read_text())

    def test_cloud_health_requires_readiness_registration_and_fresh_ack(self):
        state = dict(registered=True, ready=True, acknowledged_at=100, provider_id='gpu_test')
        self.assertTrue(status.healthy(state, now=129))
        self.assertFalse(status.healthy(state, now=130))
        self.assertFalse(status.healthy(state, now=99))
        self.assertFalse(status.healthy(state | {'registered': False}, now=101))
        self.assertFalse(status.healthy(state | {'ready': False}, now=101))
        self.assertFalse(status.healthy({}, now=101))

    def test_runtime_status_persists_only_nonsecret_health_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            self.assertEqual(status.read_status(path), {})
            status.write_status('gpu_test', registered=True, ready=True, acknowledged_at=100, path=path)
            saved = status.read_status(path)
            self.assertEqual(set(saved), {'provider_id', 'registered', 'ready', 'acknowledged_at', 'busy'})
            self.assertTrue(status.healthy(saved, now=101))
            path.write_text('partial invalid json')
            self.assertEqual(status.read_status(path), {})


if __name__ == '__main__':
    unittest.main()
