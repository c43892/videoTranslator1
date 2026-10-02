import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('httpx'), 'Run with httpx or in the GPU agent image')
class AgentStatusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.status_path = Path(self.directory.name) / 'status.json'
        module_dir = os.environ.get('GPU_AGENT_MODULE_DIR')
        if not module_dir:
            module_dir = Path(__file__).resolve().parents[2] / 'deploy/home-gpu-worker'
        self.path_patch = patch.object(sys, 'path', [str(module_dir), *sys.path])
        self.path_patch.start()
        self.env_patch = patch.dict(os.environ, {
            'GPU_AGENT_SERVER_URL': 'https://example.invalid', 'GPU_WORKER_TOKEN': 'x' * 48,
            'GPU_PROVIDER_ID': 'gpu_unit', 'GPU_PROVIDER_TYPE': 'local',
            'GPU_AGENT_DATA_ROOT': self.directory.name,
        })
        self.env_patch.start()
        spec = importlib.util.spec_from_file_location('agent_test', Path(module_dir) / 'agent.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        import runtime_status
        self.status = runtime_status
        self.writer_patch = patch.object(self.module, 'write_status',
            side_effect=lambda *args, **kwargs: runtime_status.write_status(*args, **kwargs, path=self.status_path))
        self.writer_patch.start()
        self.agent = self.module.Agent()
        self.agent.ready = lambda: True

    def tearDown(self):
        self.agent.local.close()
        self.agent.remote.close()
        self.writer_patch.stop()
        self.env_patch.stop()
        self.path_patch.stop()
        self.directory.cleanup()

    def respond(self, method, path, **kwargs):
        self.agent.stopping.set()
        return self.module.httpx.Response(200, json={'cancelled': ['task_test']})

    def test_successful_heartbeat_publishes_fresh_ready_state(self):
        self.agent.request = self.respond
        self.agent.heartbeat_loop()
        state = self.status.read_status(self.status_path)
        self.assertTrue(self.status.healthy(state))
        self.assertEqual(state['provider_id'], 'gpu_unit')

    def test_connection_failure_does_not_refresh_cloud_ack(self):
        self.status.write_status('gpu_unit', ready=True, registered=True,
                                 acknowledged_at=time.time() - 60, path=self.status_path)
        def fail(*args, **kwargs):
            self.agent.stopping.set()
            raise self.module.httpx.ConnectError('offline')
        self.agent.request = fail
        self.agent.heartbeat_loop()
        self.assertFalse(self.status.healthy(self.status.read_status(self.status_path)))

    def test_registration_failure_resets_stale_ready_state(self):
        self.status.write_status('gpu_unit', ready=True, registered=True,
                                 acknowledged_at=time.time(), path=self.status_path)
        def fail(*args, **kwargs):
            raise self.module.httpx.ConnectError('offline')
        self.agent.request = fail
        with self.assertRaises(self.module.httpx.ConnectError):
            self.agent.run()
        self.assertFalse(self.status.healthy(self.status.read_status(self.status_path)))

    def test_status_write_failure_does_not_break_task_cancellation(self):
        self.agent.request = self.respond
        self.agent.active = {'id': 'task_test', 'lease_token': 'test_lease'}
        with patch.object(self.module, 'write_status', side_effect=OSError('disk full')), \
                patch.object(self.module, 'gpu_active', return_value=False):
            self.agent.heartbeat_loop()
        self.assertTrue(self.agent.cancelled.is_set())


if __name__ == '__main__':
    unittest.main()
