"""Local separation adapter; shared GPU service owns model residency."""
import time
import httpx
from .domain import Stems, Cancelled, ProviderError
from .providers import require_success


class DemucsSeparator:
    def __init__(self, config, storage):
        self.config, self.storage = config, storage

    def separate(self, audio, prefix, checkpoint, remote, check_cancel):
        base = self.config.demucs_url.rstrip('/')
        with httpx.Client(timeout=30) as client:
            check_cancel()
            state = require_success(client.post(base+'/separations', json={
                'audio': audio, 'prefix': prefix+'/demucs', 'model': self.config.demucs_model,
                'segment': self.config.demucs_segment_seconds}), 'Demucs')
            job_id = state['id']
            checkpoint({'provider':'demucs', 'id':job_id})
            try:
                while state['status'] in ('queued','running'):
                    check_cancel()
                    time.sleep(1)
                    response = client.get(base+'/separations/'+job_id)
                    if response.status_code == 404:
                        raise ProviderError('Demucs 服务已重启，请重试以继续处理')
                    state = require_success(response, 'Demucs')
                check_cancel()
            except Cancelled:
                response = client.delete(base+'/separations/'+job_id)
                require_success(response, 'Demucs cancellation')
                raise
            if state['status'] != 'completed':
                raise ProviderError(state.get('error') or 'Demucs 分离未完成，请重试')
            result = Stems(**state['stems'])
            for key in (result.dialogue, result.music, result.effects):
                if not key.startswith(prefix+'/demucs/') or not self.storage.exists(key):
                    raise ProviderError('Demucs returned an invalid output')
            return result


def create_separator(config, storage):
    if config.separation_provider == 'demucs':
        return DemucsSeparator(config, storage)
    from .providers import MVSeparator
    return MVSeparator(config, storage)
