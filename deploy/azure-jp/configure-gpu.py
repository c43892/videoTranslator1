"""Install least-privilege PostgreSQL roles and the idle business GPU revision."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
AZ = 'az.cmd' if sys.platform == 'win32' else 'az'
RESOURCE = ('https://management.azure.com/subscriptions/65168226-66f0-4c42-ba76-6dfde11c451e/'
    'resourceGroups/videotranslator-jpe-rg/providers/Microsoft.App/containerApps/videotranslator-gpu?api-version=2025-01-01')


def az(*args):
    result = subprocess.run([AZ, *args, '--only-show-errors', '-o', 'json'], capture_output=True, text=True, check=True,
        env=os.environ | {'PYTHONIOENCODING': 'utf-8'})
    return json.loads(result.stdout) if result.stdout.strip() else None


def main():
    state = json.loads((ROOT / 'secrets/azure-jp/state.json').read_text())
    # Generated passwords are restricted hexadecimal, never shell-interpolated.
    for field in ('worker_password', 'scaler_password'):
        assert all(c in '0123456789abcdef' for c in state[field])
    sql = r"""\set ON_ERROR_STOP on
DO $$ BEGIN
 IF EXISTS (SELECT FROM gpu_runnable_work) THEN RAISE EXCEPTION 'Refusing GPU deployment while work is active'; END IF;
 IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='gpuworker') THEN CREATE ROLE gpuworker LOGIN; END IF;
 IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='gpu_scaler') THEN CREATE ROLE gpu_scaler LOGIN; END IF;
END $$;
ALTER ROLE gpuworker PASSWORD '%s';
ALTER ROLE gpu_scaler PASSWORD '%s';
GRANT CONNECT ON DATABASE videotranslator TO gpuworker, gpu_scaler;
GRANT USAGE ON SCHEMA public TO gpuworker, gpu_scaler;
GRANT SELECT, UPDATE ON jobs, cloud_executions TO gpuworker;
GRANT SELECT ON gpu_workers TO gpuworker;
GRANT SELECT, INSERT, UPDATE ON gpu_provider_workers TO gpuworker;
GRANT SELECT ON gpu_runnable_work TO gpu_scaler;
""" % (state['worker_password'], state['scaler_password'])
    subprocess.run(['ssh', '-i', str(ROOT/'secrets/azure-jp-ed25519'),
        '-o', 'UserKnownHostsFile='+str(ROOT/'secrets/azure-known-hosts'), 'vtadmin@40.115.182.114',
        'sudo docker exec -i videotranslator-cloud-postgres-1 psql -U engineadmin -d videotranslator'],
        input=sql, text=True, check=True, capture_output=True)
    print('Worker/scaler database roles configured.', flush=True)
    for revision in az('containerapp', 'revision', 'list', '-g', 'videotranslator-jpe-rg', '-n', 'videotranslator-gpu'):
        if revision['properties'].get('active'):
            az('containerapp', 'revision', 'deactivate', '-g', 'videotranslator-jpe-rg', '-n', 'videotranslator-gpu',
                '--revision', revision['name'])
    body = json.loads((ROOT/'deploy/azure-jp/gpu.template.json').read_text())
    body['properties']['configuration']['secrets'] = json.loads((ROOT/'secrets/azure-jp/gpu-secrets.json').read_text())['properties']['configuration']['secrets']
    body['properties']['configuration']['ingress'] = None
    template = body['properties']['template']
    template['revisionSuffix'] = 'pipeline-' + str(int(time.time()))
    template['containers'][0]['image'] = dotenv_values(ROOT/'secrets/azure-jp/.env')['TTS_IMAGE']
    template['containers'][0]['env'].append({'name': 'HF_HUB_DISABLE_XET', 'value': '1'})
    engine_image = (sys.argv[1] if len(sys.argv) == 2 else
                    dotenv_values(ROOT/'secrets/azure-jp/.env')['ENGINE_IMAGE'])
    expected_prefix = 'vtranslatorjpe43892.azurecr.io/videotranslator/engine@sha256:'
    if not engine_image.startswith(expected_prefix) or len(engine_image) != len(expected_prefix) + 64:
        raise ValueError('Use the immutable project engine image digest')
    template['containers'][1]['image'] = engine_image
    env = dotenv_values(ROOT/'secrets/azure-jp/engine.env')
    for name in ('DEEPSEEK_MODEL','DEEPSEEK_BASE_URL','SEPARATION_PROVIDER','DEMUCS_MODEL','DEMUCS_SEGMENT_SECONDS','TTS_MAX_TEXT_TOKENS'):
        if name in env:
            template['containers'][1]['env'].append({'name': name, 'value': env[name]})
    body_path = ROOT/'secrets/azure-jp/gpu-business.json'
    body_path.write_text(json.dumps(body), encoding='utf-8')
    result = az('rest', '--method', 'patch', '--url', RESOURCE, '--body', '@'+str(body_path))
    print('Business GPU revision submitted with min=0/max=1; CPU admission is controlled separately.', flush=True)
    print(json.dumps({'revision':template['revisionSuffix']}))


if __name__ == '__main__':
    main()
