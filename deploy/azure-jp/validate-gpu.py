"""Bounded, attended B0 validation. Always return the test GPU revision to zero.

Usage: python validate-gpu.py REGISTRY/REPOSITORY@sha256:DIGEST
The current app must have zero replicas. This temporarily replaces its test image.
Does not submit customer work or expose a public GPU endpoint.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

GROUP = 'videotranslator-jpe-rg'
APP = 'videotranslator-gpu'
RESOURCE = '/subscriptions/65168226-66f0-4c42-ba76-6dfde11c451e/resourceGroups/' + GROUP + '/providers/Microsoft.App/containerApps/' + APP
AZ = 'az.cmd' if sys.platform == 'win32' else 'az'
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'vt-data' / 'azure-tmp'


def az(*args, timeout=90):
    result = subprocess.run([AZ, *args, '--only-show-errors', '-o', 'json'],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout) if result.stdout.strip() else None


def revisions():
    return az('containerapp', 'revision', 'list', '-g', GROUP, '-n', APP)


def apply(body, name):
    path = EVIDENCE / name
    path.write_text(json.dumps(body), encoding='utf-8')
    return az('rest', '--method', 'patch', '--url', 'https://management.azure.com' + RESOURCE + '?api-version=2025-01-01',
              '--body', '@' + str(path))


def main(image):
    if '@sha256:' not in image or not image.startswith('vtranslatorjpe43892.azurecr.io/'):
        raise ValueError('Use the verified project registry image digest')
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    previous = az('containerapp', 'show', '-g', GROUP, '-n', APP)
    if any(r['properties'].get('replicas', 0) for r in revisions()):
        raise RuntimeError('GPU app is not idle; refusing to interrupt work')
    (EVIDENCE / 'gpu-before-validation.json').write_text(json.dumps(previous), encoding='utf-8')
    body = {
        'identity': {'type': 'SystemAssigned'},
        'properties': {
            'workloadProfileName': 'gpu-t4',
            'configuration': {
                'activeRevisionsMode': 'Multiple', 'ingress': None,
                'registries': [{'server': 'vtranslatorjpe43892.azurecr.io', 'identity': 'system'}]},
            'template': {
                'revisionSuffix': 'b0-' + str(int(time.time())),
                'containers': [{'name': 'tts', 'image': image,
                    'resources': {'cpu': 8, 'memory': '56Gi'},
                    'env': [{'name': 'HF_HUB_DISABLE_XET', 'value': '1'}],
                    'volumeMounts': [{'volumeName': 'data', 'mountPath': '/data'}, {'volumeName': 'models', 'mountPath': '/models'}],
                    'probes': [{'type': 'Readiness', 'httpGet': {'path': '/health', 'port': 8001},
                                'initialDelaySeconds': 10, 'periodSeconds': 15, 'timeoutSeconds': 5, 'failureThreshold': 48}]}],
                'volumes': [{'name': n, 'storageType': 'AzureFile', 'storageName': n} for n in ('data', 'models')],
                'scale': {'minReplicas': 1, 'maxReplicas': 1, 'cooldownPeriod': 60, 'rules': []}}}}
    started = time.time()
    success = False
    try:
        apply(body, 'gpu-validation.json')
        print('GPU validation submitted; maximum attended window is 25 minutes.', flush=True)
        last = None
        while time.time() - started < 1500:
            app = az('containerapp', 'show', '-g', GROUP, '-n', APP)
            current = revisions()
            summary = [(r['name'], r['properties'].get('runningState'), r['properties'].get('healthState'),
                        r['properties'].get('replicas')) for r in current if r['properties'].get('active')]
            if summary != last:
                print(json.dumps(summary), flush=True)
                last = summary
            (EVIDENCE / 'gpu-validation-revisions.json').write_text(json.dumps(current), encoding='utf-8')
            if app['properties'].get('provisioningState') == 'Failed':
                raise RuntimeError('Container app provisioning failed')
            target = [r for r in current if r['name'].endswith(body['properties']['template']['revisionSuffix'])]
            if target and target[0]['properties'].get('healthState') == 'Healthy' and target[0]['properties'].get('replicas', 0) > 0:
                # Readiness /health becomes 200 only after IndexTTS2 CUDA model load.
                success = True
                print('TTS model readiness passed on Azure T4.', flush=True)
                break
            time.sleep(20)
        if not success:
            raise TimeoutError('GPU startup did not pass within the validation window')
    finally:
        # Deactivate first so even a failed minReplicas update cannot leave GPUs on.
        errors = []
        for revision in revisions():
            if revision['properties'].get('active'):
                try:
                    az('containerapp', 'revision', 'deactivate', '-g', GROUP, '-n', APP, '--revision', revision['name'])
                except Exception as exc:
                    errors.append(str(exc))
        try:
            # Use the last accepted template, not a potentially rejected input.
            accepted = az('containerapp', 'show', '-g', GROUP, '-n', APP)['properties']['template']
            accepted['scale']['minReplicas'] = 0
            accepted['revisionSuffix'] = 'idle-' + str(int(time.time()))
            apply({'properties': {'template': accepted}}, 'gpu-idle.json')
            # ARM PATCH is asynchronous. Wait for the revision to exist before
            # deactivating; otherwise a late rollout can start it after cleanup.
            cleanup_deadline = time.time() + 300
            stable = 0
            while time.time() < cleanup_deadline:
                app = az('containerapp', 'show', '-g', GROUP, '-n', APP)
                current = revisions()
                for revision in current:
                    if revision['properties'].get('active'):
                        az('containerapp', 'revision', 'deactivate', '-g', GROUP, '-n', APP, '--revision', revision['name'])
                settled = (app['properties'].get('provisioningState') in ('Succeeded', 'Failed')
                    and not any(r['properties'].get('active') or r['properties'].get('replicas', 0) for r in current))
                stable = stable + 1 if settled else 0
                if stable >= 2:
                    break
                time.sleep(10)
            if stable < 2:
                raise RuntimeError('GPU zero-replica state was not confirmed')
        except Exception as exc:
            errors.append(str(exc))
        report = {'image': image, 'model_ready': success, 'elapsed_seconds': round(time.time() - started),
                  'cleanup_errors': errors, 'revisions': revisions()}
        (EVIDENCE / 'gpu-validation-result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k != 'revisions'}), flush=True)
        if errors:
            raise RuntimeError('GPU cleanup requires immediate attention')


if __name__ == '__main__':
    main(sys.argv[1])
