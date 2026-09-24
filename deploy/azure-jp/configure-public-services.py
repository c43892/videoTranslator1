"""Preserve existing Firebase domains/CORS rules and add the validation origin."""
import json
from pathlib import Path
import subprocess
import sys

from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

ROOT = Path(__file__).resolve().parents[2]
HOST = 'vidyi.cc'
AZ = 'az.cmd' if sys.platform == 'win32' else 'az'


def az(*args):
    result = subprocess.run([AZ, *args, '-o', 'json', '--only-show-errors'], capture_output=True, text=True, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def main():
    resource = ('https://management.azure.com/subscriptions/65168226-66f0-4c42-ba76-6dfde11c451e/'
        'resourceGroups/videotranslator-jpe-rg/providers/Microsoft.Storage/storageAccounts/'
        'vtranslatorjpe43892/blobServices/default?api-version=2023-05-01')
    current = az('rest', '--method', 'get', '--url', resource)
    rules = current['properties'].get('cors', {}).get('corsRules', [])
    origin = 'https://' + HOST
    rules = [r for r in rules if origin not in r.get('allowedOrigins', [])]
    rules.append({'allowedOrigins': [origin], 'allowedMethods': ['PUT', 'GET', 'HEAD', 'OPTIONS'],
        'allowedHeaders': ['content-type', 'x-ms-*'], 'exposedHeaders': ['Content-Length', 'ETag', 'x-ms-request-id'],
        'maxAgeInSeconds': 600})
    body = ROOT / 'vt-data/azure-tmp/blob-cors.json'
    body.write_text(json.dumps({'properties': {'cors': {'corsRules': rules}}}), encoding='utf-8')
    result = az('rest', '--method', 'put', '--url', resource, '--body', '@' + str(body))
    assert any(origin in r['allowedOrigins'] for r in result['properties']['cors']['corsRules'])
    print('Blob CORS configured for the exact validation HTTPS origin.', flush=True)
    credentials = service_account.Credentials.from_service_account_file(str(ROOT / 'secrets/firebase-admin.json'),
        scopes=['https://www.googleapis.com/auth/cloud-platform'])
    session = AuthorizedSession(credentials)
    url = 'https://identitytoolkit.googleapis.com/admin/v2/projects/' + credentials.project_id + '/config'
    response = session.get(url, timeout=30)
    response.raise_for_status()
    domains = response.json().get('authorizedDomains', [])
    if HOST not in domains:
        response = session.patch(url, params={'updateMask': 'authorizedDomains'},
            json={'authorizedDomains': domains + [HOST]}, timeout=30)
        response.raise_for_status()
    assert HOST in response.json()['authorizedDomains']
    print('Firebase authorized domain configured; existing domains preserved.')


if __name__ == '__main__':
    main()
