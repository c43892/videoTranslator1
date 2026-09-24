"""Run on the CPU VM: exchange its pull-only managed identity for an ACR login."""
import json
import subprocess
import urllib.parse
import urllib.request

registry = 'vtranslatorjpe43892.azurecr.io'
request = urllib.request.Request(
    'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fmanagement.azure.com%2F',
    headers={'Metadata': 'true'})
identity = json.load(urllib.request.urlopen(request, timeout=20))
request = urllib.request.Request('https://' + registry + '/oauth2/exchange',
    data=urllib.parse.urlencode({'grant_type': 'access_token', 'service': registry,
        'tenant': '024591ed-7dac-4345-b44b-01686ae9f7c7', 'access_token': identity['access_token']}).encode())
token = json.load(urllib.request.urlopen(request, timeout=20))['refresh_token']
subprocess.run(['docker', 'login', registry, '--username', '00000000-0000-0000-0000-000000000000',
                '--password-stdin'], input=token.encode(), check=True)
