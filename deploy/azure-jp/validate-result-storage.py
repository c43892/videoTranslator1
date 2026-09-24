"""Run in the CPU web container with the owned validation result key as argv[1]."""
import sys
import time
from urllib.parse import parse_qs, urlsplit
import httpx
from videotranslator.adapters.azure_blob import AzureBlobStorage

key=sys.argv[1]
assert key.startswith('outputs/users/azure-validation-20260923/')
storage=AzureBlobStorage.from_env()
url=storage.create_download_url(key,10)
assert parse_qs(urlsplit(url).query)['sp']==['r']
response=httpx.get(url,headers={'Range':'bytes=0-127'},timeout=15)
assert response.status_code==206 and len(response.content)==128
assert response.headers['Content-Type'].startswith('video/mp4')
time.sleep(12)
assert httpx.get(url,timeout=15).status_code==403
print('Private video supports ranged playback; read-only SAS expires and is rejected.')
