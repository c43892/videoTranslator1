"""Public HTTPS validation with a dedicated Firebase user and Stripe sandbox."""
import base64
import json
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs, urlsplit

import firebase_admin
from firebase_admin import auth, credentials
import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
HOST = 'https://vidyi.cc'
PRIVATE = ROOT / 'secrets/azure-jp'
USER = 'azure-validation-20260923'


def client():
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(str(ROOT/'secrets/firebase-admin.json')))
    # Dedicated synthetic test identity; never mint credentials for a real user.
    token = auth.create_custom_token(USER, {'email':USER+'@example.invalid',
        'email_verified':True, 'name':'Azure validation'}).decode()
    api_key = dotenv_values(PRIVATE/'studio.env')['FIREBASE_API_KEY']
    response = httpx.post('https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken',
        params={'key':api_key}, json={'token':token,'returnSecureToken':True}, timeout=30)
    response.raise_for_status()
    return httpx.Client(base_url=HOST, headers={'Authorization':'Bearer '+response.json()['idToken'],
        'X-Payment-Mode':'sandbox'}, timeout=120)


def check(response, expected=200):
    if response.status_code != expected:
        raise RuntimeError(f'API status {response.status_code}: {response.text[:300]}')
    return response.json()


def main(action):
    state_path = PRIVATE/'web-validation.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    with client() as api:
        if action == 'prepare':
            assert check(api.get('/api/v1/health/ready'))
            config = check(api.get('/api/v1/chat/config'))
            assert config['payment_mode']=='sandbox' and config['profile']=='azure-jp-t4'
            identity = check(api.get('/api/v1/me'))
            assert identity['user_id']==USER and identity['email_verified']
            assert httpx.get(HOST+'/api/v1/me',timeout=30).status_code==401
            assert api.get('/api/v1/me',headers={'X-Payment-Mode':'live'}).status_code==409
            print('HTTPS, Firebase token verification, sandbox separation and anonymous denial passed.', flush=True)
            body={'package_id':'points_1_v1','provider':'stripe','idempotency_key':'azure-validation-20260923-checkout'}
            payment=check(api.post('/api/v1/billing/sessions',json=body),201)
            assert payment==check(api.post('/api/v1/billing/sessions',json=body),201)
            state['payment']=payment
            state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
            print('Sandbox checkout created; idempotency passed. Checkout saved privately.',flush=True)
        elif action == 'upload':
            data=(ROOT/'vt-data/azure-tmp/cloud-validation.mp4').read_bytes()
            upload=check(api.post('/api/v1/uploads',json={'filename':'cloud-validation.mp4','size_bytes':len(data),'target_language':'zh'}),201)
            sas=upload['upload_url']
            assert parse_qs(urlsplit(sas).query)['sp'][0] in ('cw','wc')
            assert httpx.get(sas,timeout=30).status_code==403
            preflight=httpx.options(sas,headers={'Origin':HOST,'Access-Control-Request-Method':'PUT',
                'Access-Control-Request-Headers':'content-type,x-ms-blob-type'},timeout=30)
            assert preflight.status_code==200 and preflight.headers['Access-Control-Allow-Origin']==HOST
            block=base64.b64encode(b'validation-000001').decode()
            response=httpx.put(sas+'&comp=block&blockid='+block,content=data,headers={'Content-Type':'application/octet-stream'},timeout=120)
            assert response.status_code==201, response.status_code
            response=httpx.put(sas+'&comp=blocklist',content=f'<BlockList><Latest>{block}</Latest></BlockList>'.encode(),
                headers={'Content-Type':'application/xml','x-ms-blob-content-type':'video/mp4'},timeout=30)
            assert response.status_code==201, response.status_code
            check(api.post('/api/v1/uploads/'+upload['upload_id']+'/renew'))
            job=check(api.post('/api/v1/uploads/'+upload['upload_id']+'/complete'),201)
            state['job_id']=job['job_id']
            state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
            print('Blob block upload, scoped SAS, CORS, renewal and commit passed. Job:',job['job_id'],flush=True)
            for _ in range(60):
                job=check(api.get('/api/v1/jobs/'+state['job_id']))
                if job['status']!='inspecting':
                    print(json.dumps({k:job.get(k) for k in ('job_id','status','duration_ms','error_code')}),flush=True)
                    return
                time.sleep(3)
            raise TimeoutError('CPU inspection did not finish')
        elif action=='status':
            print('Balance:',check(api.get('/api/v1/me'))['point_balance_units'])
            if state.get('job_id'):
                job=check(api.get('/api/v1/jobs/'+state['job_id']))
                print(json.dumps({k:job.get(k) for k in ('job_id','status','duration_ms','failure_code','failure_message','progress','stage')}))
            payment=check(api.post('/api/v1/billing/payments/'+state['payment']['payment_id']+'/reconcile'))
            print('Payment:',payment['status'])
        elif action=='start':
            print(json.dumps(check(api.post('/api/v1/jobs/'+state['job_id']+'/start'))))
        elif action=='watch':
            started=time.time()
            last=None
            while time.time()-started<1800:
                job=check(api.get('/api/v1/jobs/'+state['job_id']))
                summary={k:job.get(k) for k in ('job_id','status','progress_percent','stage','error_code','error_message')}
                if summary!=last:
                    print(json.dumps(summary),flush=True)
                    last=summary
                evidence=ROOT/'vt-data/azure-tmp/cloud-job-result.json'
                evidence.write_text(json.dumps(job,indent=2),encoding='utf-8')
                if job['status']=='succeeded':
                    result=check(api.get('/api/v1/jobs/'+state['job_id']+'/result'))
                    state['result']=result
                    state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
                    url=result['download_url']
                    assert parse_qs(urlsplit(url).query)['sp']==['r']
                    anonymous=httpx.get(url.split('?')[0],timeout=30)
                    assert (anonymous.status_code in (403,404) or
                        (anonymous.status_code==409 and 'PublicAccessNotPermitted' in anonymous.text))
                    response=httpx.get(url,timeout=120)
                    assert response.status_code==200
                    (ROOT/'vt-data/azure-tmp/cloud-validation-result.mp4').write_bytes(response.content)
                    if 'subtitle_url' in result:
                        response=httpx.get(result['subtitle_url'],timeout=30)
                        assert response.status_code==200 and response.text.startswith('WEBVTT')
                        (ROOT/'vt-data/azure-tmp/cloud-validation-result.vtt').write_text(response.text,encoding='utf-8')
                    print('Cloud translation, private result SAS download and subtitles passed.',flush=True)
                    return
                if job['status'] in ('failed','cancelled','needs_review'):
                    raise RuntimeError('Cloud job ended without success')
                time.sleep(15)
            api.post('/api/v1/jobs/'+state['job_id']+'/cancel')
            raise TimeoutError('Validation deadline exceeded; cancellation requested')
        else:
            raise ValueError('Unknown validation action')


if __name__=='__main__':
    main(sys.argv[1])
