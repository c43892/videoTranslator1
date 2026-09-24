"""Generate isolated cloud secrets locally; never print credentials or copy live keys."""
import json
import os
from pathlib import Path
import secrets
import subprocess

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / 'secrets' / 'azure-jp'
HOST = 'vidyi.cc'
ACR = 'vtranslatorjpe43892.azurecr.io'


def write_env(path, values):
    path.write_text(''.join(k + '=' + json.dumps(str(v).replace('$', '$$')) + '\n' for k, v in values.items()), encoding='utf-8')


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    local = dotenv_values(ROOT / '.env')
    stripe_key = local.get('STRIPE_SANDBOX_SECRET_KEY', '')
    if local.get('PAYMENT_MODE') != 'sandbox' or not stripe_key.startswith(('sk_test_', 'rk_test_')):
        raise ValueError('Local Stripe configuration must explicitly select sandbox')
    state_file = PRIVATE / 'state.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        'engine_token': secrets.token_hex(32), 'postgres_password': secrets.token_hex(24),
        'worker_password': secrets.token_hex(24), 'scaler_password': secrets.token_hex(24)}
    if not state.get('stripe_webhook_secret'):
        response = httpx.post('https://api.stripe.com/v1/webhook_endpoints', auth=(stripe_key, ''),
            data={'url': 'https://' + HOST + '/api/v1/webhooks/stripe',
                  'enabled_events[0]': 'checkout.session.completed',
                  'enabled_events[1]': 'checkout.session.async_payment_succeeded',
                  'description': 'VideoTranslator Azure Japan sandbox validation'},
            headers={'Idempotency-Key': 'videotranslator-azure-jpe-sandbox-webhook-20260923'}, timeout=30)
        response.raise_for_status()
        webhook = response.json()
        if webhook.get('livemode') is not False:
            raise ValueError('Refusing a non-sandbox webhook')
        state.update(stripe_webhook_id=webhook['id'], stripe_webhook_secret=webhook['secret'])
    state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
    raw = subprocess.check_output(['docker', 'inspect', 'videotranslator-worker-1'], text=True)
    runtime = dict(item.split('=', 1) for item in json.loads(raw)[0]['Config']['Env'] if '=' in item)
    provider_names = ('ELEVENLABS_API_KEY', 'DEEPSEEK_API_KEY', 'TRANSCRIPTION_PROVIDER', 'SCRIBE_MODEL',
                      'DEEPSEEK_MODEL', 'DEEPSEEK_BASE_URL', 'SEPARATION_PROVIDER', 'DEMUCS_MODEL',
                      'DEMUCS_SEGMENT_SECONDS', 'TTS_MAX_TEXT_TOKENS')
    providers = {name: runtime[name] for name in provider_names if name in runtime}
    if providers.get('TRANSCRIPTION_PROVIDER') != 'scribe' or not providers.get('ELEVENLABS_API_KEY') or not providers.get('DEEPSEEK_API_KEY'):
        raise ValueError('Stable runtime provider configuration is missing')
    write_env(PRIVATE / 'engine.env', providers | {
        'DATABASE_URL': f"postgresql+psycopg://engineadmin:{state['postgres_password']}@postgres:5432/videotranslator?sslmode=require",
        'ENGINE_CONTROL_TOKEN': state['engine_token']})
    write_env(PRIVATE / 'postgres.env', {'POSTGRES_USER': 'engineadmin', 'POSTGRES_DB': 'videotranslator',
                                       'POSTGRES_PASSWORD': state['postgres_password']})
    write_env(PRIVATE / 'studio.env', {name: local.get(name, '') for name in (
        'FIREBASE_API_KEY', 'FIREBASE_AUTH_DOMAIN', 'FIREBASE_PROJECT_ID', 'FIREBASE_APP_ID',
        'DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL', 'CHAT_MODEL')} | {
        'AUTH_MODE': 'firebase', 'PUBLIC_APP_URL': 'https://' + HOST,
        'STRIPE_SANDBOX_SECRET_KEY': stripe_key, 'STRIPE_SANDBOX_WEBHOOK_SECRET': state['stripe_webhook_secret'],
        'AZURE_STORAGE_ACCOUNT_URL': 'https://vtranslatorjpe43892.blob.core.windows.net',
        'AZURE_STORAGE_CONTAINER': 'uploads', 'AZURE_RESULTS_CONTAINER': 'results',
        'ENGINE_CONTROL_TOKEN': state['engine_token']})
    write_env(PRIVATE / '.env', {
        'CONTROL_IMAGE': ACR + '/videotranslator/control@sha256:c1671fc9086a7880cc41fb616357ff31cfcdd71169e8a59041eb77a845d6aeb4',
        'ENGINE_IMAGE': ACR + '/videotranslator/engine@sha256:3b6bb4134f94fd9cf974c2be207426a72adefc5ecdb8cca7e2a4156fd7dbbc74',
        'CPU_PRIVATE_IP': '10.0.2.4', 'PUBLIC_HOST': HOST})
    # These values are passed only to the GPU app secret store, never command logs.
    (PRIVATE / 'gpu-secrets.json').write_text(json.dumps({'properties': {'configuration': {'secrets': [
        {'name': 'worker-database-url', 'value': f"postgresql+psycopg://gpuworker:{state['worker_password']}@10.0.2.4:5432/videotranslator?sslmode=require"},
        {'name': 'scaler-connection', 'value': f"host=10.0.2.4 port=5432 user=gpu_scaler password={state['scaler_password']} dbname=videotranslator sslmode=require"},
        {'name': 'elevenlabs-api-key', 'value': providers['ELEVENLABS_API_KEY']},
        {'name': 'deepseek-api-key', 'value': providers['DEEPSEEK_API_KEY']}
    ]}}}), encoding='utf-8')
    print('Sandbox cloud files prepared; no live Stripe keys or local databases copied.')


if __name__ == '__main__':
    main()
