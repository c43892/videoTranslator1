"""Forward Stripe test webhooks locally without printing credentials.

Run with the project's Python environment after setting STRIPE_SECRET_KEY.
Updates STRIPE_WEBHOOK_SECRET in .env; restart the API if that value changes.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlsplit

from dotenv import dotenv_values, set_key


def main():
    env_file = Path(__file__).resolve().parents[1] / '.env'
    config = dotenv_values(env_file)
    key = config.get('STRIPE_SECRET_KEY', '')
    if config.get('PAYMENT_MODE') != 'sandbox' or not key.startswith(('sk_test_', 'rk_test_')):
        raise SystemExit('Requires PAYMENT_MODE=sandbox and a Stripe test key in .env.')
    stripe = shutil.which('stripe')
    if not stripe:
        raise SystemExit('Install the official Stripe CLI first.')
    base = config.get('PUBLIC_APP_URL', 'http://localhost:8090').rstrip('/')
    url = urlsplit(base)
    if url.scheme not in {'http', 'https'} or url.hostname not in {'localhost', '127.0.0.1', '::1'} or url.username or url.password or url.query or url.fragment or url.path:
        raise SystemExit('PUBLIC_APP_URL must be a local HTTP(S) origin.')
    env = {**os.environ, 'STRIPE_API_KEY': key}
    args = [stripe, 'listen', '--skip-update', '--events',
            'checkout.session.completed,checkout.session.async_payment_succeeded',
            '--forward-to', base + '/api/v1/webhooks/stripe']
    secret_result = subprocess.run(args + ['--print-secret'], env=env,
                                   capture_output=True, text=True, timeout=30)
    match = re.search(r'whsec_[A-Za-z0-9]+', secret_result.stdout)
    if secret_result.returncode or not match:
        raise SystemExit('Could not obtain the signing secret. Check Stripe CLI authorization.')
    if config.get('STRIPE_WEBHOOK_SECRET') != match.group():
        set_key(env_file, 'STRIPE_WEBHOOK_SECRET', match.group())
        print('Webhook signing secret updated in .env; restart the API.', flush=True)
    print('Forwarding Stripe test events to ' + base + '/api/v1/webhooks/stripe', flush=True)
    process = subprocess.Popen(args, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    try:
        for line in process.stdout:
            safe = re.sub(r'(?:whsec_|[sr]k_(?:test|live)_)[A-Za-z0-9_]+', '[redacted]', line)
            print(safe, end='', flush=True)
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == '__main__':
    raise SystemExit(main())
