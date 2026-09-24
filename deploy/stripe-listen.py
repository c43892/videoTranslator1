"""Forward the explicitly configured Stripe mode locally without printing secrets.

Run with the project's Python environment after setting STRIPE_SECRET_KEY.
Updates STRIPE_WEBHOOK_SECRET in .env; restart the API if that value changes.
"""

import os
import argparse
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlsplit

from dotenv import dotenv_values, set_key


def listener_args(config, stripe):
    from videotranslator.payment_environment import profiled, select_environment
    if profiled(config):
        selected = select_environment(config)
        config = {**config, 'STRIPE_SECRET_KEY': selected.key}
    mode = config.get('PAYMENT_MODE')
    prefixes = {'sandbox': ('sk_test_', 'rk_test_'), 'live': ('sk_live_', 'rk_live_')}
    if mode not in prefixes or not config.get('STRIPE_SECRET_KEY', '').startswith(prefixes[mode]):
        raise ValueError('PAYMENT_MODE and Stripe key must both be sandbox or both live.')
    base = config.get('PUBLIC_APP_URL', 'http://localhost:8090').rstrip('/')
    url = urlsplit(base)
    if url.scheme not in {'http', 'https'} or url.hostname not in {'localhost', '127.0.0.1', '::1'} or url.username or url.password or url.query or url.fragment or url.path:
        raise ValueError('PUBLIC_APP_URL must be a local HTTP(S) origin.')
    return [stripe, 'listen', '--skip-update', *(['--live'] if mode == 'live' else []), '--events',
            'checkout.session.completed,checkout.session.async_payment_succeeded',
            '--forward-to', base + '/api/v1/webhooks/stripe']


def prepare_listener(env_file):
    from videotranslator.payment_environment import profiled, select_environment
    config = dotenv_values(env_file)
    selected = select_environment(config) if profiled(config) else None
    key = selected.key if selected else config.get('STRIPE_SECRET_KEY', '')
    stripe = shutil.which('stripe')
    if not stripe:
        raise SystemExit('Install the official Stripe CLI first.')
    env = {**os.environ, 'STRIPE_API_KEY': key}
    args = listener_args(config, stripe)
    secret_result = subprocess.run(args + ['--print-secret'], env=env,
                                   capture_output=True, text=True, timeout=30)
    match = re.search(r'whsec_[A-Za-z0-9]+', secret_result.stdout)
    if secret_result.returncode or not match:
        raise SystemExit('Could not obtain the signing secret. Check Stripe CLI authorization.')
    field = f"STRIPE_{selected.mode.upper()}_WEBHOOK_SECRET" if selected else 'STRIPE_WEBHOOK_SECRET'
    if config.get(field) != match.group():
        set_key(env_file, field, match.group())
        print('Webhook signing secret updated in .env; restart the API.', flush=True)
    return config, args, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=Path(__file__).resolve().parents[1] / '.env')
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args()
    config, args, env = prepare_listener(options.env_file)
    if options.prepare_only:
        print('Stripe listener authorization verified; signing secret saved.')
        return 0
    print(f"Forwarding Stripe {config['PAYMENT_MODE']} events to {args[-1]}", flush=True)
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
