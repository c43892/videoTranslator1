"""Local switch preflight and atomic .env update; orchestration is in stripe.ps1."""
import argparse
import json
from pathlib import Path
import os
import sqlite3
import subprocess
import sys
import tempfile
import httpx

from dotenv import dotenv_values, set_key
from videotranslator.payment_environment import select_environment, profiled

ROOT = Path(__file__).resolve().parents[1]


def ensure_idle(config):
    current = select_environment(config)
    database = Path(current.store_path)
    if not database.exists():
        raise ValueError('Current payment database is missing; refusing an unchecked switch')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        jobs = [json.loads(row[0]) for row in conn.execute("SELECT data FROM docs WHERE collection='jobs'")]
        active = [j for j in jobs if j.get('status') in {
            'uploaded', 'inspecting', 'queued', 'submitting', 'provisioning', 'running', 'cancelling', 'awaiting_capacity'}]
        if active:
            raise ValueError(f'{len(active)} unfinished task(s); wait for completion before switching')
        payments = [json.loads(row[0]) for row in conn.execute("SELECT data FROM docs WHERE collection='payments'")]
        for payment in payments:
            if payment.get('status') != 'pending':
                continue
            session = payment.get('provider_order_id') or ''
            if payment.get('provider') != 'stripe' or not session.startswith('cs_'):
                raise ValueError('Unresolved top-up; finish payment verification before switching')
            try:
                response = httpx.get(f'https://api.stripe.com/v1/checkout/sessions/{session}',
                                     auth=(current.key, ''), timeout=15)
                data = response.json()
            except (httpx.HTTPError, ValueError):
                raise ValueError('Cannot verify pending top-ups; retry before switching') from None
            if (response.status_code != 200 or data.get('id') != session
                    or data.get('livemode') is not (current.mode == 'live')
                    or data.get('client_reference_id') != payment.get('payment_id')
                    or data.get('status') != 'expired' or data.get('payment_status') != 'unpaid'):
                raise ValueError('Unresolved top-up; finish payment verification or wait for checkout expiry before switching')


def check(config, mode):
    if not profiled(config):
        raise ValueError('Set up STRIPE_SANDBOX_* and STRIPE_LIVE_* first; see docs/STRIPE_ENVIRONMENTS.md')
    target = select_environment(config, mode)
    # This command manages only this repository's local Compose deployment.
    from urllib.parse import urlsplit
    if urlsplit(config.get('PUBLIC_APP_URL', '')).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Local switch requires localhost; cloud deployments select PAYMENT_MODE via deployment settings')
    if any(config.get(k) for k in ('PAYPAL_CLIENT_ID', 'PAYPAL_CLIENT_SECRET', 'PAYPAL_WEBHOOK_ID')):
        raise ValueError('This Stripe switch does not switch PayPal credentials; remove or separately configure PayPal first')
    if config.get('PAYMENT_MODE') != mode:
        ensure_idle(config)
    return target


def write_mode(env_file, mode):
    config = dotenv_values(env_file)
    target = check(config, mode)
    fd, temp = tempfile.mkstemp(prefix='.env.stripe-prepare-', dir=env_file.parent)
    os.close(fd)
    temp = Path(temp)
    try:
        temp.write_bytes(env_file.read_bytes())
        set_key(temp, 'PAYMENT_MODE', target.mode)
        # Verify the target's real CLI authorization before replacing the active file.
        result = subprocess.run([sys.executable, str(ROOT / 'deploy/stripe-listen.py'),
                                 '--env-file', str(temp), '--prepare-only'], capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise ValueError('Target Stripe listener authorization failed; active configuration was not changed')
        os.replace(temp, env_file)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'check', 'set'])
    parser.add_argument('mode', nargs='?', choices=['sandbox', 'live'])
    args = parser.parse_args()
    os.chdir(ROOT)
    config = dotenv_values(ROOT / '.env')
    if args.action == 'status':
        active = select_environment(config)
        print(f'Configured Stripe mode: {active.mode}')
        print(f'Wallet/history database: {active.store_path}')
        for mode in ('sandbox', 'live'):
            try:
                select_environment(config, mode)
                print(f'{mode}: configuration ready')
            except ValueError:
                print(f'{mode}: configuration incomplete')
    elif not args.mode:
        parser.error('mode is required for check/set')
    elif args.action == 'check':
        check(config, args.mode)
        print(f'Preflight passed for {args.mode}')
    else:
        write_mode(ROOT / '.env', args.mode)
        print(f'Configured Stripe mode: {args.mode}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
