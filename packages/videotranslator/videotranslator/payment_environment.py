"""Select an entire payment environment, identically on local and cloud hosts."""
from dataclasses import dataclass, field
from pathlib import Path

MODES = ('sandbox', 'live')


def profiled(env):
    return any(env.get(f'STRIPE_{mode.upper()}_SECRET_KEY') for mode in MODES)


@dataclass(frozen=True)
class PaymentEnvironment:
    mode: str
    key: str = field(repr=False)
    webhook_secret: str = field(repr=False)
    store_path: str
    queue_path: str


def select_environment(env, mode=None):
    mode = mode or env.get('PAYMENT_MODE', 'disabled')
    if mode not in MODES:
        raise ValueError('Payment environment must be sandbox or live')
    prefix = mode.upper()
    if profiled(env):
        key = env.get(f'STRIPE_{prefix}_SECRET_KEY', '') or ''
        hook = env.get(f'STRIPE_{prefix}_WEBHOOK_SECRET', '') or ''
        paths = {kind: [env.get(f'PAYMENT_{m.upper()}_{kind}', '') for m in MODES]
                 for kind in ('STORE_PATH', 'QUEUE_PATH')}
        for kind, pair in paths.items():
            if any(not p or p == ':memory:' for p in pair):
                raise ValueError(f'Both PAYMENT_SANDBOX_{kind} and PAYMENT_LIVE_{kind} are required')
            if Path(pair[0]).resolve() == Path(pair[1]).resolve():
                raise ValueError(f'Sandbox and live {kind} must be separate')
        if {Path(p).resolve() for p in paths['STORE_PATH']} & {Path(p).resolve() for p in paths['QUEUE_PATH']}:
            raise ValueError('Payment stores and inspection queues must be separate')
        store = env[f'PAYMENT_{prefix}_STORE_PATH']
        queue = env[f'PAYMENT_{prefix}_QUEUE_PATH']
    else:
        key = env.get('STRIPE_SECRET_KEY', '') or ''
        hook = env.get('STRIPE_WEBHOOK_SECRET', '') or ''
        store = env.get('STORE_PATH', './vt-data/store.db')
        queue = env.get('LOCAL_QUEUE_DB', './vt-data/scheduler.db')
    allowed = ('sk_live_', 'rk_live_') if mode == 'live' else ('sk_test_', 'rk_test_')
    if not key.startswith(allowed):
        raise ValueError(f'Missing or mismatched Stripe key for {mode}')
    if not hook.startswith('whsec_'):
        raise ValueError(f'Missing Stripe webhook signing secret for {mode}')
    return PaymentEnvironment(mode, key, hook, store, queue)


def bind_store(store, mode):
    """Prevent future configuration changes from reinterpreting an existing wallet."""
    from .domain.models import PaymentEnvironmentRecord, Payment
    with store.transaction() as tx:
        record = tx.get(PaymentEnvironmentRecord, 'stripe')
        if record:
            if record.mode != mode:
                raise ValueError('Payment database belongs to another environment; use its separate database')
            return
        expected = 'cs_live_' if mode == 'live' else 'cs_test_'
        for payment in tx.query(Payment):
            session = payment.provider_order_id or ''
            if payment.provider == 'stripe' and session.startswith('cs_') and not session.startswith(expected):
                raise ValueError('Existing Stripe payments belong to another environment')
        tx.insert(PaymentEnvironmentRecord(mode=mode), 'stripe')
