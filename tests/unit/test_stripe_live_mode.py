"""Live wallets must never accept sandbox checkout responses or signed events."""
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import time

import httpx
import pytest

from videotranslator.adapters.payments import StripePaymentGateway
from videotranslator.domain.enums import BackendError
from videotranslator.domain.models import LedgerEntry, PaymentEvent, User


@pytest.mark.parametrize('key', ['sk_live_example', 'rk_live_example'])
@pytest.mark.parametrize('livemode', [False, None])
def test_live_wallet_rejects_even_validly_signed_nonlive_event(container, user, key, livemode):
    session = container.billing.create_session(user.user_id, package_id='points_10_v1', provider='stripe')
    container.gateways['stripe'] = StripePaymentGateway(key, 'hook')
    event = {'id': 'evt_example', 'livemode': livemode, 'type': 'checkout.session.completed',
             'data': {'object': {'client_reference_id': session.payment_id, 'payment_status': 'paid',
                                'amount_total': 1000, 'currency': 'usd', 'payment_intent': 'pi_example'}}}
    body = json.dumps(event).encode()
    timestamp = str(int(time.time()))
    signature = hmac.new(b'hook', timestamp.encode() + b'.' + body, hashlib.sha256).hexdigest()
    with pytest.raises(BackendError, match='mode mismatch'):
        container.billing.record_webhook('stripe', {'stripe-signature': f't={timestamp},v1={signature}'}, body)
    with container.store.transaction() as tx:
        assert tx.get(User, user.user_id).point_balance_units == 0
        assert tx.query(LedgerEntry) == []
        assert tx.query(PaymentEvent) == []


@pytest.mark.parametrize('key,expected', [('rk_live_example', True), ('sk_test_example', False)])
@pytest.mark.parametrize('matching', [True, False])
def test_checkout_and_lookup_enforce_key_mode(container, user, monkeypatch, key, expected, matching):
    session = container.billing.create_session(user.user_id, package_id='points_10_v1', provider='stripe')
    request = container.gateways['stripe']._sessions[session.payment_id]
    gateway = StripePaymentGateway(key, 'hook')
    data = dict(id='cs_example', url='https://checkout.stripe.com/example', mode='payment',
                livemode=expected if matching else not expected, status='complete', payment_status='paid',
                amount_total=1000, amount=1000, currency='usd', client_reference_id=session.payment_id)
    monkeypatch.setattr(httpx, 'post', lambda *a, **kw: httpx.Response(200, json=data))
    monkeypatch.setattr(httpx, 'get', lambda *a, **kw: httpx.Response(200, json=data))
    operations = [lambda: gateway.create_session(request), lambda: gateway.get_payment('cs_example'),
                  lambda: gateway.get_payment('pi_example')]
    for operation in operations:
        if matching:
            operation()
        else:
            with pytest.raises(BackendError, match='mode mismatch'):
                operation()


def listener_module():
    path = Path(__file__).resolve().parents[2] / 'deploy' / 'stripe-listen.py'
    spec = importlib.util.spec_from_file_location('stripe_listener', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode,key,live', [('live','rk_live_example',True), ('sandbox','sk_test_example',False)])
def test_local_listener_explicitly_selects_mode(mode, key, live):
    args = listener_module().listener_args({'PAYMENT_MODE': mode, 'STRIPE_SECRET_KEY': key,
                                          'PUBLIC_APP_URL': 'http://localhost:8090'}, 'stripe')
    assert ('--live' in args) is live
    assert args[-1] == 'http://localhost:8090/api/v1/webhooks/stripe'
    assert key not in ' '.join(args)


@pytest.mark.parametrize('override', [dict(PAYMENT_MODE='sandbox'), dict(STRIPE_SECRET_KEY='sk_test_example'),
    dict(PUBLIC_APP_URL='https://external.example'), dict(PUBLIC_APP_URL='http://localhost:8090/other'),
    dict(PUBLIC_APP_URL='http://user@localhost:8090')])
def test_local_listener_rejects_wrong_mode_or_destination(override):
    config = {'PAYMENT_MODE':'live', 'STRIPE_SECRET_KEY':'rk_live_example', 'PUBLIC_APP_URL':'http://localhost:8090'}
    with pytest.raises(ValueError):
        listener_module().listener_args({**config, **override}, 'stripe')
