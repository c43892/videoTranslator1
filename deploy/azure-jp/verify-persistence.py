"""Run inside the web container; inspect only dedicated validation records."""
import json
import os
import httpx

from videotranslator.bootstrap import build_container
from videotranslator.domain.models import LedgerEntry, Payment, User

container = build_container()
assert container.settings.payment_mode == 'sandbox'
with container.store.transaction() as tx:
    user = tx.get(User, 'azure-validation-20260923')
    entries = tx.query(LedgerEntry, where=('user_id', '==', user.user_id))
    payments = tx.query(Payment, where=('user_id', '==', user.user_id))
assert len(payments) == 1 and payments[0].status == 'succeeded'
print(json.dumps({'payment_count': len(payments), 'balance': user.point_balance_units,
    'ledger_entries': len(entries), 'sandbox': container.settings.payment_mode == 'sandbox'}))
url = os.environ['ENGINE_CONTROL_URL'] + '/health'
assert httpx.get(url, timeout=10).status_code == 401
assert httpx.get(url, headers={'Authorization':'Bearer '+os.environ['ENGINE_CONTROL_TOKEN']}, timeout=10).json() == {'ready':True}
print('Private control requires its bearer token; health does not need a GPU.')
