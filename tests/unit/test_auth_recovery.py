"""Auth outages must not masquerade as invalid credentials or bypass verification."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
import time

import pytest
from fastapi.testclient import TestClient

from videotranslator.adapters import auth
from videotranslator.api.app import create_app
from videotranslator.domain.enums import BackendError, FailureClass


class InvalidToken(Exception): pass
class RevokedToken(Exception): pass
class DisabledUser(Exception): pass
class MissingUser(Exception): pass


def verifier(outcomes):
    instance = auth.FirebaseIdentityVerifier.__new__(auth.FirebaseIdentityVerifier)
    instance._auth = SimpleNamespace(
        verify_id_token=Mock(side_effect=outcomes), InvalidIdTokenError=InvalidToken,
        RevokedIdTokenError=RevokedToken, UserDisabledError=DisabledUser,
        UserNotFoundError=MissingUser,
    )
    return instance


def test_transient_verification_recovers_without_skipping_revocation(monkeypatch):
    monkeypatch.setattr(auth.time, 'sleep', lambda _: None)
    check = verifier([ConnectionError('private SDK response'),
                      {'uid': 'u1', 'email_verified': True}])
    assert check.verify('test-token').uid == 'u1'
    assert check._auth.verify_id_token.call_count == 2
    assert all(call.kwargs == {'check_revoked': True}
               for call in check._auth.verify_id_token.call_args_list)


@pytest.mark.parametrize('error', [InvalidToken, RevokedToken, DisabledUser, MissingUser, ValueError])
def test_invalid_or_revoked_credentials_stay_rejected(error):
    check = verifier([error('sensitive-value')])
    with pytest.raises(BackendError) as caught:
        check.verify('test-token')
    assert caught.value.failure_class == FailureClass.PERMANENT
    assert check._auth.verify_id_token.call_count == 1


def test_outage_exhaustion_is_retryable_and_logs_no_secrets(monkeypatch, caplog):
    monkeypatch.setattr(auth.time, 'sleep', lambda _: None)
    check = verifier([ConnectionError('secret-credential')] * 2)
    with pytest.raises(BackendError) as caught:
        check.verify('secret-token')
    assert caught.value.failure_class == FailureClass.RETRYABLE
    assert 'secret-' not in caplog.text


def test_rejection_logs_only_a_fixed_reason_and_never_sdk_details(caplog):
    check = verifier([InvalidToken('Token used too early: private-claim secret-token')])
    with pytest.raises(BackendError):
        check.verify('secret-token')
    assert 'reason=issued_in_future' in caplog.text
    assert 'private-claim' not in caplog.text and 'secret-token' not in caplog.text


def test_parallel_first_requests_initialize_one_verifier(monkeypatch):
    built = []
    def build():
        built.append(True)
        time.sleep(.02)
        return SimpleNamespace(verify=lambda token: token)
    monkeypatch.setattr(auth, 'FirebaseIdentityVerifier', build)
    check = auth.LazyFirebaseIdentityVerifier()
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(check.verify, range(8))) == list(range(8))
    assert len(built) == 1


@pytest.mark.parametrize('failure,status,code', [
    (FailureClass.RETRYABLE, 503, 'auth_unavailable'),
    (FailureClass.PERMANENT, 401, 'unauthenticated'),
])
def test_api_distinguishes_auth_outage_from_rejection(container, failure, status, code):
    container.identity = SimpleNamespace(verify=Mock(side_effect=BackendError('hidden', failure_class=failure)))
    response = TestClient(create_app(container)).post('/api/v1/conversations',
        json={'locale':'en'}, headers={'Authorization':'Bearer test-token'})
    assert response.status_code == status
    assert response.json() == {'detail': {'code': code}}
    if status == 503:
        assert response.headers['retry-after'] == '1'
