"""Identity verifier adapters (§6.1): Firebase, emulator, and fake."""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time

from ..domain.enums import BackendError, FailureClass
from ..ports import UserIdentity

logger = logging.getLogger(__name__)
_firebase_init_lock = threading.Lock()


def _rejection_reason(error):
    # Only fixed categories are logged. Never log the SDK message or the token.
    text = str(error).lower()
    for needle, reason in [('token used too early', 'issued_in_future'), ('token expired', 'expired'),
                           ('signature', 'signature'), ('audience', 'audience'), ('issuer', 'issuer')]:
        if needle in text:
            return reason
    return 'invalid'


class FirebaseIdentityVerifier:
    """Production verifier; firebase_admin is imported lazily."""

    def __init__(self):
        try:
            import firebase_admin
            from firebase_admin import auth as fb_auth
        except ImportError as exc:
            raise BackendError("firebase-admin is required for AUTH_BACKEND=firebase") from exc
        with _firebase_init_lock:
            if not firebase_admin._apps:
                options = {"projectId": os.environ["FIREBASE_PROJECT_ID"]} if os.environ.get("FIREBASE_PROJECT_ID") else None
                firebase_admin.initialize_app(options=options)
        self._auth = fb_auth

    def verify(self, bearer_token: str) -> UserIdentity:
        for attempt in range(2):
            try:
                claims = self._auth.verify_id_token(bearer_token, check_revoked=True)
                break
            except (self._auth.InvalidIdTokenError, self._auth.RevokedIdTokenError,
                    self._auth.UserDisabledError, self._auth.UserNotFoundError, ValueError) as exc:
                logger.warning("Firebase token rejected (%s, reason=%s)", type(exc).__name__, _rejection_reason(exc))
                raise BackendError("invalid firebase token", failure_class=FailureClass.PERMANENT) from exc
            except Exception as exc:
                # Network/certificate/IAM failures do not mean the user signed out.
                # Log types only: SDK messages may include credential material.
                logger.warning("Firebase verification unavailable (%s)", type(exc).__name__)
                if attempt:
                    raise BackendError("firebase verification unavailable") from exc
                time.sleep(0.2)
        return UserIdentity(
            uid=claims["uid"],
            email=claims.get("email", ""),
            email_verified=bool(claims.get("email_verified", False)),
            display_name=claims.get("name", ""),
            photo_url=claims.get("picture"),
        )


class LazyFirebaseIdentityVerifier:
    def __init__(self):
        self._delegate = None
        self._lock = threading.Lock()

    def verify(self, bearer_token: str) -> UserIdentity:
        if self._delegate is None:
            try:
                with self._lock:
                    if self._delegate is None:
                        self._delegate = FirebaseIdentityVerifier()
            except Exception as exc:
                logger.warning("Firebase initialization failed (%s)", type(exc).__name__)
                raise BackendError("firebase configuration unavailable") from exc
        return self._delegate.verify(bearer_token)


class FirebaseEmulatorIdentityVerifier:
    """Decodes emulator-issued JWTs without signature verification (dev only)."""

    def verify(self, bearer_token: str) -> UserIdentity:
        try:
            payload = bearer_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
        except Exception as exc:
            raise BackendError("malformed emulator token", failure_class=FailureClass.PERMANENT) from exc
        return UserIdentity(
            uid=claims.get("user_id") or claims.get("sub", ""),
            email=claims.get("email", ""),
            email_verified=bool(claims.get("email_verified", False)),
            display_name=claims.get("name", ""),
        )
