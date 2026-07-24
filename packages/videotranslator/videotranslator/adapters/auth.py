"""Identity verifier adapters (§6.1): Firebase, emulator, and fake."""

from __future__ import annotations

import base64
import json

from ..domain.enums import BackendError, FailureClass
from ..ports import UserIdentity


class FirebaseIdentityVerifier:
    """Production verifier; firebase_admin is imported lazily."""

    def __init__(self):
        try:
            import firebase_admin
            from firebase_admin import auth as fb_auth
        except ImportError as exc:
            raise BackendError("firebase-admin is required for AUTH_BACKEND=firebase") from exc
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        self._auth = fb_auth

    def verify(self, bearer_token: str) -> UserIdentity:
        try:
            claims = self._auth.verify_id_token(bearer_token)
        except Exception as exc:
            raise BackendError("invalid firebase token", failure_class=FailureClass.PERMANENT) from exc
        return UserIdentity(
            uid=claims["uid"],
            email=claims.get("email", ""),
            email_verified=bool(claims.get("email_verified", False)),
            display_name=claims.get("name", ""),
            photo_url=claims.get("picture"),
        )


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
