"""Local filesystem object storage (§13.1 local profiles)."""

from __future__ import annotations

import hmac
from pathlib import Path

from .fake import FakeObjectStorage


class LocalObjectStorage(FakeObjectStorage):
    """Same filesystem semantics as the fake, with signed local URLs.

    URLs carry an HMAC expiry token; the API's local-download endpoint
    validates them, mirroring the short-lived SAS contract (§3.4).
    """

    def __init__(self, root: str | Path, *, secret: str | None = None):
        super().__init__(root)
        import secrets
        self._secret = (secret or secrets.token_urlsafe(32)).encode()

    def _token(self, key: str, expires: int) -> str:
        import hashlib

        msg = f"{key}:{expires}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()[:32]

    def create_upload_url(self, object_key: str, expires_in: int) -> str:
        import time

        expires = int(time.time()) + expires_in
        return f"local://upload/{object_key}?expires={expires}&token={self._token(object_key, expires)}"

    def create_download_url(self, object_key: str, expires_in: int) -> str:
        import time

        expires = int(time.time()) + expires_in
        return f"local://download/{object_key}?expires={expires}&token={self._token(object_key, expires)}"

    def verify_url(self, object_key: str, expires: int, token: str) -> bool:
        import time

        if int(time.time()) > expires:
            return False
        expected = self._token(object_key, expires)
        return hmac.compare_digest(expected, token)
