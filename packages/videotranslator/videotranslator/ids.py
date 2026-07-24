"""Server-side id generation (§11.11): lowercase, digits, ``_``/``-`` only."""

from __future__ import annotations

import secrets


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"
