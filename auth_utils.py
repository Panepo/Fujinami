"""Shared auth helpers for HTTP clients."""

from __future__ import annotations


def normalize_bearer_token(raw_token: str) -> str:
    """Normalize an auth token into an Authorization Bearer value.

    Returns an empty string when the input is empty.
    """
    token = raw_token.strip()
    if not token:
        return ""
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"
