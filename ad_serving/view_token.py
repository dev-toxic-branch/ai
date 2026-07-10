"""View Token Service - HMAC signed one-time-use tokens for fraud prevention.

Inspired by BetterAds ViewTokenService. When the embed widget is rendered,
a signed token is issued. A view-recording call presenting a valid token
proves it came from a genuine widget load and can't be replayed.
"""

import hashlib
import hmac
import os
import time
import uuid


class ViewTokenService:
    """Issues and validates short-lived, one-time-use signed view tokens."""

    TOKEN_TTL_SECONDS = 120  # 2 minutes

    def __init__(self, secret: str = None):
        self.secret = secret or os.getenv("VIEW_TOKEN_SECRET", os.getenv("JWT_SECRET", "change-me-in-production"))

    def issue_token(self, ad_id: int) -> str:
        """Issue a signed token for a specific ad."""
        expiry = int(time.time()) + self.TOKEN_TTL_SECONDS
        nonce = uuid.uuid4().hex
        payload = f"{ad_id}.{expiry}.{nonce}"
        signature = self._sign(payload)
        return f"{payload}.{signature}"

    def validate_and_consume(self, token: str, ad_id: int, cache=None) -> bool:
        """Validate a token and ensure it's used only once.

        Returns True if the token is valid, belongs to this ad_id,
        hasn't expired, and hasn't been used before.
        """
        if not token or not token.strip():
            return False

        parts = token.split(".", 4)
        if len(parts) != 4:
            return False

        payload = f"{parts[0]}.{parts[1]}.{parts[2]}"
        expected_sig = self._sign(payload)

        if not hmac.compare_digest(expected_sig, parts[3]):
            return False

        try:
            token_ad_id = int(parts[0])
            expiry = int(parts[1])
        except ValueError:
            return False

        if token_ad_id != ad_id:
            return False
        if time.time() > expiry:
            return False

        # Consume the token (one-time use)
        nonce = parts[2]
        used_key = f"view-token-used:{nonce}"
        if cache:
            # Redis-backed: SETNX with TTL
            existing = cache.get(used_key)
            if existing:
                return False
            cache.set(used_key, "1", ttl=self.TOKEN_TTL_SECONDS)
            return True
        else:
            # In-memory fallback (not distributed-safe)
            if not hasattr(self, "_used_tokens"):
                self._used_tokens = set()
            if nonce in self._used_tokens:
                return False
            self._used_tokens.add(nonce)
            return True

    def _sign(self, payload: str) -> str:
        return hmac.new(
            self.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
