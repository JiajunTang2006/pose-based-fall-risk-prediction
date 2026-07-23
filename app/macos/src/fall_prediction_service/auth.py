"""
Bearer-token authentication for the FallGuard AI Service.

All ``/api/v1/`` routes must present a valid token.  The comparison uses
``hmac.compare_digest`` to resist timing attacks.
"""

from __future__ import annotations

import hmac

BEARER_PREFIX = "Bearer "


def validate_token(header: str | None, expected_token: str) -> bool:
    """Return True if *header* carries the expected Bearer token.

    Uses constant-time comparison via ``hmac.compare_digest``.
    """
    if not header or not header.startswith(BEARER_PREFIX):
        return False
    presented = header[len(BEARER_PREFIX):]
    return hmac.compare_digest(presented, expected_token)
