from __future__ import annotations

import hmac

BEARER_PREFIX = "Bearer "


def validate_token(header: str | None, expected_token: str) -> bool:
    if not header or not header.startswith(BEARER_PREFIX):
        return False
    presented = header[len(BEARER_PREFIX):]
    return hmac.compare_digest(presented, expected_token)
