from __future__ import annotations

import os


def _parse_allowed_telegram_ids(raw: str | None) -> set[int]:
    result: set[int] = set()
    for chunk in str(raw or "").replace(";", ",").split(","):
        value = chunk.strip()
        if not value:
            continue
        try:
            result.add(int(value))
        except ValueError:
            continue
    return result


def _parse_allowed_usernames(raw: str | None) -> set[str]:
    result: set[str] = set()
    for chunk in str(raw or "").replace(";", ",").split(","):
        value = chunk.strip().lstrip("@").lower()
        if value:
            result.add(value)
    return result


STORYTRACKING_ALLOWED_TELEGRAM_IDS = _parse_allowed_telegram_ids(
    os.getenv("STORYTRACKING_ALLOWED_TELEGRAM_IDS", "")
)
STORYTRACKING_ALLOWED_USERNAMES = _parse_allowed_usernames(
    os.getenv("STORYTRACKING_ALLOWED_USERNAMES", "")
)


def storytracking_rollout_restricted() -> bool:
    return False


def storytracking_allowed_for_user(
    user: object | None = None,
    *,
    telegram_id: int | None = None,
    username: str | None = None,
) -> bool:
    return True
