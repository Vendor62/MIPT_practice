from __future__ import annotations

import asyncio
import json
import os
import ssl
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote, urlparse

import structlog

from app.nlu_router import PendingActionPayload


log = structlog.get_logger()

PENDING_ACTION_REDIS_URL = os.getenv("NLU_PENDING_ACTION_REDIS_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
PENDING_ACTION_TTL_SECONDS = int(os.getenv("NLU_PENDING_ACTION_TTL_SECONDS", "900"))
PENDING_ACTION_KEY_PREFIX = os.getenv("NLU_PENDING_ACTION_KEY_PREFIX", "nlu:pending")


class PendingActionStoreError(RuntimeError):
    pass


def pending_action_ttl_seconds() -> int:
    return max(60, int(PENDING_ACTION_TTL_SECONDS))


def _pending_action_key(telegram_id: int) -> str:
    return f"{PENDING_ACTION_KEY_PREFIX}:{int(telegram_id)}"


def _encode_command(*parts: str | bytes | int) -> bytes:
    encoded_parts: list[bytes] = []
    for part in parts:
        if isinstance(part, bytes):
            payload = part
        else:
            payload = str(part).encode("utf-8")
        encoded_parts.append(b"$" + str(len(payload)).encode("ascii") + b"\r\n" + payload + b"\r\n")
    return b"*" + str(len(encoded_parts)).encode("ascii") + b"\r\n" + b"".join(encoded_parts)


async def _read_resp(reader: asyncio.StreamReader) -> Any:
    prefix = await reader.readexactly(1)
    if prefix == b"+":
        return (await reader.readline()).rstrip(b"\r\n").decode("utf-8")
    if prefix == b"-":
        error = (await reader.readline()).rstrip(b"\r\n").decode("utf-8", errors="replace")
        raise PendingActionStoreError(error)
    if prefix == b":":
        return int((await reader.readline()).rstrip(b"\r\n"))
    if prefix == b"$":
        length = int((await reader.readline()).rstrip(b"\r\n"))
        if length == -1:
            return None
        payload = await reader.readexactly(length)
        await reader.readexactly(2)
        return payload.decode("utf-8")
    if prefix == b"*":
        size = int((await reader.readline()).rstrip(b"\r\n"))
        if size == -1:
            return None
        return [await _read_resp(reader) for _ in range(size)]
    raise PendingActionStoreError(f"Unsupported RESP prefix: {prefix!r}")


class RedisPendingActionStore:
    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or PENDING_ACTION_REDIS_URL

    async def _execute(self, *commands: tuple[str | bytes | int, ...]) -> list[Any]:
        parsed = urlparse(self.redis_url)
        if parsed.scheme not in {"redis", "rediss"}:
            raise PendingActionStoreError(f"Unsupported Redis URL scheme: {parsed.scheme or 'empty'}")

        host = parsed.hostname or "localhost"
        port = int(parsed.port or 6379)
        database = int((parsed.path or "/0").lstrip("/") or "0")
        password = unquote(parsed.password) if parsed.password else None
        username = unquote(parsed.username) if parsed.username else None
        ssl_context = ssl.create_default_context() if parsed.scheme == "rediss" else None

        reader, writer = await asyncio.open_connection(host=host, port=port, ssl=ssl_context)
        try:
            if password:
                auth_command: tuple[str | bytes | int, ...]
                if username:
                    auth_command = ("AUTH", username, password)
                else:
                    auth_command = ("AUTH", password)
                writer.write(_encode_command(*auth_command))
                await writer.drain()
                await _read_resp(reader)

            if database:
                writer.write(_encode_command("SELECT", database))
                await writer.drain()
                await _read_resp(reader)

            results: list[Any] = []
            for command in commands:
                writer.write(_encode_command(*command))
                await writer.drain()
                results.append(await _read_resp(reader))
            return results
        finally:
            writer.close()
            await writer.wait_closed()

    async def save(self, payload: PendingActionPayload, *, ttl_seconds: int | None = None) -> bool:
        ttl = max(1, int(ttl_seconds or pending_action_ttl_seconds()))
        key = _pending_action_key(payload.telegram_id)
        try:
            results = await self._execute(("SETEX", key, ttl, json.dumps(payload.to_dict(), ensure_ascii=False)))
            return str(results[0] or "").upper() == "OK"
        except Exception as exc:
            log.warning("pending_action.store_failed", telegram_id=payload.telegram_id, error=repr(exc))
            return False

    async def load(self, telegram_id: int) -> PendingActionPayload | None:
        key = _pending_action_key(telegram_id)
        try:
            results = await self._execute(("GET", key))
        except Exception as exc:
            log.warning("pending_action.load_failed", telegram_id=telegram_id, error=repr(exc))
            return None
        raw = results[0]
        if not raw:
            return None
        try:
            return PendingActionPayload.from_dict(json.loads(raw))
        except Exception as exc:
            log.warning("pending_action.invalid_payload", telegram_id=telegram_id, error=repr(exc))
            return None

    async def delete(self, telegram_id: int) -> None:
        key = _pending_action_key(telegram_id)
        try:
            await self._execute(("DEL", key))
        except Exception as exc:
            log.warning("pending_action.delete_failed", telegram_id=telegram_id, error=repr(exc))


class InMemoryPendingActionStore:
    def __init__(self):
        self._items: dict[int, PendingActionPayload] = {}

    @staticmethod
    def _is_expired(payload: PendingActionPayload) -> bool:
        try:
            expires_at = datetime.fromisoformat(payload.expires_at)
        except Exception:
            return True
        return expires_at <= datetime.now(timezone.utc)

    async def save(self, payload: PendingActionPayload, *, ttl_seconds: int | None = None) -> bool:
        self._items[int(payload.telegram_id)] = payload
        return True

    async def load(self, telegram_id: int) -> PendingActionPayload | None:
        payload = self._items.get(int(telegram_id))
        if payload is None:
            return None
        if self._is_expired(payload):
            self._items.pop(int(telegram_id), None)
            return None
        return payload

    async def delete(self, telegram_id: int) -> None:
        self._items.pop(int(telegram_id), None)


class FallbackPendingActionStore:
    def __init__(
        self,
        primary: RedisPendingActionStore | None = None,
        fallback: InMemoryPendingActionStore | None = None,
    ):
        self.primary = primary or RedisPendingActionStore()
        self.fallback = fallback or InMemoryPendingActionStore()

    async def save(self, payload: PendingActionPayload, *, ttl_seconds: int | None = None) -> bool:
        fallback_saved = await self.fallback.save(payload, ttl_seconds=ttl_seconds)
        primary_saved = await self.primary.save(payload, ttl_seconds=ttl_seconds)
        return bool(primary_saved or fallback_saved)

    async def load(self, telegram_id: int) -> PendingActionPayload | None:
        payload = await self.primary.load(telegram_id)
        if payload is not None:
            await self.fallback.save(payload)
            return payload
        return await self.fallback.load(telegram_id)

    async def delete(self, telegram_id: int) -> None:
        await self.primary.delete(telegram_id)
        await self.fallback.delete(telegram_id)


_store: FallbackPendingActionStore | None = None


def get_pending_action_store() -> FallbackPendingActionStore:
    global _store
    if _store is None:
        _store = FallbackPendingActionStore()
    return _store
