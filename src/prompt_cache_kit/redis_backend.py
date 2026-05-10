from __future__ import annotations

import hmac
import hashlib
import pickle  # nosec B403
import time
from typing import Any, Optional, Union

from .backend import CacheStats


class RedisCacheBackend:
    """Redis backend for response caching.

    Values are pickled so arbitrary Python model responses can be cached. Use a
    dedicated Redis database or namespace when caching sensitive LLM outputs.
    """

    def __init__(
        self,
        *,
        url: str = "redis://localhost:6379/0",
        namespace: str = "prompt-cache-kit",
        client: Optional[Any] = None,
        signing_key: Optional[Union[bytes, str]] = None,
    ) -> None:
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise ImportError("Install redis with `pip install prompt-cache-kit[redis]`.") from exc
            client = redis.Redis.from_url(url)
        self.client = client
        self.namespace = namespace
        self._signing_key = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
        self._hits = 0
        self._misses = 0
        self._sets = 0

    def get(self, key: str) -> Any:
        raw = self.client.get(self._key(key))
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        payload = bytes(raw)
        if self._signing_key:
            payload = _verify_and_strip_signature(payload, self._signing_key)
        try:
            return pickle.loads(payload)  # nosec B301
        except Exception:
            # Treat corrupted/untrusted payloads as cache misses.
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if self._signing_key:
            payload = _add_signature(payload, self._signing_key)
        redis_key = self._key(key)
        if ttl_seconds is None:
            self.client.set(redis_key, payload)
        else:
            self.client.setex(redis_key, int(ttl_seconds), payload)
        self._sets += 1

    def clear(self, namespace: Optional[str] = None) -> None:
        prefix = self._key(f"{namespace or ''}") if namespace else f"{self.namespace}:"
        cursor = 0
        pattern = f"{prefix}*"
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                self.client.delete(*keys)
            if cursor == 0:
                break

    def stats(self) -> CacheStats:
        return CacheStats(hits=self._hits, misses=self._misses, sets=self._sets, size=self._size())

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def _size(self) -> int:
        cursor = 0
        total = 0
        pattern = f"{self.namespace}:*"
        deadline = time.time() + 0.1
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=500)
            total += len(keys)
            if cursor == 0 or time.time() > deadline:
                return total


_SIG_PREFIX = b"pck1:"


def _add_signature(payload: bytes, key: bytes) -> bytes:
    mac = hmac.new(key, payload, hashlib.sha256).digest()
    return _SIG_PREFIX + mac + payload


def _verify_and_strip_signature(blob: bytes, key: bytes) -> bytes:
    if not blob.startswith(_SIG_PREFIX) or len(blob) < len(_SIG_PREFIX) + 32:
        raise ValueError("missing cache signature")
    mac = blob[len(_SIG_PREFIX) : len(_SIG_PREFIX) + 32]
    payload = blob[len(_SIG_PREFIX) + 32 :]
    expected = hmac.new(key, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        raise ValueError("invalid cache signature")
    return payload
