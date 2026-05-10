from __future__ import annotations

import time
from typing import Any, Optional, Union

from ..serializers import PickleSerializer, SignedSerializer
from ..types import CacheStats, Serializer


class RedisCacheBackend:
    """Redis backend for response caching.

    The serializer is injectable to allow alternative encodings without changing
    backend logic (DIP/OCP). By default, Prompt Cache Kit uses pickle.
    """

    def __init__(
        self,
        *,
        url: str = "redis://localhost:6379/0",
        namespace: str = "prompt-cache-kit",
        client: Optional[Any] = None,
        serializer: Optional[Serializer[object, object]] = None,
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

        base_serializer: Serializer[object, object] = serializer or PickleSerializer()
        if signing_key is not None:
            key = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
            base_serializer = SignedSerializer(inner=base_serializer, signing_key=key)
        self._serializer = base_serializer

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
        try:
            return self._serializer.loads(payload)
        except Exception:
            # Treat corrupted/untrusted payloads as cache misses.
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        payload = self._serializer.dumps(value)
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
