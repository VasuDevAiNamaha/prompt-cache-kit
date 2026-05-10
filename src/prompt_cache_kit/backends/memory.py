from __future__ import annotations

import time
from threading import RLock
from typing import Any, Dict, Optional

from ..types import CacheEntry, CacheStats


class MemoryCacheBackend:
    """Small threadsafe in-memory backend with TTL and optional max-size eviction."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        self._entries: Dict[str, CacheEntry] = {}
        self._last_access: Dict[str, float] = {}
        self._max_size = max_size
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0

    def get(self, key: str) -> Any:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expired(now):
                self._entries.pop(key, None)
                self._last_access.pop(key, None)
                self._misses += 1
                self._evictions += 1
                return None
            self._hits += 1
            self._last_access[key] = now
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        now = time.time()
        expires_at = None if ttl_seconds is None else now + ttl_seconds
        with self._lock:
            self._entries[key] = CacheEntry(value=value, created_at=now, expires_at=expires_at)
            self._last_access[key] = now
            self._sets += 1
            self._evict_if_needed()

    def clear(self, namespace: Optional[str] = None) -> None:
        with self._lock:
            if namespace is None:
                self._entries.clear()
                self._last_access.clear()
                return
            prefix = f"{namespace}:"
            for key in list(self._entries):
                if key.startswith(prefix):
                    self._entries.pop(key, None)
                    self._last_access.pop(key, None)

    def stats(self) -> CacheStats:
        with self._lock:
            self._prune_expired()
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                sets=self._sets,
                evictions=self._evictions,
                size=len(self._entries),
            )

    def _evict_if_needed(self) -> None:
        if self._max_size is None:
            return
        while len(self._entries) > self._max_size:
            oldest_key = min(self._last_access, key=self._last_access.__getitem__)
            self._entries.pop(oldest_key, None)
            self._last_access.pop(oldest_key, None)
            self._evictions += 1

    def _prune_expired(self) -> None:
        now = time.time()
        for key, entry in list(self._entries.items()):
            if entry.expired(now):
                self._entries.pop(key, None)
                self._last_access.pop(key, None)
                self._evictions += 1
