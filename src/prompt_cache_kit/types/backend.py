from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class CacheEntry:
    value: Any
    created_at: float
    expires_at: Optional[float] = None

    def expired(self, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at


@dataclass(frozen=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0.0 if total == 0 else self.hits / total


class CacheBackend(Protocol):
    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None: ...

    def clear(self, namespace: Optional[str] = None) -> None: ...

    def stats(self) -> CacheStats: ...
