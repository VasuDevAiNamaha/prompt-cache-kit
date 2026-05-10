from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

KeyFunc = Callable[[str, tuple, dict], str]


@dataclass(frozen=True)
class CachePolicy:
    """Controls cache-key identity and cache behavior for wrapped calls."""

    namespace: str = "default"
    ttl_seconds: Optional[float] = None
    enabled: bool = True
    bypass_kwarg: str = "_cache_bypass"
    ignored_kwargs: Iterable[str] = field(
        default_factory=lambda: (
            "api_key",
            "authorization",
            "headers",
            "timeout",
            "request_id",
            "user",
            "_cache_bypass",
        )
    )
    key_func: Optional[KeyFunc] = None
