from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence

from .compilers import (
    CachePointCompiler,
    compiler_for_provider,
)
from .messages import normalize_message
from .strategies import (
    CachePointSuggestion,
    CacheUntilPromptCachingStrategy,
    ManualPromptCachingStrategy,
    MessageCachePlan,
    PromptCachingStrategy,
    RollingPromptCachingStrategy,
    plan_cache_points,
)
from ..telemetry import TokenCounter, estimate_tokens


def apply_cache_points(
    messages: Iterable[Any],
    suggestions: Optional[Sequence[CachePointSuggestion]] = None,
    *,
    plan: Optional[PromptCachingStrategy] = None,
    strategy: Optional[PromptCachingStrategy] = None,
    provider: str = "anthropic",
    min_tokens: int = 1024,
    max_points: int = 4,
    token_counter: TokenCounter = estimate_tokens,
    compiler: Optional[CachePointCompiler] = None,
) -> List[Any]:
    """Return messages with provider-specific prompt-cache markers applied."""

    normalized = [normalize_message(message) for message in messages]
    active = list(
        suggestions
        if suggestions is not None
        else plan_cache_points(
            normalized,
            strategy
            or plan
            or RollingPromptCachingStrategy(
                every_messages=1,
                min_tokens=min_tokens,
                max_points=max_points,
            ),
            provider=provider,
            token_counter=token_counter,
        )
    )
    by_index = {item.message_index: item.cache_point for item in active}
    active_compiler = compiler or compiler_for_provider(provider)
    return [active_compiler.apply(message, by_index.get(index)) for index, message in enumerate(normalized)]


def cache_until(message_index: int, *, id: Optional[str] = None) -> MessageCachePlan:
    return CacheUntilPromptCachingStrategy(message_index=message_index, id=id)


def cache_at(*message_indexes: int, ids: Optional[Mapping[int, str]] = None) -> MessageCachePlan:
    return ManualPromptCachingStrategy(message_indexes=tuple(message_indexes), ids=ids or {})


def rolling_cache(
    *,
    start_index: int = 0,
    every_messages: int = 2,
    min_tokens: int = 1024,
    max_points: int = 4,
) -> MessageCachePlan:
    return RollingPromptCachingStrategy(
        start_index=start_index,
        every_messages=every_messages,
        min_tokens=min_tokens,
        max_points=max_points,
    )
