from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..telemetry import CachePoint, TokenCounter, estimate_tokens
from .messages import content_to_text, normalize_message


@dataclass(frozen=True)
class CachePointSuggestion:
    message_index: int
    role: str
    cumulative_tokens: int
    cache_point: CachePoint
    reason: str


@dataclass(frozen=True)
class CacheDirective:
    """Provider-neutral instruction for where cache boundaries should exist."""

    message_index: int
    id: Optional[str] = None
    strategy: str = "prefix"
    provider_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reason: str = "explicit"


@dataclass(frozen=True)
class PromptCachingContext:
    """Normalized context passed to prompt caching strategies."""

    messages: Tuple[Mapping[str, Any], ...]
    provider: str
    cumulative_tokens: Tuple[int, ...]
    token_counter: TokenCounter = estimate_tokens

    def resolve_index(self, index: int) -> Optional[int]:
        return _resolve_index(index, len(self.messages))

    def message_tokens(self, index: int) -> int:
        previous = self.cumulative_tokens[index - 1] if index > 0 else 0
        return self.cumulative_tokens[index] - previous


class PromptCachingStrategy(ABC):
    """Strategy pattern extension point for provider-neutral prompt caching."""

    @abstractmethod
    def select(self, context: PromptCachingContext) -> Sequence[CacheDirective]:
        """Return cache directives for the normalized message array."""


@dataclass(frozen=True)
class ManualPromptCachingStrategy(PromptCachingStrategy):
    message_indexes: Tuple[int, ...]
    ids: Mapping[int, str] = field(default_factory=dict)
    excluded_ranges: Tuple[Tuple[int, int], ...] = ()

    def select(self, context: PromptCachingContext) -> Sequence[CacheDirective]:
        directives: List[CacheDirective] = []
        for raw_index in self.message_indexes:
            index = context.resolve_index(raw_index)
            if index is None or _excluded(index, self.excluded_ranges):
                continue
            directives.append(CacheDirective(message_index=index, id=self.ids.get(raw_index), reason="manual"))
        return directives

    def excluding(self, start: int, end: Optional[int] = None) -> "ManualPromptCachingStrategy":
        end_index = start if end is None else end
        return ManualPromptCachingStrategy(
            message_indexes=self.message_indexes,
            ids=self.ids,
            excluded_ranges=self.excluded_ranges + ((start, end_index),),
        )


@dataclass(frozen=True)
class CacheUntilPromptCachingStrategy(PromptCachingStrategy):
    message_index: int
    id: Optional[str] = None

    def select(self, context: PromptCachingContext) -> Sequence[CacheDirective]:
        index = context.resolve_index(self.message_index)
        if index is None:
            return ()
        return (CacheDirective(message_index=index, id=self.id, reason="cache-until"),)


@dataclass(frozen=True)
class StablePrefixPromptCachingStrategy(PromptCachingStrategy):
    min_tokens: int = 1024
    max_points: int = 4

    def select(self, context: PromptCachingContext) -> Sequence[CacheDirective]:
        last_index = len(context.messages) - 1
        directives: List[CacheDirective] = []
        for index, message in enumerate(context.messages):
            if not _message_stable(message, index, last_index):
                break
            if context.cumulative_tokens[index] >= self.min_tokens:
                directives.append(CacheDirective(message_index=index, reason="auto-stable-prefix"))
        if len(directives) <= self.max_points:
            return directives
        return _spread_directives(directives, self.max_points)


@dataclass(frozen=True)
class RollingPromptCachingStrategy(PromptCachingStrategy):
    start_index: int = 0
    every_messages: int = 2
    min_tokens: int = 1024
    max_points: int = 4
    excluded_ranges: Tuple[Tuple[int, int], ...] = ()

    def select(self, context: PromptCachingContext) -> Sequence[CacheDirective]:
        start = max(context.resolve_index(self.start_index) or 0, 0)
        last_index = len(context.messages) - 1
        directives: List[CacheDirective] = []
        for index in range(start, len(context.messages)):
            if _excluded(index, self.excluded_ranges):
                continue
            if not _message_stable(context.messages[index], index, last_index):
                continue
            if context.cumulative_tokens[index] < self.min_tokens:
                continue
            if (index - start) % max(self.every_messages, 1) != 0 and index != last_index - 1:
                continue
            directives.append(
                CacheDirective(
                    message_index=index,
                    strategy="rolling",
                    provider_hint=context.provider,
                    metadata={
                        "every_messages": self.every_messages,
                        "min_tokens": self.min_tokens,
                        "max_points": self.max_points,
                    },
                    reason="rolling-old-prefix",
                )
            )
        if len(directives) <= self.max_points:
            return directives
        return _spread_directives(directives, self.max_points)

    def excluding(self, start: int, end: Optional[int] = None) -> "RollingPromptCachingStrategy":
        end_index = start if end is None else end
        return RollingPromptCachingStrategy(
            start_index=self.start_index,
            every_messages=self.every_messages,
            min_tokens=self.min_tokens,
            max_points=self.max_points,
            excluded_ranges=self.excluded_ranges + ((start, end_index),),
        )


MessageCachePlan = PromptCachingStrategy


def suggest_cache_points(
    messages: Iterable[Any],
    *,
    provider: str = "anthropic",
    min_tokens: int = 1024,
    max_points: int = 4,
    token_counter: TokenCounter = estimate_tokens,
) -> List[CachePointSuggestion]:
    """Suggest useful cache breakpoints for a message array."""

    return plan_cache_points(
        messages,
        StablePrefixPromptCachingStrategy(min_tokens=min_tokens, max_points=max_points),
        provider=provider,
        token_counter=token_counter,
    )


def plan_cache_points(
    messages: Iterable[Any],
    strategy: Optional[PromptCachingStrategy] = None,
    *,
    provider: str = "anthropic",
    token_counter: TokenCounter = estimate_tokens,
) -> List[CachePointSuggestion]:
    """Compile a provider-neutral prompt caching strategy into concrete suggestions."""

    normalized = [normalize_message(message) for message in messages]
    if not normalized:
        return []
    cumulative_tokens = _cumulative_tokens(normalized, token_counter)
    context = PromptCachingContext(
        messages=tuple(normalized),
        provider=provider,
        cumulative_tokens=tuple(cumulative_tokens),
        token_counter=token_counter,
    )
    active_strategy = strategy or RollingPromptCachingStrategy()
    suggestions: List[CachePointSuggestion] = []
    for directive in active_strategy.select(context):
        index = _resolve_index(directive.message_index, len(normalized))
        if index is None:
            continue
        suggestions.append(_suggestion_from_directive(normalized[index], index, directive, provider, cumulative_tokens[index]))
    return suggestions


def _message_stable(message: Mapping[str, Any], index: int, last_index: int) -> bool:
    if "stable" in message:
        return bool(message["stable"])
    role = str(message.get("role", "")).lower()
    if index == last_index and role in {"user", "human"}:
        return False
    return role in {"system", "user", "human", "assistant", "ai"}


def _cumulative_tokens(messages: Sequence[Mapping[str, Any]], token_counter: TokenCounter) -> List[int]:
    total = 0
    cumulative: List[int] = []
    for message in messages:
        total += token_counter(content_to_text(message.get("content", "")))
        cumulative.append(total)
    return cumulative


def _suggestion_from_directive(
    message: Mapping[str, Any],
    index: int,
    directive: CacheDirective,
    provider: str,
    cumulative_tokens: int,
) -> CachePointSuggestion:
    point_id = directive.id or f"{provider}-prefix-{index}"
    metadata = dict(directive.metadata)
    metadata.setdefault("message_index", index)
    metadata.setdefault("cumulative_tokens", cumulative_tokens)
    return CachePointSuggestion(
        message_index=index,
        role=str(message.get("role", "")),
        cumulative_tokens=cumulative_tokens,
        cache_point=CachePoint(
            id=point_id,
            strategy=directive.strategy,
            provider_hint=directive.provider_hint or provider,
            metadata=metadata,
        ),
        reason=directive.reason,
    )


def _resolve_index(index: int, length: int) -> Optional[int]:
    resolved = index if index >= 0 else length + index
    if resolved < 0 or resolved >= length:
        return None
    return resolved


def _excluded(index: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    for start, end in ranges:
        low, high = sorted((start, end))
        if low <= index <= high:
            return True
    return False


def _spread_directives(directives: Sequence[CacheDirective], max_points: int) -> List[CacheDirective]:
    if max_points <= 1:
        return [directives[-1]]
    step = (len(directives) - 1) / (max_points - 1)
    indexes = {round(step * i) for i in range(max_points)}
    return [directives[index] for index in sorted(indexes)]
