from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class CachePoint:
    """A user- or library-defined boundary that should remain stable for prefix caching."""

    id: str
    strategy: str = "prefix"
    provider_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MessageUsage:
    index: int
    role: str
    content_tokens: int
    cache_point_id: Optional[str] = None
    stable: Optional[bool] = None


@dataclass(frozen=True)
class UsageStats:
    """Normalized token/cache usage with OpenAI/OpenRouter and OpenTelemetry exports."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_tokens: int = 0
    message_usages: Sequence[MessageUsage] = ()
    cache_points: Sequence[CachePoint] = ()
    cache_hit: Optional[bool] = None
    provider: Optional[str] = None
    raw_usage: Optional[Mapping[str, Any]] = None

    @property
    def uncached_input_tokens(self) -> int:
        return max(self.input_tokens - self.cache_read_input_tokens, 0)

    def to_openai_usage(self) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            "cached_tokens": self.cache_read_input_tokens,
            "cache_write_tokens": self.cache_write_input_tokens,
        }
        if self.reasoning_tokens:
            details["reasoning_tokens"] = self.reasoning_tokens
        return {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.total_tokens or self.input_tokens + self.output_tokens,
            "prompt_tokens_details": details,
        }

    def to_otel_attributes(self) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "gen_ai.usage.input_tokens": self.input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
            "gen_ai.usage.cache_read.input_tokens": self.cache_read_input_tokens,
            "gen_ai.usage.cache_creation.input_tokens": self.cache_write_input_tokens,
        }
        if self.provider:
            attrs["gen_ai.system"] = self.provider
        if self.cache_hit is not None:
            attrs["gen_ai.cache.hit"] = self.cache_hit
        if self.reasoning_tokens:
            attrs["gen_ai.usage.reasoning_tokens"] = self.reasoning_tokens
        return attrs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens or self.input_tokens + self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "cache_hit": self.cache_hit,
            "provider": self.provider,
            "messages": [usage.__dict__ for usage in self.message_usages],
            "cache_points": [point.__dict__ for point in self.cache_points],
        }


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer fallback for analysis; callers can pass tiktoken/etc. instead."""

    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def analyze_messages(
    messages: Iterable[Mapping[str, Any]],
    *,
    token_counter: TokenCounter = estimate_tokens,
) -> UsageStats:
    message_usages: List[MessageUsage] = []
    cache_points: List[CachePoint] = []
    total = 0

    for index, message in enumerate(messages):
        content = _content_to_text(message.get("content", ""))
        tokens = token_counter(content)
        total += tokens
        cache_point = _cache_point_from_mapping(message)
        if cache_point is not None:
            cache_points.append(cache_point)
        message_usages.append(
            MessageUsage(
                index=index,
                role=str(message.get("role", "")),
                content_tokens=tokens,
                cache_point_id=cache_point.id if cache_point else None,
                stable=_bool_or_none(message.get("stable")),
            )
        )

    return UsageStats(input_tokens=total, total_tokens=total, message_usages=message_usages, cache_points=cache_points)


def normalize_usage(response_or_usage: Any, *, provider: Optional[str] = None) -> UsageStats:
    usage = _extract_usage_mapping(response_or_usage)
    details = _mapping(usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {})
    completion_details = _mapping(usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {})

    input_tokens = _int_first(usage, ("prompt_tokens", "input_tokens"))
    output_tokens = _int_first(usage, ("completion_tokens", "output_tokens"))
    total_tokens = _int_first(usage, ("total_tokens",))
    cache_read = _int_first(
        usage,
        ("cache_read_input_tokens", "cached_input_tokens"),
        default=_int_first(details, ("cached_tokens", "cache_read_tokens"), default=0),
    )
    cache_write = _int_first(
        usage,
        ("cache_creation_input_tokens", "cache_write_input_tokens"),
        default=_int_first(details, ("cache_write_tokens", "cache_creation_tokens"), default=0),
    )
    reasoning = _int_first(
        usage,
        ("reasoning_tokens",),
        default=_int_first(completion_details, ("reasoning_tokens",), default=0),
    )
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
        reasoning_tokens=reasoning,
        cache_hit=cache_read > 0,
        provider=provider,
        raw_usage=usage,
    )


def merge_usage(*stats: UsageStats) -> UsageStats:
    cache_points: List[CachePoint] = []
    message_usages: List[MessageUsage] = []
    for item in stats:
        cache_points.extend(item.cache_points)
        message_usages.extend(item.message_usages)
    return UsageStats(
        input_tokens=sum(item.input_tokens for item in stats),
        output_tokens=sum(item.output_tokens for item in stats),
        total_tokens=sum(item.total_tokens or item.input_tokens + item.output_tokens for item in stats),
        cache_read_input_tokens=sum(item.cache_read_input_tokens for item in stats),
        cache_write_input_tokens=sum(item.cache_write_input_tokens for item in stats),
        reasoning_tokens=sum(item.reasoning_tokens for item in stats),
        message_usages=message_usages,
        cache_points=cache_points,
        cache_hit=any(item.cache_hit for item in stats if item.cache_hit is not None),
    )


def _extract_usage_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        if "usage" in value and isinstance(value["usage"], Mapping):
            return value["usage"]
        return value
    usage = getattr(value, "usage", None)
    if usage is not None:
        return _mapping(usage)
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _int_first(mapping: Mapping[str, Any], names: Sequence[str], *, default: int = 0) -> int:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
    return default


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: List[str] = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _cache_point_from_mapping(message: Mapping[str, Any]) -> Optional[CachePoint]:
    raw = message.get("cache_point") or message.get("cachePoint")
    if isinstance(raw, CachePoint):
        return raw
    if isinstance(raw, Mapping):
        return CachePoint(
            id=str(raw.get("id")),
            strategy=str(raw.get("strategy", "prefix")),
            provider_hint=raw.get("provider_hint") or raw.get("providerHint"),
            metadata=dict(raw.get("metadata") or {}),
        )
    if raw:
        return CachePoint(id=str(raw))
    return None


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
