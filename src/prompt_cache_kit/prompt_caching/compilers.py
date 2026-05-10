from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..telemetry import CachePoint
from .messages import content_blocks


class CachePointCompiler(ABC):
    """Compiles provider-neutral cache points into provider-specific messages."""

    @abstractmethod
    def apply(self, message: Mapping[str, Any], point: Optional[CachePoint]) -> Dict[str, Any]:
        """Return a message with the cache point rendered, leaving inputs untouched."""


@dataclass(frozen=True)
class GenericCachePointCompiler(CachePointCompiler):
    def apply(self, message: Mapping[str, Any], point: Optional[CachePoint]) -> Dict[str, Any]:
        result = dict(message)
        if point is None:
            return result
        result["cache_point"] = {
            "id": point.id,
            "strategy": point.strategy,
            "provider_hint": point.provider_hint,
            "metadata": point.metadata,
        }
        return result


@dataclass(frozen=True)
class BedrockCachePointCompiler(CachePointCompiler):
    cache_type: str = "default"

    def apply(self, message: Mapping[str, Any], point: Optional[CachePoint]) -> Dict[str, Any]:
        result = dict(message)
        if point is None:
            return result
        blocks = content_blocks(result.get("content", ""))
        blocks.append(create_bedrock_cache_point(str(point.metadata.get("cache_type", self.cache_type))))
        result["content"] = blocks
        return result


@dataclass(frozen=True)
class AnthropicCachePointCompiler(CachePointCompiler):
    def apply(self, message: Mapping[str, Any], point: Optional[CachePoint]) -> Dict[str, Any]:
        result = dict(message)
        if point is None:
            return result
        ttl = point.metadata.get("ttl")
        blocks = content_blocks(result.get("content", ""))
        for index in range(len(blocks) - 1, -1, -1):
            block = blocks[index]
            if isinstance(block, dict) and block.get("type", "text") == "text":
                updated = dict(block)
                updated["cache_control"] = create_anthropic_cache_control(ttl=ttl)
                blocks[index] = updated
                result["content"] = blocks
                return result
        blocks.append({"type": "text", "text": "", "cache_control": create_anthropic_cache_control(ttl=ttl)})
        result["content"] = blocks
        return result


def compiler_for_provider(provider: str) -> CachePointCompiler:
    provider_name = provider.lower()
    if provider_name in {"bedrock", "aws", "langchain-bedrock"}:
        return BedrockCachePointCompiler()
    if provider_name in {"anthropic", "claude", "langchain-anthropic"}:
        return AnthropicCachePointCompiler()
    return GenericCachePointCompiler()


def create_bedrock_cache_point(cache_type: str = "default") -> Dict[str, Dict[str, str]]:
    return {"cachePoint": {"type": cache_type}}


def create_anthropic_cache_control(*, ttl: Optional[str] = None) -> Dict[str, str]:
    control = {"type": "ephemeral"}
    if ttl:
        control["ttl"] = ttl
    return control
