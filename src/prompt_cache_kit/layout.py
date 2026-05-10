from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .telemetry import CachePoint, TokenCounter, UsageStats, analyze_messages


@dataclass(frozen=True)
class CacheBlock:
    role: str
    content: str
    stable: bool
    name: Optional[str] = None
    provider_cache_control: bool = False
    cache_point: Optional[CachePoint] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LintIssue:
    code: str
    message: str
    block_index: Optional[int] = None


class PromptLayout:
    """Builds prompts with stable reusable blocks before dynamic blocks."""

    def __init__(self, blocks: Optional[List[CacheBlock]] = None) -> None:
        self._blocks = list(blocks or [])

    @property
    def blocks(self) -> List[CacheBlock]:
        return list(self._blocks)

    def stable_system(self, content: str, *, name: Optional[str] = None, cache_control: bool = True) -> "PromptLayout":
        return self.add("system", content, stable=True, name=name, cache_control=cache_control)

    def stable_context(self, content: str, *, name: Optional[str] = None, cache_control: bool = True) -> "PromptLayout":
        return self.add("system", content, stable=True, name=name or "context", cache_control=cache_control)

    def dynamic_user(self, content: str, *, name: Optional[str] = None) -> "PromptLayout":
        return self.add("user", content, stable=False, name=name)

    def cache_point(
        self,
        id: str,
        *,
        strategy: str = "prefix",
        provider_hint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "PromptLayout":
        if not self._blocks:
            raise ValueError("Add a prompt block before attaching a cache point.")
        point = CachePoint(id=id, strategy=strategy, provider_hint=provider_hint, metadata=dict(metadata or {}))
        last = self._blocks[-1]
        self._blocks[-1] = CacheBlock(
            role=last.role,
            content=last.content,
            stable=last.stable,
            name=last.name,
            provider_cache_control=last.provider_cache_control,
            cache_point=point,
            metadata=last.metadata,
        )
        return self

    def add(
        self,
        role: str,
        content: str,
        *,
        stable: bool,
        name: Optional[str] = None,
        cache_control: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        cache_point: Optional[CachePoint] = None,
    ) -> "PromptLayout":
        self._blocks.append(
            CacheBlock(
                role=role,
                content=content,
                stable=stable,
                name=name,
                provider_cache_control=cache_control,
                cache_point=cache_point,
                metadata=dict(metadata or {}),
            )
        )
        return self

    def reorder_stable_first(self) -> "PromptLayout":
        stable = [block for block in self._blocks if block.stable]
        dynamic = [block for block in self._blocks if not block.stable]
        return PromptLayout(stable + dynamic)

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        return [
            _message_dict(block, include_cache_control=False, include_instrumentation=False)
            for block in self.reorder_stable_first().blocks
        ]

    def to_anthropic_messages(self) -> List[Dict[str, Any]]:
        return [
            _message_dict(block, include_cache_control=True, include_instrumentation=False)
            for block in self.reorder_stable_first().blocks
        ]

    def to_instrumented_messages(self) -> List[Dict[str, Any]]:
        return [
            _message_dict(block, include_cache_control=True, include_instrumentation=True)
            for block in self.reorder_stable_first().blocks
        ]

    def prefix_text(self, separator: str = "\n\n") -> str:
        return separator.join(block.content for block in self.reorder_stable_first().blocks)

    def usage(self, *, token_counter: TokenCounter = None) -> UsageStats:  # type: ignore[assignment]
        if token_counter is None:
            return analyze_messages(self.to_instrumented_messages())
        return analyze_messages(self.to_instrumented_messages(), token_counter=token_counter)

    def lint(self) -> List[LintIssue]:
        issues: List[LintIssue] = []
        seen_dynamic = False
        for index, block in enumerate(self._blocks):
            if not block.stable:
                seen_dynamic = True
            elif seen_dynamic:
                issues.append(
                    LintIssue(
                        code="stable-after-dynamic",
                        message="Stable prompt block appears after dynamic content, reducing prefix-cache hits.",
                        block_index=index,
                    )
                )
            if block.stable and _looks_dynamic(block.content):
                issues.append(
                    LintIssue(
                        code="dynamic-marker-in-stable-block",
                        message="Stable block contains timestamp, UUID, or random-looking marker.",
                        block_index=index,
                    )
                )
        return issues


def _message_dict(block: CacheBlock, *, include_cache_control: bool, include_instrumentation: bool) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": block.role, "content": block.content}
    if block.name:
        message["name"] = block.name
    if include_cache_control and block.provider_cache_control:
        message["cache_control"] = {"type": "ephemeral"}
    if include_instrumentation and block.cache_point:
        message["cache_point"] = {
            "id": block.cache_point.id,
            "strategy": block.cache_point.strategy,
            "provider_hint": block.cache_point.provider_hint,
            "metadata": block.cache_point.metadata,
        }
    if include_instrumentation:
        message["stable"] = block.stable
    return message


_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
_ISO_TS_RE = re.compile(r"\b20\d\d-\d\d-\d\d[T ][0-2]\d:[0-5]\d")
_RAND_RE = re.compile(r"\b(?:nonce|request_id|trace_id|timestamp|current time)\b", re.I)


def _looks_dynamic(text: str) -> bool:
    return bool(_UUID_RE.search(text) or _ISO_TS_RE.search(text) or _RAND_RE.search(text))
