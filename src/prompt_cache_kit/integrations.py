from __future__ import annotations

from typing import Any, Optional

from .backend import CacheBackend
from .policy import CachePolicy
from .telemetry import UsageStats, normalize_usage
from .wrappers import CachedModel


def wrap_langchain_model(
    model: Any,
    *,
    backend: Optional[CacheBackend] = None,
    policy: Optional[CachePolicy] = None,
) -> CachedModel:
    """Wrap common LangChain/LangGraph model methods without importing LangChain."""

    return CachedModel(
        model,
        backend=backend,
        policy=policy,
        methods=("invoke", "ainvoke", "generate", "agenerate", "batch", "abatch", "__call__"),
        identity=f"langchain:{type(model).__module__}.{type(model).__qualname__}",
    )


def wrap_crewai_llm(
    llm: Any,
    *,
    backend: Optional[CacheBackend] = None,
    policy: Optional[CachePolicy] = None,
) -> CachedModel:
    """Wrap common CrewAI LLM call surfaces without importing CrewAI."""

    return CachedModel(
        llm,
        backend=backend,
        policy=policy,
        methods=("call", "acall", "invoke", "ainvoke", "__call__"),
        identity=f"crewai:{type(llm).__module__}.{type(llm).__qualname__}",
    )


def extract_langchain_usage(result: Any) -> UsageStats:
    """Extract normalized usage from common LangChain AIMessage/LLMResult shapes."""

    usage = getattr(result, "usage_metadata", None)
    if usage:
        return normalize_usage(_langchain_usage_to_openai_shape(usage), provider="langchain")

    response_metadata = getattr(result, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if token_usage:
            return normalize_usage(token_usage, provider="langchain")

    llm_output = getattr(result, "llm_output", None)
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if token_usage:
            return normalize_usage(token_usage, provider="langchain")

    return UsageStats(provider="langchain")


def make_langchain_cache(backend: CacheBackend, *, namespace: str = "langchain"):
    """Create a LangChain BaseCache adapter when langchain_core is installed."""

    try:
        from langchain_core.caches import BaseCache
    except ImportError as exc:
        raise ImportError("Install langchain-core to use make_langchain_cache().") from exc

    class PromptCacheKitLangChainCache(BaseCache):
        def lookup(self, prompt: str, llm_string: str):
            return backend.get(f"{namespace}:{llm_string}:{prompt}")

        def update(self, prompt: str, llm_string: str, return_val) -> None:
            backend.set(f"{namespace}:{llm_string}:{prompt}", return_val)

        def clear(self, **kwargs: Any) -> None:
            backend.clear(namespace)

    return PromptCacheKitLangChainCache()


def _langchain_usage_to_openai_shape(usage: Any) -> dict:
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        usage = vars(usage) if hasattr(usage, "__dict__") else {}
    return {
        "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "completion_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
        "prompt_tokens_details": {
            "cached_tokens": usage.get("cache_read_input_tokens", usage.get("cached_tokens", 0)),
            "cache_write_tokens": usage.get("cache_creation_input_tokens", usage.get("cache_write_tokens", 0)),
        },
    }
