from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..backends import MemoryCacheBackend
from ..core import CachePolicy, default_cache_key
from ..types import CacheBackend


@dataclass(frozen=True)
class _CachedNone:
    marker: str = "prompt_cache_kit.cached_none"


_CACHED_NONE = _CachedNone()


def cached(
    func: Optional[Callable[..., Any]] = None,
    *,
    backend: Optional[CacheBackend] = None,
    policy: Optional[CachePolicy] = None,
    identity: Optional[str] = None,
) -> Callable[..., Any]:
    """Cache a synchronous or asynchronous callable by normalized args/kwargs."""

    active_backend = backend or MemoryCacheBackend()
    active_policy = policy or CachePolicy()

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        call_identity = identity or _callable_identity(target)

        if inspect.isasyncgenfunction(target) or inspect.isgeneratorfunction(target):

            @functools.wraps(target)
            def passthrough_wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs.pop(active_policy.bypass_kwarg, None)
                return target(*args, **kwargs)

            return passthrough_wrapper

        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                bypass = kwargs.pop(active_policy.bypass_kwarg, False)
                if not active_policy.enabled or bypass:
                    return await target(*args, **kwargs)
                key = _make_key(active_policy, call_identity, args, kwargs)
                cached_value = active_backend.get(key)
                if cached_value is not None:
                    return None if cached_value == _CACHED_NONE else cached_value
                value = await target(*args, **kwargs)
                if _cacheable_value(value):
                    active_backend.set(key, _CACHED_NONE if value is None else value, active_policy.ttl_seconds)
                return value

            return async_wrapper

        @functools.wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bypass = kwargs.pop(active_policy.bypass_kwarg, False)
            if not active_policy.enabled or bypass:
                return target(*args, **kwargs)
            key = _make_key(active_policy, call_identity, args, kwargs)
            cached_value = active_backend.get(key)
            if cached_value is not None:
                return None if cached_value == _CACHED_NONE else cached_value
            value = target(*args, **kwargs)
            if _cacheable_value(value):
                active_backend.set(key, _CACHED_NONE if value is None else value, active_policy.ttl_seconds)
            return value

        return wrapper

    return decorate(func) if func is not None else decorate


def _make_key(policy: CachePolicy, identity: str, args: tuple, kwargs: dict) -> str:
    if policy.key_func is not None:
        return policy.key_func(identity, args, kwargs)
    return default_cache_key(policy.namespace, identity, args, kwargs, policy.ignored_kwargs)


def _callable_identity(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "unknown")
    qualname = getattr(func, "__qualname__", repr(func))
    return f"{module}.{qualname}"


def _cacheable_value(value: Any) -> bool:
    return not inspect.isgenerator(value) and not inspect.isasyncgen(value)
