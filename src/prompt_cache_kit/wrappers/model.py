from __future__ import annotations

from typing import Any, Iterable, Optional

from ..backends import MemoryCacheBackend
from ..core import CachePolicy
from ..types import CacheBackend
from .cached import cached


class CachedModel:
    """Proxy that wraps common model methods while delegating everything else."""

    def __init__(
        self,
        model: Any,
        *,
        backend: Optional[CacheBackend] = None,
        policy: Optional[CachePolicy] = None,
        methods: Iterable[str] = ("invoke", "ainvoke", "generate", "agenerate", "__call__"),
        identity: Optional[str] = None,
    ) -> None:
        self._model = model
        self._backend = backend or MemoryCacheBackend()
        self._policy = policy or CachePolicy()
        self._methods = set(methods)
        self._identity = identity or f"{type(model).__module__}.{type(model).__qualname__}"

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._model, name)
        if name not in self._methods or not callable(attr):
            return attr
        return cached(
            attr,
            backend=self._backend,
            policy=self._policy,
            identity=f"{self._identity}.{name}",
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if "__call__" not in self._methods:
            return self._model(*args, **kwargs)
        wrapped = cached(
            self._model,
            backend=self._backend,
            policy=self._policy,
            identity=f"{self._identity}.__call__",
        )
        return wrapped(*args, **kwargs)

    @property
    def wrapped(self) -> Any:
        return self._model

    @property
    def cache_backend(self) -> CacheBackend:
        return self._backend
