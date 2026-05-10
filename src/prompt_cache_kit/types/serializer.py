from __future__ import annotations

from typing import Protocol, TypeVar

T = TypeVar("T", contravariant=True)
R = TypeVar("R", covariant=True)


class Serializer(Protocol[T, R]):
    def dumps(self, value: T) -> bytes: ...

    def loads(self, payload: bytes) -> R: ...
