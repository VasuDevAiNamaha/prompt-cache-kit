from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable


def default_cache_key(
    namespace: str,
    identity: str,
    args: tuple,
    kwargs: dict,
    ignored_kwargs: Iterable[str] = (),
) -> str:
    ignored = {name.lower() for name in ignored_kwargs}
    clean_kwargs = {key: value for key, value in kwargs.items() if key.lower() not in ignored}
    payload = {
        "identity": identity,
        "args": _normalize(args),
        "kwargs": _normalize(clean_kwargs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        return _normalize(value.model_dump())
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _normalize(value.dict())
        except TypeError:
            pass
    return repr(value)
