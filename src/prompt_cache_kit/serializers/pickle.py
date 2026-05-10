from __future__ import annotations

import pickle  # nosec B403
from typing import Any

from ..types import Serializer


class PickleSerializer(Serializer[Any, Any]):
    def dumps(self, value: Any) -> bytes:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def loads(self, payload: bytes) -> Any:
        return pickle.loads(payload)  # nosec B301
