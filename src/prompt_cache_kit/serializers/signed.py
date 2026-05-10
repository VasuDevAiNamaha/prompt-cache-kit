from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from ..types import Serializer

_SIG_PREFIX = b"pck1:"


@dataclass(frozen=True)
class SignedSerializer(Serializer[object, object]):
    inner: Serializer[object, object]
    signing_key: bytes

    def dumps(self, value: object) -> bytes:
        payload = self.inner.dumps(value)
        mac = hmac.new(self.signing_key, payload, hashlib.sha256).digest()
        return _SIG_PREFIX + mac + payload

    def loads(self, payload: bytes) -> object:
        verified = _verify_and_strip_signature(payload, self.signing_key)
        return self.inner.loads(verified)


def _verify_and_strip_signature(blob: bytes, key: bytes) -> bytes:
    if not blob.startswith(_SIG_PREFIX) or len(blob) < len(_SIG_PREFIX) + 32:
        raise ValueError("missing cache signature")
    mac = blob[len(_SIG_PREFIX) : len(_SIG_PREFIX) + 32]
    payload = blob[len(_SIG_PREFIX) + 32 :]
    expected = hmac.new(key, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        raise ValueError("invalid cache signature")
    return payload
