from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def normalize_message(message: Any) -> Dict[str, Any]:
    """Convert common chat-message shapes into a mutable mapping."""

    if isinstance(message, Mapping):
        return dict(message)
    if isinstance(message, tuple) and len(message) == 2:
        role, content = message
        return {"role": str(role), "content": content}
    role = getattr(message, "type", None) or getattr(message, "role", None) or message.__class__.__name__
    content = getattr(message, "content", "")
    return {"role": str(role), "content": content}


def content_to_text(content: Any) -> str:
    """Flatten provider content blocks into approximate text for planning."""

    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def content_blocks(content: Any) -> list[Any]:
    """Return provider-style content blocks without mutating caller input."""

    if isinstance(content, list):
        return [dict(item) if isinstance(item, dict) else item for item in content]
    return [{"type": "text", "text": str(content)}]
