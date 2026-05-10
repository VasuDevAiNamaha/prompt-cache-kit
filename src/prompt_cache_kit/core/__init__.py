"""Core primitives (keys, policies) that are backend-agnostic."""

from .keys import default_cache_key
from .policy import CachePolicy, KeyFunc

__all__ = ["CachePolicy", "KeyFunc", "default_cache_key"]
