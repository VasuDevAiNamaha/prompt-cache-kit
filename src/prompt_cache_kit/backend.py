"""Compatibility exports for the old single-module backend layout."""

from .backends import MemoryCacheBackend
from .types import CacheBackend, CacheEntry, CacheStats

__all__ = ["CacheBackend", "CacheEntry", "CacheStats", "MemoryCacheBackend"]
