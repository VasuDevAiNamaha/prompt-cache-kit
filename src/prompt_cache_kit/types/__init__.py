"""Shared Protocols and datatypes used across Prompt Cache Kit."""

from .backend import CacheBackend, CacheEntry, CacheStats
from .serializer import Serializer

__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "Serializer",
]
