"""Embedding cache layer for reducing redundant API calls."""

import hashlib
import threading
from collections import OrderedDict
from typing import List, Optional


class EmbeddingCache:
    """Thread-safe LRU cache for embeddings.

    Caches embedding vectors by content hash to avoid redundant
    embedding calls for identical text.

    Args:
        max_size: Maximum number of embeddings to cache (default: 10000)
        enabled: Whether caching is enabled (default: True)

    Example:
        >>> cache = EmbeddingCache(max_size=5000)
        >>> cache.get("hello world")  # Returns None (cache miss)
        >>> cache.set("hello world", [0.1, 0.2, ...])
        >>> cache.get("hello world")  # Returns [0.1, 0.2, ...]
    """

    def __init__(self, max_size: int = 10000, enabled: bool = True) -> None:
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._max_size = max_size
        self._enabled = enabled
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        """Create a hash key for text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text.

        Args:
            text: The text to look up

        Returns:
            Cached embedding vector or None if not found
        """
        if not self._enabled:
            return None

        key = self._hash_text(text)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None

    def get_batch(self, texts: List[str]) -> tuple[List[Optional[List[float]]], List[int]]:
        """Get cached embeddings for multiple texts.

        Args:
            texts: List of texts to look up

        Returns:
            Tuple of (list of embeddings or None for misses, list of indices that missed)
        """
        results: List[Optional[List[float]]] = []
        missed_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = self.get(text)
            results.append(cached)
            if cached is None:
                missed_indices.append(i)

        return results, missed_indices

    def set(self, text: str, embedding: List[float]) -> None:
        """Cache embedding for text.

        Args:
            text: The text that was embedded
            embedding: The embedding vector to cache
        """
        if not self._enabled:
            return

        key = self._hash_text(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                self._cache[key] = embedding
                # Evict oldest if over capacity
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

    def set_batch(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Cache multiple embeddings at once.

        Args:
            texts: List of texts that were embedded
            embeddings: List of embedding vectors to cache
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Return current number of cached embeddings."""
        return len(self._cache)

    @property
    def stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dictionary with size, max_size, hits, misses, and hit_rate
        """
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"EmbeddingCache(size={self.size}, max_size={self._max_size}, "
            f"hit_rate={self.stats['hit_rate']:.2%})"
        )
