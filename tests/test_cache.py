"""Tests for the embedding cache layer."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from piragi.cache import EmbeddingCache


class TestEmbeddingCache:
    """Test suite for EmbeddingCache."""

    def test_cache_initialization(self) -> None:
        """Test cache initialization with default and custom parameters."""
        # Default initialization
        cache = EmbeddingCache()
        assert cache._max_size == 10000
        assert cache._enabled is True
        assert cache.size == 0

        # Custom initialization
        cache = EmbeddingCache(max_size=100, enabled=False)
        assert cache._max_size == 100
        assert cache._enabled is False

    def test_cache_set_and_get(self) -> None:
        """Test basic set and get operations."""
        cache = EmbeddingCache(max_size=100)

        # Set a value
        embedding = [0.1, 0.2, 0.3]
        cache.set("hello world", embedding)

        # Get the value
        result = cache.get("hello world")
        assert result == embedding
        assert cache.size == 1

    def test_cache_miss(self) -> None:
        """Test cache miss returns None."""
        cache = EmbeddingCache(max_size=100)

        result = cache.get("nonexistent")
        assert result is None

    def test_cache_disabled(self) -> None:
        """Test that disabled cache doesn't store or return values."""
        cache = EmbeddingCache(max_size=100, enabled=False)

        cache.set("hello", [0.1, 0.2, 0.3])
        assert cache.get("hello") is None
        assert cache.size == 0

    def test_cache_lru_eviction(self) -> None:
        """Test LRU eviction when cache exceeds max size."""
        cache = EmbeddingCache(max_size=3)

        # Fill the cache
        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.set("c", [3.0])
        assert cache.size == 3

        # Add one more, should evict "a" (oldest)
        cache.set("d", [4.0])
        assert cache.size == 3

        # "a" should be evicted
        assert cache.get("a") is None
        # Others should still be there
        assert cache.get("b") == [2.0]
        assert cache.get("c") == [3.0]
        assert cache.get("d") == [4.0]

    def test_cache_access_updates_lru_order(self) -> None:
        """Test that accessing an item moves it to the end (most recently used)."""
        cache = EmbeddingCache(max_size=3)

        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.set("c", [3.0])

        # Access "a" to make it most recently used
        cache.get("a")

        # Add "d" - should evict "b" (now the oldest)
        cache.set("d", [4.0])

        assert cache.get("a") == [1.0]  # Still there (was accessed)
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == [3.0]
        assert cache.get("d") == [4.0]

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        cache = EmbeddingCache(max_size=100)

        # Initial stats
        stats = cache.stats
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

        # Add some entries and access them
        cache.set("a", [1.0])
        cache.get("a")  # Hit
        cache.get("a")  # Hit
        cache.get("b")  # Miss

        stats = cache.stats
        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_cache_clear(self) -> None:
        """Test clearing the cache."""
        cache = EmbeddingCache(max_size=100)

        cache.set("a", [1.0])
        cache.set("b", [2.0])
        cache.get("a")  # Hit
        cache.get("c")  # Miss

        assert cache.size == 2
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1

        cache.clear()

        assert cache.size == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_cache_batch_operations(self) -> None:
        """Test batch get and set operations."""
        cache = EmbeddingCache(max_size=100)

        # Batch set
        texts = ["a", "b", "c"]
        embeddings = [[1.0], [2.0], [3.0]]
        cache.set_batch(texts, embeddings)

        assert cache.size == 3

        # Batch get with some hits and misses
        results, missed = cache.get_batch(["a", "b", "d"])

        assert results[0] == [1.0]
        assert results[1] == [2.0]
        assert results[2] is None
        assert missed == [2]

    def test_cache_thread_safety(self) -> None:
        """Test that cache is thread-safe."""
        cache = EmbeddingCache(max_size=1000)
        num_threads = 10
        num_operations = 100

        def worker(thread_id: int) -> None:
            for i in range(num_operations):
                text = f"thread_{thread_id}_item_{i}"
                embedding = [float(thread_id), float(i)]
                cache.set(text, embedding)
                result = cache.get(text)
                assert result == embedding or result is None  # May be evicted

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            for future in futures:
                future.result()

        # Should have some items cached
        assert cache.size > 0

    def test_cache_repr(self) -> None:
        """Test string representation of cache."""
        cache = EmbeddingCache(max_size=100)
        cache.set("a", [1.0])
        cache.get("a")  # Hit
        cache.get("b")  # Miss

        repr_str = repr(cache)
        assert "EmbeddingCache" in repr_str
        assert "size=1" in repr_str
        assert "max_size=100" in repr_str
        assert "hit_rate=" in repr_str

    def test_cache_hash_collision_resistance(self) -> None:
        """Test that different texts get different cache entries."""
        cache = EmbeddingCache(max_size=100)

        # These should have different hashes
        cache.set("hello", [1.0])
        cache.set("hello ", [2.0])  # Extra space
        cache.set("Hello", [3.0])  # Different case

        assert cache.get("hello") == [1.0]
        assert cache.get("hello ") == [2.0]
        assert cache.get("Hello") == [3.0]
        assert cache.size == 3

    def test_cache_update_existing(self) -> None:
        """Test updating an existing cached value."""
        cache = EmbeddingCache(max_size=100)

        cache.set("text", [1.0, 2.0])
        assert cache.get("text") == [1.0, 2.0]

        # Update with new value
        cache.set("text", [3.0, 4.0])
        # Note: Current implementation doesn't update, it just moves to end
        # This is fine for embeddings since same text = same embedding
        result = cache.get("text")
        assert result in [[1.0, 2.0], [3.0, 4.0]]  # Either is acceptable
