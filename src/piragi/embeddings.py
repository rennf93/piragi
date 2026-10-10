"""Embedding generation using local or remote models."""

import os
from collections.abc import Callable
from typing import Any

from .cache import EmbeddingCache
from .types import Chunk


class EmbeddingGenerator:
    """Generate embeddings using local sentence-transformers or remote API."""

    def __init__(
        self,
        model: str = "nvidia/llama-embed-nemotron-8b",
        device: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 32,
        cache: EmbeddingCache | None = None,
        cache_enabled: bool = False,
        cache_max_size: int = 10000,
    ) -> None:
        """
        Initialize the embedding generator.

        Args:
            model: Embedding model to use (default: nvidia/llama-embed-nemotron-8b)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect) - only for local models
            base_url: Optional API base URL for remote embeddings (e.g., https://api.openai.com/v1)
            api_key: Optional API key for remote embeddings
            batch_size: Number of texts to embed in a single batch (default: 32)
            cache: Optional pre-configured EmbeddingCache instance to use
            cache_enabled: Whether to enable embedding caching (default: False)
            cache_max_size: Maximum number of embeddings to cache (default: 10000)
        """  # noqa: E501
        self.model_name = model
        self.base_url = base_url
        self.api_key = api_key
        self.use_remote = base_url is not None
        self.batch_size = batch_size

        # Initialize cache
        self._cache: EmbeddingCache | None = None
        if cache is not None:
            self._cache = cache
        elif cache_enabled:
            self._cache = EmbeddingCache(max_size=cache_max_size)

        # Initialize model/client (one will be set, the other None)
        self.model: Any = None
        self.client: Any = None

        if self.use_remote:
            # Use OpenAI-compatible API client
            from openai import OpenAI

            if self.api_key is None:
                self.api_key = os.getenv("EMBEDDING_API_KEY", "not-needed")

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            # Use local sentence-transformers
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model,
                trust_remote_code=True,
                device=device,
            )

    def embed_chunks(
        self,
        chunks: list[Chunk],
        on_progress: Callable[[str], None] | None = None,
    ) -> list[Chunk]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of chunks to embed
            on_progress: Optional callback for progress updates

        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks

        # Extract texts
        texts = [chunk.text for chunk in chunks]
        all_embeddings = []
        total = len(texts)

        # Process in batches for memory efficiency and progress reporting
        for i in range(0, total, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_embeddings = self._generate_embeddings(batch_texts, batch_size=self.batch_size)
            all_embeddings.extend(batch_embeddings)

            # Report progress
            if on_progress:
                completed = min(i + self.batch_size, total)
                on_progress(f"Embedded {completed}/{total} chunks")

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings, strict=False):
            # Handle both numpy arrays (local) and lists (remote/Ollama)
            if hasattr(embedding, "tolist"):
                chunk.embedding = embedding.tolist()
            else:
                chunk.embedding = embedding

        return chunks

    def _generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        try:
            if self.use_remote:
                # Use OpenAI-compatible API
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model_name,
                )
                return [item.embedding for item in response.data]
            else:
                # Use local sentence-transformers
                # Use encode_document for document chunks if available
                if hasattr(self.model, "encode_document"):
                    embeddings = self.model.encode_document(texts, batch_size=batch_size)
                else:
                    embeddings = self.model.encode(texts, batch_size=batch_size)
                # Convert to list of lists if needed (numpy array -> list)
                result: list[list[float]]
                result = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
                return result

        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    def get_dimensions(self) -> int:
        """
        Get the embedding dimensions by inferring from a test embedding.

        This is useful when you need to know the vector dimensions without
        hardcoding them, e.g., for initializing a vector store.

        Returns:
            Number of dimensions in the embedding vector

        Examples:
            >>> embedder = EmbeddingGenerator(model="all-mpnet-base-v2")
            >>> dims = embedder.get_dimensions()
            >>> print(dims)  # 768
        """
        test_embedding = self.embed_query("dimension test")
        return len(test_embedding)

    @property
    def cache(self) -> EmbeddingCache | None:
        """Return the cache instance if caching is enabled."""
        return self._cache

    @property
    def cache_stats(self) -> dict | None:
        """Return cache statistics if caching is enabled.

        Returns:
            Dictionary with cache statistics or None if caching is disabled
        """
        if self._cache is not None:
            return self._cache.stats
        return None

    def embed_query(self, query: str, task_instruction: str | None = None) -> list[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text
            task_instruction: Optional task instruction for query
                (e.g., "Retrieve relevant documents for this question")

        Returns:
            Embedding vector
        """
        # Build cache key (include instruction if present)
        cache_key = f"{task_instruction}:{query}" if task_instruction else query

        # Check cache first
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            embedding: list[float]
            if self.use_remote:
                # Use OpenAI-compatible API
                query_text = query
                if task_instruction:
                    query_text = f"{task_instruction}\n{query}"

                response = self.client.embeddings.create(
                    input=query_text,
                    model=self.model_name,
                )
                embedding = response.data[0].embedding
            else:
                # Use local sentence-transformers
                # Use encode_query for search queries if available
                if hasattr(self.model, "encode_query"):
                    if task_instruction:
                        query_with_instruction = f"{task_instruction}\n{query}"
                        raw_embedding = self.model.encode_query(query_with_instruction)
                    else:
                        raw_embedding = self.model.encode_query(query)
                else:
                    query_text = query
                    if task_instruction:
                        query_text = f"{task_instruction}\n{query}"
                    raw_embedding = self.model.encode(query_text)

                # Handle both numpy arrays and lists
                if hasattr(raw_embedding, "tolist"):
                    embedding = raw_embedding.tolist()
                else:
                    embedding = raw_embedding

            # Cache the result
            if self._cache is not None:
                self._cache.set(cache_key, embedding)

            return embedding

        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {e}") from e
