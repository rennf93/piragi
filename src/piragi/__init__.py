"""
piragi - Zero-setup RAG library with auto-chunking, embeddings, and smart citations.

Example:
    >>> from piragi import Ragi
    >>>
    >>> # One-liner setup and query
    >>> kb = Ragi("./docs")
    >>> answer = kb.ask("How do I install this?")
    >>>
    >>> # Access answer and citations
    >>> print(answer.text)
    >>> for citation in answer.citations:
    ...     print(f"Source: {citation.source}")
    ...     print(f"Relevance: {citation.score:.2f}")
    >>>
    >>> # Callable shorthand
    >>> answer = kb("What's the API?")
    >>>
    >>> # Filter by metadata
    >>> answer = kb.filter(type="documentation").ask("How to configure?")
    >>>
    >>> # Advanced retrieval with HyDE, hybrid search, and cross-encoder
    >>> kb = Ragi("./docs", config={
    ...     "retrieval": {
    ...         "use_hyde": True,
    ...         "use_hybrid_search": True,
    ...         "use_cross_encoder": True,
    ...     }
    ... })
"""

from .async_ragi import AsyncRagi
from .cache import EmbeddingCache
from .core import Ragi
from .embeddings import EmbeddingGenerator
from .hybrid_search import BM25, HybridSearcher
from .query_transform import HyDE, MultiQueryRetriever, QueryExpander, StepBackPrompting

# Advanced components (optional imports)
from .reranker import CrossEncoderReranker, HybridReranker, TFIDFReranker
from .retry import RetryConfig, is_retriable, retry_async, retry_sync
from .semantic_chunking import (
    ContextualChunker,
    HierarchicalChunker,
    PropositionChunker,
    SemanticChunker,
)
from .stores import (
    LanceStore,
    PineconeStore,
    PostgresStore,
    VectorStoreProtocol,
)
from .types import Answer, Citation

__version__ = "0.7.9"
__all__ = [
    # Core
    "Ragi",
    "AsyncRagi",
    "Answer",
    "Citation",
    # Embeddings
    "EmbeddingGenerator",
    "EmbeddingCache",
    # Retry utilities
    "retry_sync",
    "retry_async",
    "RetryConfig",
    "is_retriable",
    # Vector stores
    "VectorStoreProtocol",
    "LanceStore",
    "PostgresStore",
    "PineconeStore",
    # Reranking
    "CrossEncoderReranker",
    "TFIDFReranker",
    "HybridReranker",
    # Hybrid search
    "BM25",
    "HybridSearcher",
    # Query transformation
    "HyDE",
    "QueryExpander",
    "MultiQueryRetriever",
    "StepBackPrompting",
    # Chunking strategies
    "SemanticChunker",
    "ContextualChunker",
    "PropositionChunker",
    "HierarchicalChunker",
]
