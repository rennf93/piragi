"""Tests for core Ragi class."""

import os
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from piragi import Ragi
from piragi.types import Answer, Chunk


@pytest.fixture
def mock_embeddings() -> list[float]:
    """Mock embedding responses (768-dim for all-mpnet-base-v2)."""
    return [0.1] * 768


@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response."""
    return "This is a test answer based on the provided context."


def create_mock_embedding_generator(mock_embeddings: list[float]) -> MagicMock:
    """Create a mock EmbeddingGenerator for testing."""
    mock_gen = MagicMock()

    def embed_chunks_side_effect(
        chunks: list[Chunk], on_progress: Callable[[str], None] | None = None
    ) -> list[Chunk]:
        for chunk in chunks:
            chunk.embedding = mock_embeddings
        return chunks

    mock_gen.embed_chunks.side_effect = embed_chunks_side_effect
    mock_gen.embed_query.return_value = mock_embeddings
    return mock_gen


class TestRagiInit:
    """Tests for Ragi initialization."""

    def test_init_without_sources(self, temp_dir: str) -> None:
        """Test initialization without sources."""
        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)

        assert kb.store.count() == 0

    @patch("piragi.retrieval.OpenAI")
    def test_init_with_config(self, mock_openai: MagicMock, temp_dir: str) -> None:
        """Test initialization with custom config."""
        persist_dir = os.path.join(temp_dir, "test_ragi")
        Ragi(persist_dir=persist_dir, config={"llm": {"model": "gpt-4", "api_key": "custom-key"}})

        # Verify OpenAI was initialized
        mock_openai.assert_called()


class TestRagiAdd:
    """Tests for adding documents."""

    @patch("piragi.core.EmbeddingGenerator")
    def test_add_single_file(
        self,
        mock_embed_gen: MagicMock,
        temp_dir: str,
        sample_text_file: str,
        mock_embeddings: list[float],
    ) -> None:
        """Test adding a single file."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        assert kb.count() > 0

    @patch("piragi.core.EmbeddingGenerator")
    def test_add_multiple_files(
        self,
        mock_embed_gen: MagicMock,
        temp_dir: str,
        sample_text_file: str,
        sample_markdown_file: str,
        mock_embeddings: list[float],
    ):
        """Test adding multiple files."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add([sample_text_file, sample_markdown_file])

        assert kb.count() > 0

    @patch("piragi.core.EmbeddingGenerator")
    def test_add_returns_self(
        self,
        mock_embed_gen: MagicMock,
        temp_dir: str,
        sample_text_file: str,
        mock_embeddings: list[float],
    ) -> None:
        """Test that add() returns self for chaining."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        result = kb.add(sample_text_file)

        assert result is kb


class TestRagiQuery:
    """Tests for querying."""

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_ask_question(
        self,
        mock_embed_gen: MagicMock,
        mock_openai: MagicMock,
        temp_dir: str,
        sample_text_file: str,
        mock_embeddings: list[float],
        mock_llm_response: str,
    ) -> None:
        """Test asking a question."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        # Mock OpenAI response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        answer = kb.ask("What is this document about?")

        assert isinstance(answer, Answer)
        assert answer.text
        assert answer.query == "What is this document about?"

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_callable_interface(
        self,
        mock_embed_gen: MagicMock,
        mock_openai: MagicMock,
        temp_dir: str,
        sample_text_file: str,
        mock_embeddings: list[float],
        mock_llm_response: str,
    ) -> None:
        """Test using Ragi as callable."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        # Mock LLM response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        answer = kb("What is this?")

        assert isinstance(answer, Answer)


class TestRagiFilter:
    """Tests for metadata filtering."""

    def test_filter_returns_self(self, temp_dir: str) -> None:
        """Test that filter() returns self for chaining."""
        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        result = kb.filter(type="test")

        assert result is kb

    @patch("piragi.retrieval.OpenAI")
    @patch("piragi.core.EmbeddingGenerator")
    def test_filter_chaining(
        self,
        mock_embed_gen: MagicMock,
        mock_openai: MagicMock,
        temp_dir: str,
        sample_text_file: str,
        mock_embeddings: list[float],
        mock_llm_response: str,
    ) -> None:
        """Test filter chaining with ask."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        # Mock LLM response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        # Filter by file_type which is added by the loader
        answer = kb.filter(file_type="txt").ask("What is this?")

        assert isinstance(answer, Answer)


class TestRagiEmbedderInjection:
    """Tests for custom embedder injection."""

    def test_init_with_custom_embedder(self, temp_dir: str, mock_embeddings: list[float]) -> None:
        """Test initialization with custom embedder."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir, embedder=mock_embedder)

        # Verify custom embedder is used
        assert kb.embedder is mock_embedder

    def test_custom_embedder_used_for_add(
        self, temp_dir: str, sample_text_file: str, mock_embeddings: list[float]
    ) -> None:
        """Test that custom embedder is used when adding documents."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir, embedder=mock_embedder)
        kb.add(sample_text_file)

        # Verify embedder's embed_chunks was called
        mock_embedder.embed_chunks.assert_called()
        assert kb.count() > 0

    def test_shared_embedder_across_instances(
        self, temp_dir: str, mock_embeddings: list[float]
    ) -> None:
        """Test that same embedder can be shared across multiple Ragi instances."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)

        persist_dir1 = os.path.join(temp_dir, "test_ragi1")
        persist_dir2 = os.path.join(temp_dir, "test_ragi2")

        kb1 = Ragi(persist_dir=persist_dir1, embedder=mock_embedder)
        kb2 = Ragi(persist_dir=persist_dir2, embedder=mock_embedder)

        # Both instances should share the same embedder
        assert kb1.embedder is kb2.embedder
        assert kb1.embedder is mock_embedder

    def test_custom_embedder_overrides_config(
        self, temp_dir: str, mock_embeddings: list[float]
    ) -> None:
        """Test that custom embedder takes precedence over config."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)
        mock_embedder.model_name = "custom-model"

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(
            persist_dir=persist_dir,
            embedder=mock_embedder,
            config={"embedding": {"model": "different-model"}},
        )

        # Custom embedder should be used, not the one from config
        assert kb.embedder is mock_embedder

    @patch("piragi.retrieval.OpenAI")
    def test_custom_embedder_used_for_query(
        self,
        mock_openai: MagicMock,
        temp_dir: str,
        sample_text_file: str,
        mock_embeddings: list[float],
        mock_llm_response: str,
    ) -> None:
        """Test that custom embedder is used for query embedding."""
        mock_embedder = create_mock_embedding_generator(mock_embeddings)

        # Mock LLM response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_response))]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir, embedder=mock_embedder)
        kb.add(sample_text_file)

        kb.ask("What is this?")

        # Verify embed_query was called for the search
        mock_embedder.embed_query.assert_called()


class TestRagiUtility:
    """Tests for utility methods."""

    def test_count_empty(self, temp_dir: str) -> None:
        """Test count on empty store."""
        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)

        assert kb.count() == 0

    @patch("piragi.core.EmbeddingGenerator")
    def test_clear(
        self,
        mock_embed_gen: MagicMock,
        temp_dir: str,
        sample_text_file: str,
        mock_embeddings: list[float],
    ) -> None:
        """Test clearing the knowledge base."""
        mock_embed_gen.return_value = create_mock_embedding_generator(mock_embeddings)

        persist_dir = os.path.join(temp_dir, "test_ragi")
        kb = Ragi(persist_dir=persist_dir)
        kb.add(sample_text_file)

        assert kb.count() > 0

        kb.clear()
        assert kb.count() == 0
