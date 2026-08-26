"""Tests for retry utilities."""

import time

import pytest

from piragi.retry import (
    RetryConfig,
    is_retriable,
    retry_async,
    retry_sync,
)


class TestIsRetriable:
    """Tests for is_retriable function."""

    def test_connection_errors(self) -> None:
        """Test that connection errors are retriable."""
        assert is_retriable(Exception("Connection reset by peer"))
        assert is_retriable(Exception("Connection refused"))
        assert is_retriable(Exception("connection closed unexpectedly"))

    def test_timeout_errors(self) -> None:
        """Test that timeout errors are retriable."""
        assert is_retriable(Exception("Request timed out"))
        assert is_retriable(Exception("timeout waiting for response"))

    def test_rate_limit_errors(self) -> None:
        """Test that rate limit errors are retriable."""
        assert is_retriable(Exception("Rate limit exceeded"))
        assert is_retriable(Exception("HTTP 429: Too many requests"))

    def test_server_errors(self) -> None:
        """Test that server errors are retriable."""
        assert is_retriable(Exception("503 Service Unavailable"))
        assert is_retriable(Exception("504 Gateway Timeout"))
        assert is_retriable(Exception("Server is overloaded"))

    def test_non_retriable_errors(self) -> None:
        """Test that non-retriable errors are not marked as retriable."""
        assert not is_retriable(Exception("Invalid API key"))
        assert not is_retriable(Exception("File not found"))
        assert not is_retriable(Exception("Permission denied"))
        assert not is_retriable(ValueError("Invalid input"))

    def test_custom_patterns(self) -> None:
        """Test custom retriable patterns."""
        custom_patterns = ("custom error", "special case")
        assert is_retriable(Exception("custom error occurred"), custom_patterns)
        assert is_retriable(Exception("this is a special case"), custom_patterns)
        assert not is_retriable(Exception("normal error"), custom_patterns)


class TestRetrySyncDecorator:
    """Tests for retry_sync decorator."""

    def test_success_on_first_try(self) -> None:
        """Test that successful calls don't retry."""
        call_count = 0

        @retry_sync(max_retries=3)
        def succeeds() -> str | None:
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_retriable_error(self) -> None:
        """Test that retriable errors trigger retries."""
        call_count = 0

        @retry_sync(max_retries=3, base_delay=0.01)
        def fails_then_succeeds() -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Connection reset")
            return "success"

        result = fails_then_succeeds()
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_retriable_error(self) -> None:
        """Test that non-retriable errors don't trigger retries."""
        call_count = 0

        @retry_sync(max_retries=3)
        def fails_with_bad_error() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError, match="Invalid input"):
            fails_with_bad_error()
        assert call_count == 1

    def test_max_retries_exceeded(self) -> None:
        """Test that max retries is respected."""
        call_count = 0

        @retry_sync(max_retries=2, base_delay=0.01)
        def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise Exception("Connection timeout")

        with pytest.raises(Exception, match="Connection timeout"):
            always_fails()
        assert call_count == 3  # Initial + 2 retries

    def test_exponential_backoff(self) -> None:
        """Test that delay increases exponentially."""
        delays = []
        start_time = time.time()

        @retry_sync(max_retries=3, base_delay=0.05, max_delay=1.0)
        def fails_with_timing() -> None:
            delays.append(time.time() - start_time)
            raise TimeoutError("timeout")

        with pytest.raises(TimeoutError):
            fails_with_timing()

        # Check that delays roughly follow exponential pattern
        # delay[1] - delay[0] should be ~0.05
        # delay[2] - delay[1] should be ~0.1
        # delay[3] - delay[2] should be ~0.2
        assert len(delays) == 4

    def test_on_retry_callback(self) -> None:
        """Test that on_retry callback is called."""
        retry_calls = []

        def on_retry(error: Exception, attempt: int) -> None:
            retry_calls.append((str(error), attempt))

        @retry_sync(max_retries=2, base_delay=0.01, on_retry=on_retry)
        def fails_twice() -> None:
            raise TimeoutError("timeout")

        with pytest.raises(TimeoutError):
            fails_twice()

        assert len(retry_calls) == 2
        assert retry_calls[0] == ("timeout", 0)
        assert retry_calls[1] == ("timeout", 1)


class TestRetryAsyncDecorator:
    """Tests for retry_async decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self) -> None:
        """Test that successful async calls don't retry."""
        call_count = 0

        @retry_async(max_retries=3)
        async def succeeds() -> str | None:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await succeeds()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retriable_error(self) -> None:
        """Test that retriable errors trigger retries in async."""
        call_count = 0

        @retry_async(max_retries=3, base_delay=0.01)
        async def fails_then_succeeds() -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Connection reset")
            return "success"

        result = await fails_then_succeeds()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retriable_error(self) -> None:
        """Test that non-retriable errors don't trigger retries in async."""
        call_count = 0

        @retry_async(max_retries=3)
        async def fails_with_bad_error() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError, match="Invalid input"):
            await fails_with_bad_error()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """Test that max retries is respected in async."""
        call_count = 0

        @retry_async(max_retries=2, base_delay=0.01)
        async def always_fails() -> None:
            nonlocal call_count
            call_count += 1
            raise Exception("Connection timeout")

        with pytest.raises(Exception, match="Connection timeout"):
            await always_fails()
        assert call_count == 3  # Initial + 2 retries


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_config_initialization(self) -> None:
        """Test RetryConfig initialization."""
        config = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
        )
        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0

    def test_config_wrap_sync(self) -> None:
        """Test wrapping sync function with RetryConfig."""
        config = RetryConfig(max_retries=2, base_delay=0.01)
        call_count = 0

        def fails_then_succeeds() -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("timeout")
            return "success"

        wrapped = config.wrap_sync(fails_then_succeeds)
        result = wrapped()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_config_wrap_async(self) -> None:
        """Test wrapping async function with RetryConfig."""
        config = RetryConfig(max_retries=2, base_delay=0.01)
        call_count = 0

        async def fails_then_succeeds() -> str | None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("timeout")
            return "success"

        wrapped = config.wrap_async(fails_then_succeeds)
        result = await wrapped()
        assert result == "success"
        assert call_count == 2

    def test_config_repr(self) -> None:
        """Test RetryConfig string representation."""
        config = RetryConfig(max_retries=5, base_delay=1.0, max_delay=30.0)
        repr_str = repr(config)
        assert "RetryConfig" in repr_str
        assert "max_retries=5" in repr_str
        assert "base_delay=1.0" in repr_str
        assert "max_delay=30.0" in repr_str
