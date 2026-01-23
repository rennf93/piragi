"""Retry utilities for resilient API calls."""

import asyncio
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retriable error patterns
DEFAULT_RETRIABLE_PATTERNS: tuple[str, ...] = (
    "connection reset",
    "connection refused",
    "connection closed",
    "timeout",
    "timed out",
    "rate limit",
    "rate_limit",
    "429",
    "503",
    "504",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "server error",
    "internal server error",
    "network error",
    "network is unreachable",
    "broken pipe",
    "ssl error",
)


def is_retriable(error: Exception, patterns: tuple[str, ...] = DEFAULT_RETRIABLE_PATTERNS) -> bool:
    """Check if an error is retriable based on error message patterns.

    Args:
        error: The exception to check
        patterns: Tuple of lowercase patterns to match against error message

    Returns:
        True if the error matches any retriable pattern
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Check error message
    if any(p in error_str for p in patterns):
        return True

    # Check error type name
    return bool(any(p in error_type for p in patterns))


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retriable_patterns: tuple[str, ...] = DEFAULT_RETRIABLE_PATTERNS,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for synchronous retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 0.5)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        retriable_patterns: Tuple of error message patterns to retry on
        on_retry: Optional callback called before each retry with (error, attempt)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_sync(max_retries=3, base_delay=1.0)
        ... def fetch_data():
        ...     # May fail with transient errors
        ...     return api.get("/data")
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not is_retriable(e, retriable_patterns):
                        raise
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): "
                            f"{e}. Retrying in {delay:.1f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        time.sleep(delay)
            # Should always have an exception at this point
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


def retry_async(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retriable_patterns: tuple[str, ...] = DEFAULT_RETRIABLE_PATTERNS,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for async retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 0.5)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        retriable_patterns: Tuple of error message patterns to retry on
        on_retry: Optional callback called before each retry with (error, attempt)

    Returns:
        Decorated async function with retry logic

    Example:
        >>> @retry_async(max_retries=3, base_delay=1.0)
        ... async def fetch_data():
        ...     # May fail with transient errors
        ...     return await api.get("/data")
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not is_retriable(e, retriable_patterns):
                        raise
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): "
                            f"{e}. Retrying in {delay:.1f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        await asyncio.sleep(delay)
            # Should always have an exception at this point
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


class RetryConfig:
    """Configuration class for retry behavior.

    This allows sharing retry configuration across multiple operations.

    Example:
        >>> config = RetryConfig(max_retries=5, base_delay=1.0)
        >>> config.wrap_sync(my_function)("arg")
        >>> await config.wrap_async(my_async_function)("arg")
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        retriable_patterns: tuple[str, ...] = DEFAULT_RETRIABLE_PATTERNS,
    ) -> None:
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            retriable_patterns: Tuple of error message patterns to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retriable_patterns = retriable_patterns

    def wrap_sync(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap a synchronous function with retry logic.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function with retry logic
        """
        return retry_sync(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            retriable_patterns=self.retriable_patterns,
        )(func)

    def wrap_async(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap an async function with retry logic.

        Args:
            func: Async function to wrap

        Returns:
            Wrapped async function with retry logic
        """
        return retry_async(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            retriable_patterns=self.retriable_patterns,
        )(func)

    def __repr__(self) -> str:
        return (
            f"RetryConfig(max_retries={self.max_retries}, "
            f"base_delay={self.base_delay}, max_delay={self.max_delay})"
        )
