"""Shared retry utilities for adapters."""

import asyncio
import logging
from typing import Callable, TypeVar, Any

T = TypeVar('T')


async def retry_with_backoff(
    coro_fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_check: Callable[[Exception], bool] | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """
    Execute async function with exponential backoff retry.

    Args:
        coro_fn: Async function to execute (no args, use closure/partial)
        max_retries: Maximum number of attempts
        base_delay: Base delay in seconds (doubles each retry)
        retryable_check: Function to check if exception is retryable
        logger: Logger for retry messages

    Returns:
        Result from successful coro_fn call

    Raises:
        Last exception if all retries exhausted
    """
    log = logger or logging.getLogger(__name__)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            last_error = e
            error_name = type(e).__name__

            # Check if error is retryable
            if retryable_check and not retryable_check(e):
                raise

            log.warning(f"{error_name}: {e} (attempt {attempt + 1}/{max_retries})")

            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)
                log.info(f"Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

    raise last_error  # type: ignore


def is_auth_error(e: Exception) -> bool:
    """Check if exception is an authentication error (not retryable)."""
    error_name = type(e).__name__
    return "AuthenticationError" in error_name or "authentication" in str(e).lower()
