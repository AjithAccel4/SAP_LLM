"""
Timing utilities for SAP_LLM.

Provides decorators and context managers for timing code execution.
"""

import time
from functools import wraps
from typing import Callable, Optional

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class Timer:
    """
    Context manager for timing code blocks.

    Example:
        >>> with Timer("Processing document"):
        ...     # Do some work
        ...     time.sleep(1)
        Processing document took 1.0023s
    """

    def __init__(self, name: str = "Operation", logger_instance: Optional = None):
        """
        Initialize timer.

        Args:
            name: Name of operation being timed
            logger_instance: Logger to use (defaults to module logger)
        """
        self.name = name
        self.logger = logger_instance or logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        """Start timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Stop timer and log elapsed time."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        self.logger.info(f"{self.name} took {self.elapsed:.4f}s")

    def __str__(self) -> str:
        """String representation."""
        if self.elapsed is not None:
            return f"{self.name}: {self.elapsed:.4f}s"
        return f"{self.name}: not completed"


def timer(func: Optional[Callable] = None, name: Optional[str] = None) -> Callable:
    """
    Decorator for timing function execution.

    Args:
        func: Function to decorate
        name: Custom name for the operation (defaults to function name)

    Example:
        >>> @timer
        ... def process_document(doc_id):
        ...     time.sleep(1)
        ...     return doc_id
        >>> process_document("123")
        process_document took 1.0023s
        '123'
    """

    def decorator(f: Callable) -> Callable:
        operation_name = name or f.__name__

        @wraps(f)
        def wrapper(*args, **kwargs):
            with Timer(operation_name):
                return f(*args, **kwargs)

        return wrapper

    # Handle both @timer and @timer() usage
    if func is None:
        return decorator
    else:
        return decorator(func)


def async_timer(func: Optional[Callable] = None, name: Optional[str] = None) -> Callable:
    """
    Decorator for timing async function execution.

    Args:
        func: Async function to decorate
        name: Custom name for the operation (defaults to function name)

    Example:
        >>> @async_timer
        ... async def process_document(doc_id):
        ...     await asyncio.sleep(1)
        ...     return doc_id
        >>> await process_document("123")
        process_document took 1.0023s
        '123'
    """

    def decorator(f: Callable) -> Callable:
        operation_name = name or f.__name__

        @wraps(f)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return await f(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                logger.info(f"{operation_name} took {elapsed:.4f}s")

        return wrapper

    # Handle both @async_timer and @async_timer() usage
    if func is None:
        return decorator
    else:
        return decorator(func)
