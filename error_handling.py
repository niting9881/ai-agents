"""
Error Handling - Retry logic with exponential backoff.
Handles transient failures gracefully using LangChain patterns.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Tuple

# LangChain exceptions (optional - only import if available)
try:
    from langchain_core.exceptions import (
        LangChainException,
        OutputParserException,
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback if LangChain not installed
    LangChainException = Exception
    OutputParserException = ValueError
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    
    Example:
        @retry_with_backoff(
            max_retries=3, 
            initial_delay=1.0,
            exceptions=(LangChainException,)
        )
        def call_llm(messages):
            return llm.invoke(messages)
    
    Note: This works with LangChain's exception hierarchy. For production,
    consider using LangChain's built-in retry mechanism via RunnableRetry.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries",
                            exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s: {str(e)}"
                    )
                    
                    time.sleep(delay)
                    delay *= backoff_factor
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise Exception("Retry logic failed unexpectedly")
        
        return wrapper
    return decorator
