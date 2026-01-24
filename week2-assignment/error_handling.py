"""
Error Handling & Retry Logic - Task 3
Comprehensive error handling with exponential backoff retries.

This module provides:
1. Custom exception classes for different error categories
2. retry_with_backoff decorator with exponential backoff
3. Error context tracking and logging
4. Error recovery mechanisms
"""

import time
import logging
import functools
import traceback
from typing import Callable, Optional, Tuple, Type, Any
from enum import Enum
from datetime import datetime


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTION CLASSES
# ════════════════════════════════════════════════════════════════════════════════

class AgentException(Exception):
    """Base exception class for all agent-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        context: Optional[dict] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize AgentException.
        
        Args:
            message: Error message
            error_code: Unique error identifier
            context: Additional context (user_id, request_id, etc.)
            timestamp: When error occurred
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = timestamp or datetime.utcnow()
        super().__init__(message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class RateLimitError(AgentException):
    """
    Raised when user exceeds rate limit.
    Includes retry-after information.
    """
    
    def __init__(
        self,
        message: str,
        retry_after_seconds: int,
        user_id: Optional[str] = None,
        context: Optional[dict] = None
    ):
        """
        Initialize RateLimitError.
        
        Args:
            message: Error message
            retry_after_seconds: Seconds until next request allowed
            user_id: User identifier
            context: Additional context
        """
        context = context or {}
        context["user_id"] = user_id
        context["retry_after_seconds"] = retry_after_seconds
        
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED", context=context)
        self.retry_after_seconds = retry_after_seconds


class MaxIterationsError(AgentException):
    """
    Raised when graph execution exceeds max iterations.
    Indicates possible infinite loop.
    """
    
    def __init__(
        self,
        message: str,
        max_iterations: int,
        current_iteration: int,
        context: Optional[dict] = None
    ):
        """
        Initialize MaxIterationsError.
        
        Args:
            message: Error message
            max_iterations: Maximum allowed iterations
            current_iteration: Current iteration when limit hit
            context: Additional context
        """
        context = context or {}
        context["max_iterations"] = max_iterations
        context["current_iteration"] = current_iteration
        
        super().__init__(
            message,
            error_code="MAX_ITERATIONS_EXCEEDED",
            context=context
        )
        self.max_iterations = max_iterations
        self.current_iteration = current_iteration


class ToolExecutionError(AgentException):
    """
    Raised when a tool fails during execution.
    Includes tool name and execution details.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        original_error: Optional[Exception] = None,
        context: Optional[dict] = None
    ):
        """
        Initialize ToolExecutionError.
        
        Args:
            message: Error message
            tool_name: Name of failed tool
            original_error: Original exception
            context: Additional context
        """
        context = context or {}
        context["tool_name"] = tool_name
        if original_error:
            context["original_error"] = str(original_error)
            context["error_type"] = type(original_error).__name__
        
        super().__init__(
            message,
            error_code="TOOL_EXECUTION_FAILED",
            context=context
        )
        self.tool_name = tool_name
        self.original_error = original_error


class InvalidOutputError(AgentException):
    """
    Raised when LLM output doesn't match expected format.
    Includes validation details.
    """
    
    def __init__(
        self,
        message: str,
        expected_format: str,
        actual_output: Optional[str] = None,
        validation_error: Optional[str] = None,
        context: Optional[dict] = None
    ):
        """
        Initialize InvalidOutputError.
        
        Args:
            message: Error message
            expected_format: Expected output format
            actual_output: What we got instead
            validation_error: Pydantic/validation error details
            context: Additional context
        """
        context = context or {}
        context["expected_format"] = expected_format
        context["validation_error"] = validation_error
        
        super().__init__(
            message,
            error_code="INVALID_OUTPUT_FORMAT",
            context=context
        )
        self.expected_format = expected_format
        self.actual_output = actual_output
        self.validation_error = validation_error


class LLMError(AgentException):
    """
    Raised when LLM call fails.
    Includes API error details.
    """
    
    def __init__(
        self,
        message: str,
        error_type: str = "API_ERROR",
        status_code: Optional[int] = None,
        api_error: Optional[str] = None,
        context: Optional[dict] = None
    ):
        """
        Initialize LLMError.
        
        Args:
            message: Error message
            error_type: Type of LLM error (API_ERROR, TIMEOUT, INVALID_REQUEST, etc.)
            status_code: HTTP status code if applicable
            api_error: Original API error response
            context: Additional context
        """
        context = context or {}
        context["error_type"] = error_type
        if status_code:
            context["status_code"] = status_code
        if api_error:
            context["api_error"] = api_error
        
        super().__init__(message, error_code="LLM_CALL_FAILED", context=context)
        self.error_type = error_type
        self.status_code = status_code
        self.api_error = api_error


class TimeoutError(AgentException):
    """Raised when operation exceeds timeout."""
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        operation: str = "unknown",
        context: Optional[dict] = None
    ):
        """
        Initialize TimeoutError.
        
        Args:
            message: Error message
            timeout_seconds: Timeout duration
            operation: What operation timed out
            context: Additional context
        """
        context = context or {}
        context["timeout_seconds"] = timeout_seconds
        context["operation"] = operation
        
        super().__init__(message, error_code="TIMEOUT", context=context)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class ValidationError(AgentException):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_errors: list,
        context: Optional[dict] = None
    ):
        """
        Initialize ValidationError.
        
        Args:
            message: Error message
            validation_errors: List of validation errors
            context: Additional context
        """
        context = context or {}
        context["validation_errors"] = validation_errors
        
        super().__init__(message, error_code="VALIDATION_FAILED", context=context)
        self.validation_errors = validation_errors


# ════════════════════════════════════════════════════════════════════════════════
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# ════════════════════════════════════════════════════════════════════════════════

class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    jitter: bool = True,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exception types to catch and retry
        strategy: Retry strategy (exponential, linear, fixed)
        jitter: Add randomness to delay to prevent thundering herd
        on_retry: Optional callback function(exception, attempt, delay)
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def call_llm():
            return llm.invoke(...)
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    # Last attempt, don't retry
                    if attempt == max_retries:
                        logger.error(
                            f"[RETRY EXHAUSTED] {func.__name__} failed after {max_retries} retries",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "error": str(e),
                                "error_type": type(e).__name__
                            }
                        )
                        raise
                    
                    # Calculate delay for next retry
                    if strategy == RetryStrategy.EXPONENTIAL:
                        next_delay = min(current_delay * backoff_factor, max_delay)
                    elif strategy == RetryStrategy.LINEAR:
                        next_delay = min(current_delay + initial_delay, max_delay)
                    else:  # FIXED
                        next_delay = current_delay
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        import random
                        jitter_amount = random.uniform(0, next_delay * 0.1)
                        next_delay += jitter_amount
                    
                    # Log retry attempt
                    logger.warning(
                        f"[RETRY {attempt + 1}/{max_retries}] {func.__name__} failed, "
                        f"retrying in {next_delay:.2f}s: {str(e)[:100]}",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "next_delay": next_delay,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1, next_delay)
                        except Exception as callback_error:
                            logger.error(
                                f"[RETRY CALLBACK ERROR] {callback_error}",
                                extra={"function": func.__name__}
                            )
                    
                    # Wait before retrying
                    time.sleep(next_delay)
                    current_delay = next_delay
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS & UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

class ErrorContext:
    """Context manager for tracking error information."""
    
    def __init__(
        self,
        operation: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        Initialize ErrorContext.
        
        Args:
            operation: What operation is being performed
            user_id: User identifier
            request_id: Request identifier
        """
        self.operation = operation
        self.user_id = user_id
        self.request_id = request_id
        self.start_time = datetime.utcnow()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            elapsed_ms = (datetime.utcnow() - self.start_time).total_seconds() * 1000
            logger.error(
                f"[ERROR CONTEXT] {self.operation} failed: {exc_val}",
                extra={
                    "operation": self.operation,
                    "user_id": self.user_id,
                    "request_id": self.request_id,
                    "error_type": exc_type.__name__,
                    "elapsed_ms": elapsed_ms
                }
            )
        return False


def handle_rate_limit_error(error: RateLimitError, user_id: str) -> dict:
    """
    Handle rate limit error with graceful degradation.
    
    Args:
        error: RateLimitError instance
        user_id: User identifier
    
    Returns:
        Error response with retry-after information
    """
    logger.warning(
        f"[RATE LIMIT] User {user_id} exceeded rate limit",
        extra={
            "user_id": user_id,
            "retry_after_seconds": error.retry_after_seconds,
            "error_code": error.error_code
        }
    )
    
    return {
        "success": False,
        "error": "RATE_LIMIT_EXCEEDED",
        "message": f"Rate limit exceeded. Please try again in {error.retry_after_seconds} seconds.",
        "retry_after_seconds": error.retry_after_seconds,
        "timestamp": datetime.utcnow().isoformat()
    }


def handle_llm_error(error: LLMError, operation: str) -> dict:
    """
    Handle LLM error with fallback response.
    
    Args:
        error: LLMError instance
        operation: What operation was being performed
    
    Returns:
        Error response with fallback suggestion
    """
    logger.error(
        f"[LLM ERROR] {operation} failed: {error.message}",
        extra={
            "operation": operation,
            "error_type": error.error_type,
            "status_code": error.status_code,
            "error_code": error.error_code
        }
    )
    
    # Return fallback response
    return {
        "success": False,
        "error": "LLM_CALL_FAILED",
        "message": f"Unable to process your request. Please try again.",
        "operation": operation,
        "error_code": error.error_code,
        "timestamp": datetime.utcnow().isoformat()
    }


def handle_tool_execution_error(error: ToolExecutionError) -> dict:
    """
    Handle tool execution error.
    
    Args:
        error: ToolExecutionError instance
    
    Returns:
        Error response with tool information
    """
    logger.error(
        f"[TOOL ERROR] Tool {error.tool_name} failed: {error.message}",
        extra={
            "tool_name": error.tool_name,
            "original_error": str(error.original_error),
            "error_code": error.error_code
        }
    )
    
    return {
        "success": False,
        "error": "TOOL_EXECUTION_FAILED",
        "message": f"Tool {error.tool_name} failed. Please try again.",
        "tool_name": error.tool_name,
        "timestamp": datetime.utcnow().isoformat()
    }


def handle_max_iterations_error(error: MaxIterationsError) -> dict:
    """
    Handle max iterations error.
    
    Args:
        error: MaxIterationsError instance
    
    Returns:
        Error response with iteration information
    """
    logger.error(
        f"[MAX ITERATIONS] Graph exceeded max iterations: {error.message}",
        extra={
            "max_iterations": error.max_iterations,
            "current_iteration": error.current_iteration,
            "error_code": error.error_code
        }
    )
    
    return {
        "success": False,
        "error": "MAX_ITERATIONS_EXCEEDED",
        "message": f"Request processing exceeded maximum iterations ({error.max_iterations}). "
                   f"Possible infinite loop detected.",
        "max_iterations": error.max_iterations,
        "current_iteration": error.current_iteration,
        "timestamp": datetime.utcnow().isoformat()
    }


def log_error_chain(error: Exception, context: Optional[dict] = None) -> None:
    """
    Log full error chain with traceback.
    
    Args:
        error: Exception to log
        context: Additional context information
    """
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        **(context or {})
    }
    
    logger.error(
        f"[ERROR CHAIN] {error_info['error_type']}: {error_info['error_message']}",
        extra=error_info
    )
