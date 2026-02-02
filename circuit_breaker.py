"""
Circuit Breaker - Prevents cascading failures in distributed systems.
If a service keeps failing, we "open the circuit" and stop calling it.

States:
- CLOSED: Everything working normally, calls go through
- OPEN: Too many failures, all calls fail immediately
- HALF_OPEN: Testing if service recovered
"""

import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Working normally
    OPEN = "open"          # Broken, failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open (service unavailable)"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker: Prevent cascading failures in distributed systems.
    
    How it works:
    1. Start in CLOSED state (everything working)
    2. If failures reach threshold → OPEN (stop calling service)
    3. After timeout → HALF_OPEN (try one request)
    4. If HALF_OPEN request succeeds → CLOSED (recovered!)
    5. If HALF_OPEN request fails → OPEN again
    
    Example:
        >>> breaker = CircuitBreaker(max_failures=3, timeout=60)
        >>> result = breaker.call(lambda: api_call())
    """
    
    def __init__(self, max_failures: int = 3, timeout: int = 60):
        """
        Initialize circuit breaker.
        
        Args:
            max_failures: How many failures before opening circuit (default: 3)
            timeout: Seconds to wait before trying again (default: 60)
        """
        self.max_failures = max_failures
        self.timeout = timeout
        
        # State tracking
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func):
        """
        Call a function through the circuit breaker.
        
        Args:
            func: Function to call (should return a result or raise exception)
        
        Returns:
            Result from func if successful
        
        Raises:
            CircuitBreakerOpen: If circuit is open and timeout hasn't elapsed
            Exception: Whatever exception func raises
        """
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            # Has enough time passed to try again?
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                # Move to HALF_OPEN and try
                logger.info("Circuit breaker timeout expired, moving to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
            else:
                # Still in timeout, fail fast
                wait_time = self.timeout - (time.time() - self.last_failure_time) if self.last_failure_time else self.timeout
                raise CircuitBreakerOpen(
                    f"Circuit breaker open. Tried {self.failures} times. "
                    f"Wait {wait_time:.0f}s"
                )
        
        # Try the function call
        try:
            result = func()
            
            # Success! Reset failure counter
            if self.state == CircuitState.HALF_OPEN:
                # We recovered! Close the circuit
                logger.info("Circuit breaker recovered, moving to CLOSED")
                self.state = CircuitState.CLOSED
            
            self.failures = 0
            return result
            
        except Exception as e:
            # Failure! Increment counter
            self.failures += 1
            self.last_failure_time = time.time()
            
            # Check if we should open the circuit
            if self.failures >= self.max_failures:
                logger.warning(f"Circuit breaker opening after {self.failures} failures")
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                # Half-open test failed, reopen
                logger.warning("Circuit breaker half-open test failed, reopening")
                self.state = CircuitState.OPEN
            
            # Re-raise the exception
            raise e
    
    def reset(self):
        """
        Manually reset the circuit breaker.
        
        Use this when you know the service is back up and want to
        force the circuit closed.
        """
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        logger.info("Circuit breaker manually reset")
    
    def get_state(self) -> dict:
        """
        Get current circuit breaker state for monitoring.
        
        Returns:
            Dict with state, failures, and time since last failure
        """
        return {
            'state': self.state.value,
            'failures': self.failures,
            'max_failures': self.max_failures,
            'timeout': self.timeout,
            'time_since_last_failure': (
                time.time() - self.last_failure_time 
                if self.last_failure_time else None
            )
        }
