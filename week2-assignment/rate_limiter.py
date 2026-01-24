"""
Rate Limiter - Prevents abuse and DoS attacks.
Uses sliding window algorithm.
"""

import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter using sliding window algorithm.
    
    In production, this would use Redis. For demo, we use in-memory dict.
    """
    
    def __init__(self, use_redis: bool = False, redis_client=None):
        """
        Initialize rate limiter.
        
        Args:
            use_redis: If True, use Redis for storage
            redis_client: Redis client instance (if use_redis=True)
        """
        self.use_redis = use_redis
        self.redis_client = redis_client
        self._in_memory_store = {}  # Fallback for demo
    
    def check_rate_limit(
        self,
        user_id: str,
        max_requests: int = 10,
        window_seconds: int = 60
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if user is within rate limit.
        
        Args:
            user_id: User identifier
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            Tuple of (allowed, retry_after_seconds)
            If allowed=True, retry_after is None
            If allowed=False, retry_after is seconds until next request allowed
        """
        key = f"rate_limit:{user_id}"
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        if self.use_redis and self.redis_client:
            # Use Redis (production)
            # Remove old entries
            self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            request_count = self.redis_client.zcard(key)
            
            if request_count >= max_requests:
                # Get oldest request timestamp
                oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = int(oldest[0][1])
                    retry_after = oldest_time + window_seconds - current_time
                    return False, max(1, retry_after)
                return False, window_seconds
            
            # Add current request
            self.redis_client.zadd(key, {str(current_time): current_time})
            self.redis_client.expire(key, window_seconds)
            
            return True, None
        else:
            # Use in-memory store (demo)
            if key not in self._in_memory_store:
                self._in_memory_store[key] = []
            
            # Remove old entries
            self._in_memory_store[key] = [
                ts for ts in self._in_memory_store[key] 
                if ts > window_start
            ]
            
            # Check limit
            if len(self._in_memory_store[key]) >= max_requests:
                # Get oldest request
                oldest_time = min(self._in_memory_store[key])
                retry_after = oldest_time + window_seconds - current_time
                return False, max(1, retry_after)
            
            # Add current request
            self._in_memory_store[key].append(current_time)
            
            return True, None
