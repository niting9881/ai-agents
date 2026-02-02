"""
Cost Tracker - Tracks LLM usage and enforces budget limits.
Prevents runaway costs in production.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class CostTracker:
    """
    Tracks LLM costs and enforces daily budgets.
    
    In production, this would use Redis. For demo, we use in-memory dict.
    """
    
    # Pricing (adjust for your provider)
    PRICES = {
        "gpt-4o-mini": {
            "input": 0.15 / 1_000_000,   # $0.15 per 1M input tokens
            "output": 0.60 / 1_000_000    # $0.60 per 1M output tokens
        },
        "gpt-4o": {
            "input": 2.50 / 1_000_000,   # $2.50 per 1M input tokens
            "output": 10.00 / 1_000_000   # $10.00 per 1M output tokens
        }
    }
    # alternatively you can mention in env file and read it from there
    # Retrieve the pricing from the direct api call to the provider
    
    def __init__(self, use_redis: bool = False, redis_client=None):
        """
        Initialize cost tracker.
        
        Args:
            use_redis: If True, use Redis for storage (requires redis_client)
            redis_client: Redis client instance (if use_redis=True)
        """
        self.use_redis = use_redis
        self.redis_client = redis_client
        self._in_memory_store = {}  # Fallback for demo
    
    def track_llm_call(
        self,
        employee_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """
        Track cost and update user's daily total.
        
        Args:
            employee_id: Employee identifier
            model: LLM model name (e.g., "gpt-4o-mini")
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        
        Returns:
            Dictionary with call_cost, daily_total, and token counts
        """
        # Calculate cost
        if model not in self.PRICES:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.PRICES.keys())}")
        
        input_cost = input_tokens * self.PRICES[model]["input"]
        output_cost = output_tokens * self.PRICES[model]["output"]
        total_cost = input_cost + output_cost
        
        # Update user's usage
        today = datetime.utcnow().date().isoformat()
        user_key = f"cost:{employee_id}:{today}"
        
        if self.use_redis and self.redis_client:
            # Use Redis (production)
            cost_cents = int(total_cost * 100)
            new_total_cents = self.redis_client.incrby(user_key, cost_cents)
            self.redis_client.expire(user_key, 60 * 60 * 24 * 30)  # 30 days
            new_total = new_total_cents / 100
        else:
            # Use in-memory store (demo)
            if user_key not in self._in_memory_store:
                self._in_memory_store[user_key] = 0
            self._in_memory_store[user_key] += total_cost
            new_total = self._in_memory_store[user_key]
        
        return {
            "call_cost": total_cost,
            "daily_total": new_total,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    
    def check_budget(self, employee_id: str, daily_limit: float = 1.0) -> bool:
        """
        Check if user has budget remaining.
        
        Args:
            employee_id: Employee identifier
            daily_limit: Maximum daily cost in dollars
        
        Returns:
            True if within budget, False otherwise
        """
        today = datetime.utcnow().date().isoformat()
        user_key = f"cost:{employee_id}:{today}"
        
        if self.use_redis and self.redis_client:
            total_cents = self.redis_client.get(user_key) or 0
            total = int(total_cents) / 100
        else:
            total = self._in_memory_store.get(user_key, 0)
        
        return total < daily_limit
    
    def get_daily_total(self, employee_id: str) -> float:
        """Get user's total cost for today."""
        today = datetime.now().date().isoformat()
        user_key = f"cost:{employee_id}:{today}"
        
        if self.use_redis and self.redis_client:
            total_cents = self.redis_client.get(user_key) or 0
            return int(total_cents) / 100
        else:
            return self._in_memory_store.get(user_key, 0)
