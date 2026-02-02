"""
Task 4: Cost Tracker - Tracks LLM usage and enforces budget limits.
Prevents runaway costs in production.

Features:
- Tracks costs per LLM call with detailed token and cost breakdowns
- Supports multiple models (GPT-4o, GPT-4o-mini)
- Provides daily totals and per-user usage tracking
- In-memory storage (demo) or Redis-backed (production)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List
import json
import logging


@dataclass
class LLMCallCost:
    """Record of a single LLM call's cost"""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    node: str = "unknown"
    user_id: str = "unknown"


@dataclass
class CostTracker:
    """
    Task 4: Comprehensive LLM Cost Tracker
    
    Tracks LLM costs and enforces daily budgets.
    Supports multiple models with accurate pricing.
    
    In production, this would use Redis. For demo, we use in-memory dict.
    """
    
    # Pricing (adjust for your provider) - Updated for current OpenAI prices
    PRICES = {
        "gpt-4o-mini": {
            "input": 0.15 / 1_000_000,    # $0.15 per 1M input tokens
            "output": 0.60 / 1_000_000,   # $0.60 per 1M output tokens
            "display_name": "GPT-4o Mini"
        },
        "gpt-4o": {
            "input": 2.50 / 1_000_000,    # $2.50 per 1M input tokens  
            "output": 10.00 / 1_000_000,  # $10.00 per 1M output tokens
            "display_name": "GPT-4o"
        }
    }
    
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
        self._call_history = {}  # Track individual calls per user per day
        self.logger = logging.getLogger(__name__)
    
    def track_llm_call(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        node: str = "unknown"
    ) -> Dict[str, any]:
        """
        Task 4: Track cost and update user's daily total.
        
        Args:
            user_id: User identifier
            model: LLM model name (e.g., "gpt-4o-mini")
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            node: Agent node name where call occurred
        
        Returns:
            Dictionary with call_cost, daily_total, token counts, and cost breakdown
        """
        # Calculate cost
        if model not in self.PRICES:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.PRICES.keys())}")
        
        input_cost = input_tokens * self.PRICES[model]["input"]
        output_cost = output_tokens * self.PRICES[model]["output"]
        total_cost = input_cost + output_cost
        
        # Create cost record
        timestamp = datetime.now().isoformat()
        cost_record = LLMCallCost(
            timestamp=timestamp,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            node=node,
            user_id=user_id
        )
        
        # Update user's usage
        today = datetime.now().date().isoformat()
        user_key = f"cost:{user_id}:{today}"
        history_key = f"history:{user_id}:{today}"
        
        if self.use_redis and self.redis_client:
            # Use Redis (production)
            cost_cents = int(total_cost * 100)
            new_total_cents = self.redis_client.incrby(user_key, cost_cents)
            self.redis_client.expire(user_key, 60 * 60 * 24 * 30)  # 30 days
            new_total = new_total_cents / 100
            
            # Store call history in Redis list
            call_json = json.dumps({
                "timestamp": timestamp,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": total_cost,
                "node": node
            })
            self.redis_client.rpush(history_key, call_json)
            self.redis_client.expire(history_key, 60 * 60 * 24 * 30)
        else:
            # Use in-memory store (demo)
            if user_key not in self._in_memory_store:
                self._in_memory_store[user_key] = 0
            self._in_memory_store[user_key] += total_cost
            new_total = self._in_memory_store[user_key]
            
            # Store call history in memory
            if history_key not in self._call_history:
                self._call_history[history_key] = []
            self._call_history[history_key].append(cost_record)
        
        # Log the cost tracking
        self.logger.info(
            f"Task 4 - Cost tracked: {model} | "
            f"Input: {input_tokens} tokens (${input_cost:.6f}) | "
            f"Output: {output_tokens} tokens (${output_cost:.6f}) | "
            f"Total: ${total_cost:.6f} | "
            f"Daily Total: ${new_total:.6f}"
        )
        
        return {
            "call_cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "daily_total": new_total,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model,
            "timestamp": timestamp,
            "node": node
        }
    
    def get_daily_total(self, user_id: str) -> float:
        """Get user's total cost for today."""
        today = datetime.utcnow().date().isoformat()
        user_key = f"cost:{user_id}:{today}"
        
        if self.use_redis and self.redis_client:
            total_cents = self.redis_client.get(user_key) or 0
            return int(total_cents) / 100
        else:
            return self._in_memory_store.get(user_key, 0)
    
    def get_call_history(self, user_id: str) -> List[Dict]:
        """
        Task 4: Get all LLM calls for a user today.
        
        Returns list of call records with detailed cost breakdown.
        """
        today = datetime.utcnow().date().isoformat()
        history_key = f"history:{user_id}:{today}"
        
        if self.use_redis and self.redis_client:
            calls = []
            raw_calls = self.redis_client.lrange(history_key, 0, -1)
            for call_json in raw_calls:
                calls.append(json.loads(call_json))
            return calls
        else:
            if history_key not in self._call_history:
                return []
            # Convert LLMCallCost objects to dicts
            calls = []
            for cost_record in self._call_history[history_key]:
                calls.append({
                    "timestamp": cost_record.timestamp,
                    "model": cost_record.model,
                    "input_tokens": cost_record.input_tokens,
                    "output_tokens": cost_record.output_tokens,
                    "input_cost": cost_record.input_cost,
                    "output_cost": cost_record.output_cost,
                    "total_cost": cost_record.total_cost,
                    "node": cost_record.node,
                    "user_id": cost_record.user_id
                })
            return calls
    
    def get_cost_summary(self, user_id: str) -> Dict:
        """
        Task 4: Get comprehensive cost summary for a user today.
        
        Includes total cost, breakdown by model, and call count.
        """
        calls = self.get_call_history(user_id)
        daily_total = self.get_daily_total(user_id)
        
        # Group by model
        by_model = {}
        for call in calls:
            model = call['model']
            if model not in by_model:
                by_model[model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost": 0
                }
            by_model[model]["calls"] += 1
            by_model[model]["input_tokens"] += call["input_tokens"]
            by_model[model]["output_tokens"] += call["output_tokens"]
            by_model[model]["total_cost"] += call["total_cost"]
        
        # Group by node
        by_node = {}
        for call in calls:
            node = call.get('node', 'unknown')
            if node not in by_node:
                by_node[node] = {
                    "calls": 0,
                    "total_cost": 0
                }
            by_node[node]["calls"] += 1
            by_node[node]["total_cost"] += call["total_cost"]
        
        return {
            "daily_total": daily_total,
            "total_calls": len(calls),
            "by_model": by_model,
            "by_node": by_node,
            "calls": calls
        }

    def check_budget(self, user_id: str, estimated_cost: float, budget_per_request: float = 0.50, budget_per_user_daily: float = 5.0):
        """
        Simple budget check helper.

        Args:
            user_id: user identifier
            estimated_cost: estimated cost for the pending operation
            budget_per_request: per-request budget threshold
            budget_per_user_daily: per-user daily budget

        Returns:
            (allowed: bool, reason: Optional[str])
        """
        # Check per-request limit
        if estimated_cost > budget_per_request:
            return False, f"Estimated cost ${estimated_cost:.6f} exceeds per-request budget ${budget_per_request:.6f}"

        # Check daily budget
        daily_total = self.get_daily_total(user_id)
        if (daily_total + estimated_cost) > budget_per_user_daily:
            return False, f"Estimated daily total ${(daily_total + estimated_cost):.6f} exceeds daily budget ${budget_per_user_daily:.6f}"

        return True, None
