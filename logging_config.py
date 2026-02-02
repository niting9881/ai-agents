"""
Structured Logging - JSON-formatted logs for production debugging.
Makes it easy to search and analyze logs in Datadog/CloudWatch.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON logs."""
    
    def format(self, record):
        """Format log record as JSON."""
        return record.getMessage()


class StructuredLogger:
    """
    Structured logger for agent calls.
    Outputs JSON logs that are easy to search and analyze.
    """
    
    def __init__(self, name: str):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler with JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
    
    def log_agent_call(
        self,
        employee_id: str,
        agent_name: str,
        prompt_version: str,
        user_message: str,
        response: Optional[Any],
        tokens_used: int,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """
        Log an agent call with all relevant metadata.
        
        Args:
            employee_id: Employee identifier
            agent_name: Name of the agent
            prompt_version: Version of prompt used
            user_message: User's input message
            response: Response object (if success=True)
            tokens_used: Total tokens consumed
            latency_ms: Request latency in milliseconds
            success: Whether the call succeeded
            error: Error message (if success=False)
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "agent_call",
            "employee_id": employee_id,
            "agent_name": agent_name,
            "prompt_version": prompt_version,
            "user_message_length": len(user_message),
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "success": success
        }
        
        if success and response:
            # Add response fields if available
            if hasattr(response, 'action'):
                log_entry["action"] = response.action
            if hasattr(response, 'confidence'):
                log_entry["confidence"] = response.confidence
            if hasattr(response, 'requires_approval'):
                log_entry["requires_approval"] = response.requires_approval
            if hasattr(response, 'cost'):
                log_entry["cost_usd"] = response.cost
        else:
            log_entry["error"] = error
        
        self.logger.info(json.dumps(log_entry))
