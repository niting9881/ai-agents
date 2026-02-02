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
        user_id: str,
        agent_name: str,
        prompt_version: str,
        user_message: str,
        response: Optional[Any],
        tokens_used: int,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
        trace_id: Optional[str] = None
    ):
        """
        Log an agent call with all relevant metadata.
        
        Args:
            user_id: User identifier
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
            "user_id": user_id,
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
        else:
            log_entry["error"] = error
        if trace_id:
            log_entry["trace_id"] = trace_id

        self.logger.info(json.dumps(log_entry))

    def log_event(self, event_type: str, message: str, trace_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        """Generic event logger for informational messages and headings."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "message": message,
        }
        if extra:
            entry.update(extra)
        if trace_id:
            entry["trace_id"] = trace_id
        self.logger.info(json.dumps(entry))

    def log_node_execution(self, trace_id: str, node: str, state_summary: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "node_execution",
            "node": node,
        }
        if state_summary:
            entry["state_summary"] = state_summary
        entry["trace_id"] = trace_id
        self.logger.info(json.dumps(entry))

    def log_llm_call(self, trace_id: str, node: str, model: str, input_tokens: int, output_tokens: int, cost_info: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "llm_call",
            "node": node,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if cost_info:
            entry["cost_info"] = cost_info
        entry["trace_id"] = trace_id
        self.logger.info(json.dumps(entry))

    def log_tool_invocation(self, trace_id: str, tool_name: str, params: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "tool_invocation",
            "tool_name": tool_name,
        }
        if params:
            entry["params"] = params
        if result:
            entry["result"] = result
        entry["trace_id"] = trace_id
        self.logger.info(json.dumps(entry))
