# Your Week 1 Agent Code (Starting Point)
import redis
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import (
    LangChainException,
    OutputParserException,
)
from models import RoutingDecision, IssueAnalysis, EscalationDecision, SupportResponse, SupportOutput
from error_handling import (
    retry_with_backoff,
    RateLimitError,
    MaxIterationsError,
    ToolExecutionError,
    InvalidOutputError,
    LLMError,
    ErrorContext,
    RetryStrategy,
    handle_rate_limit_error,
    handle_llm_error,
    handle_tool_execution_error,
    handle_max_iterations_error,
    log_error_chain,
)
from rate_limiter import RateLimiter
from cost_tracker import CostTracker
from input_sanitizer import InputSanitizer
from output_validator import OutputValidator
from prompt_hardener import PromptHardener, DefenseLevel
from logging_config import StructuredLogger
from ab_test_manager import ABTestManager
import operator
from datetime import datetime
import logging
from pathlib import Path
import uuid

from dotenv import load_dotenv
load_dotenv()

from prompt_manager import PromptManager

# Task 3: Initialize structured logger
structured_logger = StructuredLogger(__name__)


# 1. State Definition
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    should_escalate: bool
    issue_type: str
    user_tier: str
    prompt_version: str  # Track which prompt version is being used
    routing_decision: RoutingDecision | None  # Structured tier routing
    issue_analysis: IssueAnalysis | None  # Structured issue classification
    escalation_decision: EscalationDecision | None  # Structured escalation logic
    support_response: SupportResponse | None  # Structured final response
    # Task 3: Error tracking fields
    error_count: int  # Number of errors encountered
    last_error: str | None  # Last error message
    error_history: list[dict]  # Full error history with timestamps
    # Task 4: Cost tracking fields
    total_cost: float  # Total cost for this conversation
    request_costs: list[dict]  # List of costs per LLM call
    cost_summary: dict | None  # Summary of costs by model and node

# Initialize PromptManager for version control with absolute path
# Resolve path relative to this script's location
script_dir = Path(__file__).parent.resolve()
prompts_dir = script_dir / "prompts" / "agents"
prompt_manager = PromptManager(prompts_dir=str(prompts_dir))

# ════════════════════════════════════════════════════════════════════════════════
# TASK 1: PROMPT LOADING & VERSIONING
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TASK 1: PROMPT LOADING & VERSIONING")
print("="*80)

# Get all available versions
all_versions = prompt_manager.get_version_history('customer_support')

# Separate and sort versions
semantic_versions = []
current_version = None

for version in all_versions:
    if version == 'current':
        # Get the actual version that 'current' points to
        try:
            current_info = prompt_manager.get_version_info('customer_support', 'current')
            current_version = current_info.get('version', 'unknown')
        except Exception:
            current_version = 'unknown'
    else:
        semantic_versions.append(version)

# Sort versions (v1.0.0, v1.1.0, v1.2.0, etc.)
if semantic_versions:
    semantic_versions.sort(key=lambda x: tuple(map(int, x.lstrip('v').split('.'))))

print("\n  Available Versions (lowest to highest):")
if semantic_versions:
    for version in semantic_versions:
        try:
            info = prompt_manager.get_version_info('customer_support', version)
            status = info.get('status', 'unknown')
            description = info.get('description', 'No description')
            if len(description) > 50:
                description = description[:50] + "..."
            print(f"   {version} ({status}): {description}")
        except Exception:
            print(f"   {version}")
else:
    print("   (None found)")

print(f"\n Current Latest Version: {current_version}")
print(f"   Status: Currently active (prompts/agents/customer_support/current.yaml)")
print("\n" + "="*80 + "\n")

# ════════════════════════════════════════════════════════════════════════════════
# TASK 2: STRUCTURED OUTPUT WITH PYDANTIC
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("Task 2: Structured Output with Pydantic")
print("="*80)

# Demonstrate Pydantic models for structured output
example_output = SupportOutput(
    action="process_refund",
    confidence=0.95,
    message="I'd be happy to process that refund for you. You'll see it back in your original payment method within 3-5 business days.",
    reasoning="Standard tier, within 30-day window, amount ($49.99) within $100 authority limit",
    requires_approval=False,
    tier_adjusted=False
)

print("\n   Pydantic SupportOutput Model Example:")
print(f"     Action: {example_output.action}")
print(f"     Confidence: {example_output.confidence:.2%}")
print(f"     Message: {example_output.message[:60]}...")
print(f"     Requires Approval: {example_output.requires_approval}")
print(f"     Tier Adjusted: {example_output.tier_adjusted}")

# Demonstrate model validation
example_escalation = SupportOutput(
    action="escalate_to_human",
    confidence=0.90,
    message="I understand your request. Since this is a larger refund, let me connect you with our manager who has authority to approve this.",
    reasoning="Within refund window but amount ($499) exceeds standard tier authority ($100)",
    requires_approval=True,
    tier_adjusted=True
)

print("\n   SupportOutput Model - Escalation Example:")
print(f"     Action: {example_escalation.action}")
print(f"     Confidence: {example_escalation.confidence:.2%}")
print(f"     Requires Approval: {example_escalation.requires_approval}")
print(f"     Tier Adjusted: {example_escalation.tier_adjusted}")
print("\n All Pydantic models are properly configured for structured output")

import sys
sys.stdout.flush()

# ════════════════════════════════════════════════════════════════════════════════
# TASK 3: ERROR HANDLING WITH RETRIES
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("Task 3: Error Handling with Retries")
print("="*80)

print("\n    Custom Exception Classes:")
print("     - RateLimitError: Raised when rate limits exceeded")
print("     - MaxIterationsError: Raised when max iterations exceeded")
print("     - ToolExecutionError: Raised when tool execution fails")
print("     - InvalidOutputError: Raised when LLM output is invalid")
print("     - LLMError: Raised when LLM call fails")

print("\n @retry_with_backoff Decorator Configuration:")
print("     Strategy: EXPONENTIAL backoff with jitter")
print("     Max Retries: 2 attempts")
print("     Initial Delay: 0.5 seconds")
print("     Backoff Factor: 2.0x")
print("     Jitter: Enabled (randomizes delay)")

print("\n[OK] Example Error Handling Scenarios:")

# Example 1: RateLimitError
try:
    error1 = RateLimitError(
        message="Rate limit exceeded for tier check",
        retry_after_seconds=60,
        user_id="user_123",
        context={"operation": "check_user_tier_node"}
    )
    print(f"     1. RateLimitError: {error1.message}")
    print(f"        Retry After: {error1.retry_after_seconds}s")
except Exception as e:
    print(f"     1. RateLimitError creation failed: {e}")

# Example 2: MaxIterationsError
try:
    error2 = MaxIterationsError(
        message="Maximum graph iterations exceeded",
        max_iterations=10,
        current_iteration=10,
        context={"operation": "graph_execution"}
    )
    print(f"     2. MaxIterationsError: {error2.message}")
    print(f"        Max Iterations: {error2.max_iterations}")
except Exception as e:
    print(f"     2. MaxIterationsError creation failed: {e}")

# Example 3: ToolExecutionError
try:
    error3 = ToolExecutionError(
        message="Tool execution failed",
        tool_name="process_refund",
        original_error=None,
        context={"operation": "tool_execution"}
    )
    print(f"     3. ToolExecutionError: {error3.message}")
    print(f"        Tool: {error3.tool_name}")
except Exception as e:
    print(f"     3. ToolExecutionError creation failed: {e}")

# Example 4: InvalidOutputError
try:
    error4 = InvalidOutputError(
        message="LLM output validation failed",
        expected_format="Pydantic SupportOutput model",
        actual_output="plain text response",
        validation_error="Missing required field 'action'",
        context={"operation": "output_validation"}
    )
    print(f"     4. InvalidOutputError: {error4.message}")
    print(f"        Expected: {error4.expected_format}")
except Exception as e:
    print(f"     4. InvalidOutputError creation failed: {e}")

print("\n    Retry Strategy Features:")
print("     - Automatic exponential backoff on failure")
print("     - Jitter prevents thundering herd problem")
print("     - Max 2 retries before final failure")
print("     - ErrorContext captures user_id for tracking")
print("     - All errors logged via StructuredLogger")

print("\n    Error Handling Integration Points:")
print("     - RateLimiter: Prevents API overload")
print("     - InputSanitizer: Validates/cleans input")
print("     - OutputValidator: Ensures response validity")
print("     - PromptHardener: Defends against injection")
print("     - StructuredLogger: Logs all errors with context")

import sys
sys.stdout.flush()

# ════════════════════════════════════════════════════════════════════════════════
# TASK 4: COST TRACKING
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("Task 4: Cost Tracking")
print("="*80)

print("\n[OK] Initializing Cost Tracking System:")
print("     - CostTracker: Monitors LLM API costs per operation")
print("     - Redis Support: Optional Redis backend for distributed tracking")
print("     - Cost Breakdown: Tracks costs by model, node, and operation")

print("\n[OK] Cost Tracking Features:")
print("     - Per-LLM call cost tracking (tokens x model rate)")
print("     - Aggregated cost summary by node")
print("     - Support for multiple LLM models (GPT-4, GPT-3.5, etc.)")
print("     - Real-time cost monitoring and reporting")

print("\n[OK] Storage Options:")
print("     - Redis: Distributed cost tracking (if available)")
print("     - In-Memory: Fallback storage for single instance")

# Initialize cost tracker (use Redis if available, otherwise in-memory)
try:
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    # Test connection
    redis_client.ping()
    cost_tracker = CostTracker(redis_client=redis_client, use_redis=True)
    print("\n[OK] Cost Tracker: Redis Backend Connected")
except Exception:
    # Fallback to in-memory if Redis unavailable
    cost_tracker = CostTracker(use_redis=False)
    print("\n[OK] Cost Tracker: In-Memory Storage (Redis unavailable)")

# Helper functions for Task 4: extract token usage and centralize cost tracking
def _extract_token_usage(llm_result, fallback_prompt_text: str = ""):
    """Return (input_tokens, output_tokens) extracted from various LLM result shapes.
    Falls back to estimates when precise usage isn't available.
    """
    usage = None
    # dict-like response
    if isinstance(llm_result, dict):
        usage = llm_result.get("usage") or llm_result.get("llm_response", {}).get("usage")
    else:
        # common attributes that may contain raw/usage info
        for attr in ("llm_response", "raw_response", "_raw_response", "response", "metadata", "_last_response"):
            obj = getattr(llm_result, attr, None)
            if obj:
                if isinstance(obj, dict):
                    usage = obj.get("usage") or obj.get("usage")
                else:
                    usage = getattr(obj, "usage", None)
                if usage:
                    break

    prompt_tokens = None
    completion_tokens = None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

    # try direct attributes
    if prompt_tokens is None:
        try:
            prompt_tokens = int(getattr(llm_result, "prompt_tokens", None) or getattr(llm_result, "input_tokens", None) or 0)
        except Exception:
            prompt_tokens = None
    if completion_tokens is None:
        try:
            completion_tokens = int(getattr(llm_result, "completion_tokens", None) or getattr(llm_result, "output_tokens", None) or 0)
        except Exception:
            completion_tokens = None

    # Fallback estimates
    if not prompt_tokens:
        prompt_tokens = max(1, (len(fallback_prompt_text) // 4) if fallback_prompt_text else 1)
    if not completion_tokens:
        completion_tokens = 50

    return int(prompt_tokens), int(completion_tokens)


def _track_and_append_cost(state: SupportState, llm_result, model: str, node: str, prompt_text: str, user_tier: str):
    """Use CostTracker to compute costs from an LLM result and append to state.
    Updates `state['request_costs']`, `state['total_cost']`, and `state['cost_summary']`.
    Returns the CostTracker return dict for the call.
    """
    in_tokens, out_tokens = _extract_token_usage(llm_result, fallback_prompt_text=prompt_text)

    cost_info = cost_tracker.track_llm_call(
        user_id=(user_tier or "unknown"),
        model=model,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        node=node
    )

    request_costs = state.get("request_costs", [])
    request_costs.append(cost_info)

    # Update totals
    total_cost = state.get("total_cost", 0.0) + cost_info.get("call_cost", 0.0)
    cost_summary = state.get("cost_summary") or {"vip": {"total": 0.0, "calls": 0}, "standard": {"total": 0.0, "calls": 0}}
    tier_key = "vip" if user_tier and str(user_tier).lower() in ("vip", "premium") else "standard"
    cost_summary.setdefault(tier_key, {"total": 0.0, "calls": 0})
    cost_summary[tier_key]["total"] = cost_summary[tier_key].get("total", 0.0) + cost_info.get("call_cost", 0.0)
    cost_summary[tier_key]["calls"] = cost_summary[tier_key].get("calls", 0) + 1

    state["request_costs"] = request_costs
    state["total_cost"] = total_cost
    state["cost_summary"] = cost_summary

    return cost_info


def sanitize_input_node(state: SupportState):
    """Sanitize incoming user message and update state before routing.

    Replaces the last message with the cleaned message and sets flags
    that downstream nodes can inspect.
    """
    try:
        with ErrorContext("sanitize_input", user_id="unknown_user"):
            messages = state.get("messages", [])
            user_message = messages[-1].content if messages else ""

            cleaned_message, is_suspicious = sanitizer.sanitize(user_message)
            has_encoding = sanitizer.detect_encoding_attacks(cleaned_message)
            has_repetition = sanitizer.detect_repetition_attacks(cleaned_message)

            # Replace last message with cleaned version
            if messages:
                messages[-1] = HumanMessage(content=cleaned_message)
            else:
                messages.append(HumanMessage(content=cleaned_message))

            # Update state flags
            state["is_suspicious"] = bool(is_suspicious or has_encoding or has_repetition)
            state["sanitization_notes"] = {
                "injection_flag": bool(is_suspicious),
                "encoding_flag": bool(has_encoding),
                "repetition_flag": bool(has_repetition)
            }

            # Log the sanitization action
            if state["is_suspicious"]:
                structured_logger.log_agent_call(
                    user_id="sanitize",
                    agent_name="sanitize_input",
                    prompt_version=state.get("prompt_version", "v1.0.0"),
                    user_message=user_message,
                    response=None,
                    tokens_used=0,
                    latency_ms=0,
                    success=False,
                    error="Suspicious input detected during sanitization",
                    trace_id=state.get("trace_id")
                )
            else:
                structured_logger.log_agent_call(
                    user_id="sanitize",
                    agent_name="sanitize_input",
                    prompt_version=state.get("prompt_version", "v1.0.0"),
                    user_message=user_message,
                    response=cleaned_message,
                    tokens_used=0,
                    latency_ms=0,
                    success=True,
                    trace_id=state.get("trace_id")
                )

            # Human-readable output for task visibility
            print("[sanitize_input] Cleaned message:", cleaned_message)
            if state["is_suspicious"]:
                print("[sanitize_input] Suspicious flags:", state.get("sanitization_notes"))

            return {
                "messages": messages,
                "is_suspicious": state["is_suspicious"],
                "sanitization_notes": state.get("sanitization_notes", {})
            }

    except Exception as e:
        log_error_chain(e, {"node": "sanitize_input"})
        return {
            "messages": state.get("messages", []),
            "is_suspicious": True,
            "sanitization_notes": {"error": str(e)}
        }


def _print_task4_costs(result: dict):
    trace_id = result.get("trace_id") if isinstance(result, dict) else None
    if not result:
        structured_logger.log_event("info", "No result available to summarize costs.", trace_id=trace_id)
        return

    total_cost = result.get("total_cost", 0.0)
    request_costs = result.get("request_costs", [])
    cost_summary = result.get("cost_summary", {})

    structured_logger.log_event("task", "Task 4: Cost Tracking (Summary)", trace_id=trace_id)
    print("\nTask 4: Cost Tracking (Summary)")
    structured_logger.log_event("cost_summary", f"Total Cost: ${total_cost:.6f}", trace_id=trace_id, extra={"total_calls": len(request_costs)})

    if request_costs:
        for i, cost in enumerate(request_costs, 1):
            structured_logger.log_event(
                "cost_detail",
                f"Call {i}: {cost.get('node', 'unknown')}",
                trace_id=trace_id,
                extra={
                    "model": cost.get("model"),
                    "input_tokens": cost.get("input_tokens"),
                    "output_tokens": cost.get("output_tokens"),
                    "input_cost": cost.get("input_cost"),
                    "output_cost": cost.get("output_cost"),
                    "call_cost": cost.get("call_cost")
                }
            )

    if cost_summary:
        for tier in ("vip", "standard"):
            cs = cost_summary.get(tier, {"total": 0.0, "calls": 0})
            structured_logger.log_event(
                "cost_tier",
                f"{tier.upper()}: Total ${cs.get('total',0.0):.6f} across {cs.get('calls',0)} calls",
                trace_id=trace_id,
            )




# ════════════════════════════════════════════════════════════════════════════════
# TASK 5: PROMPT INJECTION DEFENSE
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("Task 5: Prompt Injection Defense")
print("="*80)

print("\n   Initializing Input Sanitization & Output Validation:")
print("     - InputSanitizer: Detects and mitigates prompt injection attacks")
print("     - OutputValidator: Validates LLM output against expected schemas")
print("     - PromptHardener: Applies security hardening strategies")

print("\n   Defense Mechanisms:")
print("     - Input Sanitization: Detects encoding attacks, repetition patterns")
print("     - Schema Validation: Ensures outputs match Pydantic models")
print("     - Sandwich Defense: Security guards in top/bottom of prompts")
print("     - Output Constraints: Type-safe structured output validation")

print("\n   Attack Vectors Protected:")
print("     - Prompt injection via special characters")
print("     - Unicode/encoding-based attack vectors")
print("     - Repetition/flooding attacks")
print("     - Invalid or malformed output responses")


sanitizer = InputSanitizer()
validator = OutputValidator()
hardener = PromptHardener(defense_level=DefenseLevel.INTERMEDIATE)
# Task 3: Initialize rate limiter and logger
rate_limiter = RateLimiter(use_redis=False)

# Task 3: Rate limit config (10 requests per 60 seconds)
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
MAX_GRAPH_ITERATIONS = 10

# Integration check: ensure the key components are imported and instantiated
try:
    pm = prompt_manager  # alias used by some integration snippets
    logger = structured_logger  # alias to match reference code expectations
    # Basic sanity checks
    assert isinstance(sanitizer, InputSanitizer)
    assert isinstance(cost_tracker, CostTracker)
    assert hasattr(pm, "load_prompt") or hasattr(pm, "get_system_prompt")

    structured_logger.log_event("info", "Integration check passed: PromptManager, CostTracker, Sanitizer, Logger available", trace_id=None)
    print("Integration check: PromptManager, CostTracker, Sanitizer, Logger initialized")
except Exception as e:
    structured_logger.log_event("error", f"Integration check failed: {str(e)}", trace_id=None)
    print(f"Integration check failed: {str(e)}")

# 2. Check tier node with structured output and error handling
@retry_with_backoff(
    max_retries=2,
    initial_delay=0.5,
    backoff_factor=2.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=True,
    exceptions=(LangChainException, Exception)
)
def check_user_tier_node(state: SupportState):
    """
    Classify user tier using LLM with structured output.
    Uses RoutingDecision Pydantic model for guaranteed valid responses.
    
    Task 3: Added error handling with retries, rate limiting, and structured logging.
    """
    try:
        with ErrorContext("check_user_tier", user_id="unknown_user"):
            messages = state["messages"]
            user_message = messages[-1].content if messages else ""
            # Note: sanitization is handled in a dedicated node `sanitize_input_node`.
            # The message here should already be cleaned by the sanitizer node.
            cleaned_message = messages[-1].content if messages else ""
            is_suspicious = state.get("is_suspicious", False)
            
            # Task 3: Rate limit check
            allowed, retry_after = rate_limiter.check_rate_limit(
                user_id="tier_check",
                max_requests=RATE_LIMIT_MAX_REQUESTS,
                window_seconds=RATE_LIMIT_WINDOW_SECONDS
            )
            
            if not allowed:
                error = RateLimitError(
                    message="Rate limit exceeded for tier check",
                    retry_after_seconds=retry_after or 60,
                    user_id="tier_check",
                    context={"operation": "check_user_tier_node"}
                )
                structured_logger.log_agent_call(
                    user_id="tier_check",
                    agent_name="tier_classifier",
                    prompt_version="v1.0.0",
                    user_message=user_message,
                    response=None,
                    tokens_used=0,
                    latency_ms=0,
                    success=False,
                    error=str(error),
                    trace_id=state.get("trace_id")
                )
                raise error
            
            # Create a tier detection prompt
            tier_detection_prompt = f"""
Analyze this customer message and classify their tier based on context clues.

Customer Message: {cleaned_message}

Classify them into one of these tiers:
- VIP: Premium/VIP customers, high-value accounts, mentions premium status
- PREMIUM: Premium tier customers, mentions premium features
- STANDARD: Regular/free tier customers, or tier not specified

Provide your classification with confidence score and reasoning.
"""
            
            # Task 3: Use LLM with structured output and error handling
            start_time = datetime.now()
            tier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            try:
                routing_decision = tier_llm.with_structured_output(RoutingDecision).invoke(
                    [HumanMessage(content=tier_detection_prompt)]
                )
            except OutputParserException as e:
                error = InvalidOutputError(
                    message="Failed to parse tier classification output",
                    expected_format="RoutingDecision",
                    actual_output=str(e),
                    validation_error=str(e),
                    context={"operation": "check_user_tier_node"}
                )
                log_error_chain(e, {"node": "check_user_tier_node"})
                raise error
            except Exception as e:
                error = LLMError(
                    message=f"LLM call failed for tier classification: {str(e)}",
                    error_type="API_ERROR",
                    api_error=str(e),
                    context={"operation": "check_user_tier_node"}
                )
                log_error_chain(e, {"node": "check_user_tier_node"})
                raise error
            
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Task 4: Track cost for this LLM call using actual usage when available
            # Original (pre-refactor) hardcoded token usage example (preserved for audit):
            # cost_info = cost_tracker.track_llm_call(
            #     user_id=str(getattr(routing_decision, "user_tier", "standard")),
            #     model="gpt-4o-mini",
            #     input_tokens=50,
            #     output_tokens=150,  # previously hardcoded estimate
            #     node="check_user_tier_node"
            # )
            cost_info = _track_and_append_cost(
                state=state,
                llm_result=routing_decision,
                model="gpt-4o-mini",
                node="check_user_tier_node",
                prompt_text=tier_detection_prompt,
                user_tier=str(getattr(routing_decision, "user_tier", "standard"))
            )
            
            # Task 3: Log successful agent call
            structured_logger.log_agent_call(
                user_id="tier_check",
                agent_name="tier_classifier",
                prompt_version="v1.0.0",
                user_message=user_message,
                response=routing_decision,
                tokens_used=0,  # Would need to track from LLM
                latency_ms=elapsed_ms,
                success=True,
                trace_id=state.get("trace_id")
            )
            
            # Map tier to version
            tier_to_version = {
                "vip": "v1.1.0",
                "premium": "v1.1.0",
                "standard": "v1.0.0"
            }
            prompt_version = tier_to_version.get(routing_decision.user_tier, "v1.0.0")

            # Human-readable output for Task 1/ routing visibility
            print(f"[check_user_tier_node] Routing decision -> tier: {routing_decision.user_tier}, confidence: {getattr(routing_decision, 'confidence', 'n/a')}")
            print(f"[check_user_tier_node] Reasoning: {getattr(routing_decision, 'reasoning', '')}")

            return {
                "user_tier": routing_decision.user_tier,
                "prompt_version": prompt_version,
                "routing_decision": routing_decision,
                "error_count": state.get("error_count", 0),
                "last_error": None,
                "error_history": state.get("error_history", []),
                "total_cost": state.get("total_cost", 0),
                "request_costs": state.get("request_costs", []),
                "cost_summary": state.get("cost_summary")
            }
    
    except RateLimitError as e:
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "RateLimitError",
            "message": e.message,
            "node": "check_user_tier_node"
        }
        error_history = state.get("error_history", [])
        error_history.append(error_entry)
        
        # Human-readable fallback
        print("[check_user_tier_node] Rate limit exceeded; falling back to STANDARD tier")
        return {
            "user_tier": "standard",  # Fallback to standard tier
            "prompt_version": "v1.0.0",
            "routing_decision": None,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"Rate limit exceeded: {e.retry_after_seconds}s retry-after",
            "error_history": error_history,
            "total_cost": state.get("total_cost", 0),
            "request_costs": state.get("request_costs", []),
            "cost_summary": None
        }
    
    except (LLMError, InvalidOutputError) as e:
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(e).__name__,
            "message": e.message,
            "node": "check_user_tier_node"
        }
        error_history = state.get("error_history", [])
        error_history.append(error_entry)
        
        # Human-readable fallback
        print(f"[check_user_tier_node] Tier detection failed: {e.message}")
        return {
            "user_tier": "standard",  # Fallback to standard tier
            "prompt_version": "v1.0.0",
            "routing_decision": None,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"Tier detection failed: {e.message}",
            "error_history": error_history,
            "total_cost": state.get("total_cost", 0),
            "request_costs": state.get("request_costs", []),
            "cost_summary": None
        }
    
    except Exception as e:
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(e).__name__,
            "message": str(e),
            "node": "check_user_tier_node"
        }
        error_history = state.get("error_history", [])
        error_history.append(error_entry)
        log_error_chain(e, {"node": "check_user_tier_node"})
        
        return {
            "user_tier": "standard",  # Fallback to standard tier
            "prompt_version": "v1.0.0",
            "routing_decision": None,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"Unexpected error: {str(e)[:100]}",
            "error_history": error_history,
            "total_cost": state.get("total_cost", 0),
            "request_costs": state.get("request_costs", []),
            "cost_summary": None
        }
# 3. Routing function
def route_by_tier(state: SupportState) -> str:
    """Route based on user tier from the routing decision"""
    tier = state.get("user_tier", "standard")
    # VIP and Premium go to vip_agent, Standard goes to standard_agent
    if tier in ["vip", "premium"]:
        return "vip"
    return "standard"

# Setup LLM (separate instance for main agent)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. Agent nodes with structured output and error handling
@retry_with_backoff(
    max_retries=2,
    initial_delay=0.5,
    backoff_factor=2.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=True,
    exceptions=(LangChainException, Exception)
)
def vip_agent_node(state: SupportState):
    """
    VIP agent with structured output and error handling.
    Uses SupportResponse model for guaranteed valid responses.
    No escalation - VIP handling is immediate.
    
    Task 3: Added comprehensive error handling with retries, rate limiting, and logging.
    """
    try:
        with ErrorContext("vip_agent", user_id="vip_user"):
            # Task 3: Check if we've hit max errors
            if state.get("error_count", 0) >= 3:
                error = MaxIterationsError(
                    message="Max errors reached in VIP agent",
                    max_iterations=3,
                    current_iteration=state.get("error_count", 0),
                    context={"node": "vip_agent"}
                )
                return {
                    "messages": [HumanMessage(content=handle_max_iterations_error(error)["message"])],
                    "should_escalate": True,
                    "prompt_version": state.get("prompt_version", "current"),
                    "support_response": None,
                    "error_count": state.get("error_count", 0) + 1,
                    "last_error": error.message,
                    "error_history": state.get("error_history", []),
                    "total_cost": state.get("total_cost", 0),
                    "request_costs": state.get("request_costs", []),
                    "cost_summary": None
                }
            
            messages = state["messages"]
            
            # Load versioned prompt based on tier
            prompt_version = state.get("prompt_version", "current")
            
            try:
                prompt_data = prompt_manager.load_prompt("customer_support", version=prompt_version)
            except Exception as e:
                error = ToolExecutionError(
                    message=f"Failed to load prompt version {prompt_version}",
                    tool_name="prompt_manager.load_prompt",
                    original_error=e,
                    context={"version": prompt_version}
                )
                log_error_chain(e, {"node": "vip_agent", "operation": "load_prompt"})
                raise error
            
            user_message = messages[-1].content if messages else ""
            
            # Task 5: Input Sanitization for VIP agent
            cleaned_message, is_suspicious_input = sanitizer.sanitize(user_message)
            if is_suspicious_input:
                structured_logger.log_agent_call(
                    user_id="vip_user",
                    agent_name="vip_agent",
                    prompt_version=prompt_version,
                    user_message=user_message,
                    response=None,
                    tokens_used=0,
                    latency_ms=0,
                    success=False,
                    error="Suspicious input detected in VIP agent",
                    trace_id=state.get("trace_id")
                )
            
            # Compile 4-layer prompt
            try:
                system_prompt = prompt_manager.compile_prompt(prompt_data, cleaned_message)
            except Exception as e:
                error = ToolExecutionError(
                    message="Failed to compile prompt",
                    tool_name="prompt_manager.compile_prompt",
                    original_error=e,
                    context={"version": prompt_version}
                )
                log_error_chain(e, {"node": "vip_agent", "operation": "compile_prompt"})
                raise error
            
            # Task 5: Prompt Hardening with sandwich defense
            hardened_system_prompt = hardener.harden_system_prompt(system_prompt, cleaned_message)
            
            # Build message list with compiled prompt
            enhanced_messages = [
                SystemMessage(content=hardened_system_prompt),
                HumanMessage(content=cleaned_message)
            ]
            
            # Task 3: Use structured output with error handling
            start_time = datetime.now()
            
            try:
                # Pre-call budget check (aligns with reference pattern)
                user_id = state.get("user_id", "vip_user")
                estimated_cost = 0.05  # conservative estimate
                allowed, reason = cost_tracker.check_budget(user_id, estimated_cost)
                if not allowed:
                    structured_logger.log_event("warning", f"Budget check failed: {reason}", trace_id=state.get("trace_id"))
                    print(f"Budget exceeded: {reason}")
                    return {
                        "messages": [HumanMessage(content=f"Budget exceeded: {reason}")],
                        "should_escalate": False,
                        "prompt_version": prompt_version,
                        "support_response": None,
                        "error_count": state.get("error_count", 0),
                        "last_error": "budget_exceeded",
                        "error_history": state.get("error_history", []),
                        "total_cost": state.get("total_cost", 0),
                        "request_costs": state.get("request_costs", []),
                        "cost_summary": state.get("cost_summary")
                    }

                support_response = llm.with_structured_output(SupportResponse).invoke(enhanced_messages)
            except OutputParserException as e:
                error = InvalidOutputError(
                    message="Failed to parse support response",
                    expected_format="SupportResponse",
                    actual_output=str(e),
                    validation_error=str(e),
                    context={"node": "vip_agent"}
                )
                log_error_chain(e, {"node": "vip_agent"})
                raise error
            except Exception as e:
                error = LLMError(
                    message=f"LLM call failed for VIP support: {str(e)}",
                    error_type="API_ERROR",
                    api_error=str(e),
                    context={"node": "vip_agent"}
                )
                log_error_chain(e, {"node": "vip_agent"})
                raise error
            
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Task 5: Output Validation - Final defense before returning response
            is_valid, error_type = validator.validate(
                support_response.message,
                user_email="customer@example.com",  # Would come from state
                action="support_response",
                requires_approval=support_response.requires_approval
            )
            
            if not is_valid:
                logger.error(f"Output validation failed: {error_type}")
                structured_logger.log_agent_call(
                    user_id="vip_user",
                    agent_name="vip_agent",
                    prompt_version=prompt_version,
                    user_message=user_message,
                    response=None,
                    tokens_used=0,
                    latency_ms=elapsed_ms,
                    success=False,
                    error=f"Output validation failed: {error_type}",
                    trace_id=state.get("trace_id")
                )
                # Return safe fallback message
                support_response.message = "I apologize, but I cannot provide that information. Please contact our support team directly."
            
            # Task 4: Track cost for this LLM call using actual usage when available
            # Original (pre-refactor) hardcoded token usage example (preserved for audit):
            # cost_info = cost_tracker.track_llm_call(
            #     user_id="vip",
            #     model="gpt-4o-mini",
            #     input_tokens=100,
            #     output_tokens=200,  # previously hardcoded estimate
            #     node="vip_agent_node"
            # )
            cost_info = _track_and_append_cost(
                state=state,
                llm_result=support_response,
                model="gpt-4o-mini",
                node="vip_agent_node",
                prompt_text=(system_prompt + "\n" + user_message),
                user_tier="vip"
            )
            
            # Task 3: Log successful call
            structured_logger.log_agent_call(
                user_id="vip_user",
                agent_name="vip_agent",
                prompt_version=prompt_version,
                user_message=user_message,
                response=support_response,
                tokens_used=0,
                latency_ms=elapsed_ms,
                success=True,
                trace_id=state.get("trace_id")
            )
            
            # Human-readable output for VIP agent
            print("[vip_agent] Support response:", support_response.message)
            print(f"[vip_agent] Action: {support_response.action}, Confidence: {support_response.confidence}, Requires approval: {support_response.requires_approval}")

            return {
                "messages": [HumanMessage(content=support_response.message)],
                "should_escalate": support_response.requires_approval,
                "prompt_version": prompt_version,
                "support_response": support_response,
                "error_count": state.get("error_count", 0),
                "last_error": None,
                "error_history": state.get("error_history", []),
                "total_cost": state.get("total_cost", 0),
                "request_costs": state.get("request_costs", []),
                "cost_summary": state.get("cost_summary")
            }
    
    except (ToolExecutionError, LLMError, InvalidOutputError) as e:
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(e).__name__,
            "message": e.message,
            "node": "vip_agent"
        }
        error_history = state.get("error_history", [])
        error_history.append(error_entry)
        
        fallback_response = SupportResponse(
            action="escalate_to_human",
            confidence=0.5,
            message=f"I apologize, but I encountered an issue processing your request. "
                    f"A human agent will assist you shortly.",
            requires_approval=True,
            tier_adjusted=True,
            reasoning=f"Error recovery: {e.message}"
        )
        
        structured_logger.log_agent_call(
            user_id="vip_user",
            agent_name="vip_agent",
            prompt_version=state.get("prompt_version", "unknown"),
            user_message=messages[-1].content if messages else "",
            response=fallback_response,
            tokens_used=0,
            latency_ms=0,
            success=False,
            error=str(e),
            trace_id=state.get("trace_id")
        )
        
        # Human-readable fallback output
        print("[vip_agent] Fallback response:", fallback_response.message)
        return {
            "messages": [HumanMessage(content=fallback_response.message)],
            "should_escalate": True,
            "prompt_version": state.get("prompt_version", "current"),
            "support_response": fallback_response,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": e.message,
            "error_history": error_history,
            "total_cost": state.get("total_cost", 0),
            "request_costs": state.get("request_costs", []),
            "cost_summary": None
        }
    
    except Exception as e:
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(e).__name__,
            "message": str(e),
            "node": "vip_agent"
        }
        error_history = state.get("error_history", [])
        error_history.append(error_entry)
        log_error_chain(e, {"node": "vip_agent"})
        
        print("[vip_agent] Unexpected error occurred:", str(e))
        return {
            "messages": [HumanMessage(content="An unexpected error occurred. Please try again later.")],
            "should_escalate": True,
            "prompt_version": state.get("prompt_version", "current"),
            "support_response": None,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"Unexpected error: {str(e)[:100]}",
            "error_history": error_history,
            "total_cost": state.get("total_cost", 0),
            "request_costs": state.get("request_costs", []),
            "cost_summary": None
        }
@retry_with_backoff(
    max_retries=2,
    initial_delay=0.5,
    backoff_factor=2.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=True,
    exceptions=(LangChainException, Exception)
)
def standard_agent_node(state: SupportState):
    """
    Standard agent with structured output and error handling.
    Uses EscalationDecision model to determine if escalation is needed.
    Then uses SupportResponse model for the final response.
    
    Task 3: Added comprehensive error handling with retries and recovery.
    """
    try:
        with ErrorContext("standard_agent", user_id="standard_user"):
            # Task 3: Check if we've hit max errors
            if state.get("error_count", 0) >= 3:
                error = MaxIterationsError(
                    message="Max errors reached in standard agent",
                    max_iterations=3,
                    current_iteration=state.get("error_count", 0),
                    context={"node": "standard_agent"}
                )
                return {
                    "messages": [HumanMessage(content=handle_max_iterations_error(error)["message"])],
                    "should_escalate": True,
                    "prompt_version": state.get("prompt_version", "current"),
                    "support_response": None,
                    "error_count": state.get("error_count", 0) + 1,
                    "last_error": error.message,
                    "error_history": state.get("error_history", []),
                    "total_cost": state.get("total_cost", 0),
                    "request_costs": state.get("request_costs", []),
                    "cost_summary": None
                }
            
            messages = state["messages"]
            
            # Load versioned prompt based on tier
            prompt_version = state.get("prompt_version", "v1.0.0")
            
            try:
                prompt_data = prompt_manager.load_prompt("customer_support", version=prompt_version)
            except Exception as e:
                error = ToolExecutionError(
                    message=f"Failed to load prompt version {prompt_version}",
                    tool_name="prompt_manager.load_prompt",
                    original_error=e,
                    context={"version": prompt_version}
                )
                log_error_chain(e, {"node": "standard_agent", "operation": "load_prompt"})
                raise error
            
            user_message = messages[-1].content if messages else ""
            
            # Task 5: Input Sanitization for standard agent
            cleaned_message, is_suspicious_input = sanitizer.sanitize(user_message)
            if is_suspicious_input:
                structured_logger.log_agent_call(
                    user_id="standard_user",
                    agent_name="standard_agent",
                    prompt_version=prompt_version,
                    user_message=user_message,
                    response=None,
                    tokens_used=0,
                    latency_ms=0,
                    success=False,
                    error="Suspicious input detected in standard agent",
                    trace_id=state.get("trace_id")
                )
            
            # Compile 4-layer prompt
            try:
                system_prompt = prompt_manager.compile_prompt(prompt_data, cleaned_message)
            except Exception as e:
                error = ToolExecutionError(
                    message="Failed to compile prompt",
                    tool_name="prompt_manager.compile_prompt",
                    original_error=e,
                    context={"version": prompt_version}
                )
                log_error_chain(e, {"node": "standard_agent", "operation": "compile_prompt"})
                raise error
            
            # Task 5: Prompt Hardening with sandwich defense
            hardened_system_prompt = hardener.harden_system_prompt(system_prompt, cleaned_message)
            
            # First decision: Should we escalate?
            escalation_prompt = f"""
{hardened_system_prompt}

User Message: {cleaned_message}

Determine if this case needs escalation to human support.
"""
            
            escalation_start = datetime.now()
            try:
                escalation_decision = llm.with_structured_output(EscalationDecision).invoke(
                    [HumanMessage(content=escalation_prompt)]
                )
            except OutputParserException as e:
                error = InvalidOutputError(
                    message="Failed to parse escalation decision",
                    expected_format="EscalationDecision",
                    actual_output=str(e),
                    validation_error=str(e),
                    context={"node": "standard_agent", "phase": "escalation"}
                )
                log_error_chain(e, {"node": "standard_agent", "phase": "escalation"})
                raise error
            except Exception as e:
                error = LLMError(
                    message=f"LLM call failed for escalation decision: {str(e)}",
                    error_type="API_ERROR",
                    api_error=str(e),
                    context={"node": "standard_agent", "phase": "escalation"}
                )
                log_error_chain(e, {"node": "standard_agent", "phase": "escalation"})
                raise error
            
            escalation_elapsed_ms = (datetime.utcnow() - escalation_start).total_seconds() * 1000
            
            # Task 4: Track cost for escalation decision using actual usage when available
            # Original (pre-refactor) hardcoded token usage example (preserved for audit):
            # cost_info_escalation = cost_tracker.track_llm_call(
            #     user_id="standard",
            #     model="gpt-4o-mini",
            #     input_tokens=80,
            #     output_tokens=150,  # previously hardcoded estimate
            #     node="standard_agent_node_escalation"
            # )
            cost_info_escalation = _track_and_append_cost(
                state=state,
                llm_result=escalation_decision,
                model="gpt-4o-mini",
                node="standard_agent_node_escalation",
                prompt_text=escalation_prompt,
                user_tier="standard"
            )
            
            # Build message list with compiled prompt
            enhanced_messages = [
                SystemMessage(content=hardened_system_prompt),
                HumanMessage(content=cleaned_message)
            ]
            
            # Second decision: Generate structured response
            response_start = datetime.now()
            try:
                support_response = llm.with_structured_output(SupportResponse).invoke(enhanced_messages)
            except OutputParserException as e:
                error = InvalidOutputError(
                    message="Failed to parse support response",
                    expected_format="SupportResponse",
                    actual_output=str(e),
                    validation_error=str(e),
                    context={"node": "standard_agent", "phase": "response"}
                )
                log_error_chain(e, {"node": "standard_agent", "phase": "response"})
                raise error
            except Exception as e:
                error = LLMError(
                    message=f"LLM call failed for support response: {str(e)}",
                    error_type="API_ERROR",
                    api_error=str(e),
                    context={"node": "standard_agent", "phase": "response"}
                )
                log_error_chain(e, {"node": "standard_agent", "phase": "response"})
                raise error
            
            response_elapsed_ms = (datetime.utcnow() - response_start).total_seconds() * 1000
            
            # Task 5: Output Validation - Final defense before returning response
            is_valid, error_type = validator.validate(
                support_response.message,
                user_email="customer@example.com",  # Would come from state
                action="support_response",
                requires_approval=support_response.requires_approval
            )
            
            if not is_valid:
                logger.error(f"Output validation failed in standard agent: {error_type}")
                structured_logger.log_agent_call(
                    user_id="standard_user",
                    agent_name="standard_agent",
                    prompt_version=prompt_version,
                    user_message=user_message,
                    response=None,
                    tokens_used=0,
                    latency_ms=response_elapsed_ms,
                    success=False,
                    error=f"Output validation failed: {error_type}",
                    trace_id=state.get("trace_id")
                )
                # Return safe fallback message
                support_response.message = "I apologize, but I cannot provide that information. Please contact our support team directly."
            
            # Task 4: Track cost for support response using actual usage when available
            # Original (pre-refactor) hardcoded token usage example (preserved for audit):
            # cost_info_response = cost_tracker.track_llm_call(
            #     user_id="standard",
            #     model="gpt-4o-mini",
            #     input_tokens=120,
            #     output_tokens=200,  # previously hardcoded estimate
            #     node="standard_agent_node_response"
            # )
            cost_info_response = _track_and_append_cost(
                state=state,
                llm_result=support_response,
                model="gpt-4o-mini",
                node="standard_agent_node_response",
                prompt_text=(system_prompt + "\n" + user_message),
                user_tier="standard"
            )
            
            # Add cost info already handled by helper (request_costs updated)
            
            # Task 3: Log successful call
            structured_logger.log_agent_call(
                user_id="standard_user",
                agent_name="standard_agent",
                prompt_version=prompt_version,
                user_message=user_message,
                response=support_response,
                tokens_used=0,
                latency_ms=escalation_elapsed_ms + response_elapsed_ms,
                success=True,
                trace_id=state.get("trace_id")
            )
            
            # Human-readable output for standard agent
            print(f"[standard_agent] Escalation decision: should_escalate={escalation_decision.should_escalate}, reason={getattr(escalation_decision, 'escalation_reason', '')}")
            print("[standard_agent] Support response:", support_response.message)
            print(f"[standard_agent] Action: {support_response.action}, Confidence: {support_response.confidence}, Requires approval: {support_response.requires_approval}")

            return {
                "messages": [HumanMessage(content=support_response.message)],
                "should_escalate": escalation_decision.should_escalate,
                "prompt_version": prompt_version,
                "escalation_decision": escalation_decision,
                "support_response": support_response,
                "error_count": state.get("error_count", 0),
                "last_error": None,
                "error_history": state.get("error_history", []),
                "total_cost": state.get("total_cost", 0),
                "request_costs": state.get("request_costs", []),
                "cost_summary": state.get("cost_summary")
            }
    
    except (ToolExecutionError, LLMError, InvalidOutputError) as e:
        error_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(e).__name__,
            "message": e.message,
            "node": "standard_agent"
        }
        error_history = state.get("error_history", [])
        error_history.append(error_entry)
        
        # Fallback response
        fallback_response = SupportResponse(
            action="escalate_to_human",
            confidence=0.5,
            message=f"I apologize, but I encountered an issue processing your request. "
                    f"A human agent will assist you shortly.",
            requires_approval=True,
            tier_adjusted=False,
            reasoning=f"Error recovery: {e.message}"
        )
        
        structured_logger.log_agent_call(
            user_id="standard_user",
            agent_name="standard_agent",
            prompt_version=state.get("prompt_version", "unknown"),
            user_message=messages[-1].content if messages else "",
            response=fallback_response,
            tokens_used=0,
            latency_ms=0,
            success=False,
            error=str(e)
        )
        
        # Human-readable fallback output
        print("[standard_agent] Fallback response:", fallback_response.message)
        return {
            "messages": [HumanMessage(content=fallback_response.message)],
            "should_escalate": True,
            "prompt_version": state.get("prompt_version", "v1.0.0"),
            "escalation_decision": None,
            "support_response": fallback_response,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": e.message,
            "error_history": error_history,
            "total_cost": state.get("total_cost", 0),
            "request_costs": state.get("request_costs", []),
            "cost_summary": None
        }
    
    except Exception as e:
        error_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(e).__name__,
            "message": str(e),
            "node": "standard_agent"
        }
        error_history = state.get("error_history", [])
        error_history.append(error_entry)
        log_error_chain(e, {"node": "standard_agent"})
        
        print("[standard_agent] Unexpected error occurred:", str(e))
        return {
            "messages": [HumanMessage(content="An unexpected error occurred. Please try again later.")],
            "should_escalate": True,
            "prompt_version": state.get("prompt_version", "v1.0.0"),
            "escalation_decision": None,
            "support_response": None,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"Unexpected error: {str(e)[:100]}",
            "error_history": error_history,
            "total_cost": state.get("total_cost", 0),
            "request_costs": state.get("request_costs", []),
            "cost_summary": None
        }

# 5. Graph setup with Task 3 error handling
workflow = StateGraph(SupportState)
workflow.add_node("sanitize_input", sanitize_input_node)
workflow.add_node("check_tier", check_user_tier_node)
workflow.add_node("vip_agent", vip_agent_node)
workflow.add_node("standard_agent", standard_agent_node)
workflow.set_entry_point("sanitize_input")
workflow.add_edge("sanitize_input", "check_tier")
workflow.add_conditional_edges("check_tier", route_by_tier, {
    "vip": "vip_agent",
    "premium": "vip_agent",  # Premium gets VIP handling
    "standard": "standard_agent"
})
workflow.add_edge("vip_agent", END)
workflow.add_edge("standard_agent", END)

# Task 3: Compile with recursion limit to prevent infinite loops
# The recursion_limit prevents infinite loops by limiting total node executions
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="support_agent_with_tiers.png")

# ════════════════════════════════════════════════════════════════════════════════
# AGENT REQUEST WITH VERSIONED PROMPTS , STRUCTURED OUTPUT ,RETRIES ,COST TRACKING,PROMPT SECURITY
# ════════════════════════════════════════════════════════════════════════════════

try:
    trace_id = str(uuid.uuid4())
    structured_logger.log_event("request_start", "Agent request started", trace_id=trace_id)
    initial_state = {
        "messages": [HumanMessage(content="I am a VIP customer and need help with my order")],
        "should_escalate": False,
        "issue_type": "",
        "user_tier": "",
        "prompt_version": "",
        "routing_decision": None,
        "issue_analysis": None,
        "escalation_decision": None,
        "support_response": None,
        "error_count": 0,  # Task 3: Error tracking
        "last_error": None,
        "error_history": [],
        "total_cost": 0.0,  # Task 4: Cost tracking
        "request_costs": [],  # Task 4: Cost tracking
        "cost_summary": None,  # Task 4: Cost tracking
        "trace_id": trace_id,
    }

    result = app.invoke(initial_state, config={"recursion_limit": MAX_GRAPH_ITERATIONS})
except Exception as e:
    structured_logger.log_event("error", f"Graph execution failed: {str(e)}", trace_id=trace_id)
    log_error_chain(e, {"context": "graph_execution"})
    result = None
print("\n" + "="*80)
print(" AGENT RESPONSE WITH VERSIONED PROMPTS , STRUCTURED OUTPUT ,RETRIES ,COST TRACKING,PROMPT SECURITY")
print("="*80)

if result:
    # Task 3: Display error tracking info
    if result.get("error_count", 0) > 0:
        print(f"\n[WARNING] ERROR TRACKING:")
        print(f"   Total Errors: {result.get('error_count', 0)}")
        if result.get("last_error"):
            print(f"   Last Error: {result.get('last_error')}")
        
        if result.get("error_history"):
            print(f"\n   Error History ({len(result['error_history'])} entries):")
            for error_entry in result["error_history"]:
                print(f"      • {error_entry['timestamp']}: {error_entry['error_type']}")
                print(f"        {error_entry['message']}")
    
    # Display routing decision
    if result.get("routing_decision"):
        rd = result["routing_decision"]
        print(f"\n[ROUTING] ROUTING DECISION (Structured):")
        print(f"   Tier: {rd.user_tier.upper()}")
        print(f"   Confidence: {rd.confidence:.1%}")
        print(f"   Reasoning: {rd.reasoning}")
    else:
        print(f"\n[ROUTING] ROUTING DECISION:")
        print(f"   Tier: {result.get('user_tier', 'unknown').upper()}")
        print(f"   Status: Fallback tier (error recovery)")
    
    # Display support response
    if result.get("support_response"):
        sr = result["support_response"]
        print(f"\n[RESPONSE] SUPPORT RESPONSE (Structured):")
        print(f"   Action: {sr.action}")
        print(f"   Confidence: {sr.confidence:.1%}")
        print(f"   Message: {sr.message}")
        print(f"   Requires Approval: {sr.requires_approval}")
        print(f"   Tier Adjusted: {sr.tier_adjusted}")
    else:
        print(f"\n[RESPONSE] SUPPORT RESPONSE:")
        print(f"   Status: No structured response (error occurred)")
    
    # Display escalation decision if applicable
    if result.get("escalation_decision"):
        ed = result["escalation_decision"]
        print(f"\n[ESCALATION] ESCALATION DECISION (Structured):")
        print(f"   Escalate: {ed.should_escalate}")
        print(f"   Reason: {ed.escalation_reason}")
        if ed.escalation_priority:
            print(f"   Priority: {ed.escalation_priority}")
        print(f"   Suggested Team: {ed.suggested_team}")
    
    # Task 4: Display cost tracking information
    if result.get("request_costs") or result.get("total_cost", 0) > 0:
        print(f"\n[COST] TASK 4 - COST TRACKING:")
        print(f"   Total Cost: ${result.get('total_cost', 0):.6f}")
        print(f"   Total Calls: {len(result.get('request_costs', []))}")
        
        if result.get("request_costs"):
            print(f"\n   Detailed Costs Per Call:")
            for i, cost in enumerate(result.get("request_costs", []), 1):
                print(f"      Call {i}: {cost.get('node', 'unknown')}")
                print(f"         Model: {cost.get('model', 'unknown')}")
                print(f"         Input Tokens: {cost.get('input_tokens', 0)}")
                print(f"         Output Tokens: {cost.get('output_tokens', 0)}")
                print(f"         Input Cost: ${cost.get('input_cost', 0):.6f}")
                print(f"         Output Cost: ${cost.get('output_cost', 0):.6f}")
                print(f"         Total Call Cost: ${cost.get('call_cost', 0):.6f}")
    
    # Show prompt version used
    print(f"\n[OK] Prompt Version Used: {result.get('prompt_version', 'unknown')}")
    print(f"   User Tier: {result.get('user_tier', 'unknown')}")
    print(f"   Should Escalate: {result.get('should_escalate', False)}")
else:
    print("\n[ERROR] Graph execution failed. See error logs above.")

# ════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATE ROLLBACK CAPABILITY & ERROR RECOVERY
# ════════════════════════════════════════════════════════════════════════════════
structured_logger.log_event("task", "TASK 3: ROLLBACK WITH ERROR RECOVERY (30-Second Capability)")
structured_logger.log_event("info", "Available Versions:")
for version in prompt_manager.get_version_history("customer_support"):
    info = prompt_manager.get_version_info("customer_support", version)
    structured_logger.log_event("version", f"{version}: {info.get('description', '')}")

try:
    structured_logger.log_event("action", "ROLLBACK 1: Rolling back from v1.1.0 to v1.0.0...")
    rollback_result = prompt_manager.rollback_to_version("customer_support", "v1.0.0")
    structured_logger.log_event("action_result", f"Status: {rollback_result.get('message')}", trace_id=trace_id, extra={"elapsed_ms": rollback_result.get('elapsed_ms')})
except Exception as e:
    structured_logger.log_event("error", f"Rollback failed: {str(e)}", trace_id=trace_id)
    log_error_chain(e, {"context": "rollback_operation"})

try:
    structured_logger.log_event("action", "ROLLBACK 2: Rolling back from v1.0.0 to v1.1.0...")
    rollback_result = prompt_manager.rollback_to_version("customer_support", "v1.1.0")
    structured_logger.log_event("action_result", f"Status: {rollback_result.get('message')}", trace_id=trace_id, extra={"elapsed_ms": rollback_result.get('elapsed_ms')})
except Exception as e:
    structured_logger.log_event("error", f"Rollback failed: {str(e)}", trace_id=trace_id)
    log_error_chain(e, {"context": "rollback_operation"})

structured_logger.log_event("info", "Rollback History:")
for record in prompt_manager.get_rollback_history():
    structured_logger.log_event("history", f"{record['timestamp']}: {record['from_version']} -> {record['to_version']} ({record['elapsed_ms']}ms)")