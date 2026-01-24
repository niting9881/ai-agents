"""
Pydantic models for structured output with LLM.
Ensures type-safe, validated responses from the LLM using with_structured_output().
"""

from pydantic import BaseModel, Field
from typing import Literal


class RoutingDecision(BaseModel):
    """Structured routing decision for customer tier classification."""
    
    user_tier: Literal["vip", "premium", "standard"] = Field(
        description="Classified customer tier based on context"
    )
    confidence: float = Field(
        description="Confidence in the tier classification (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief reasoning for the tier classification"
    )


class IssueAnalysis(BaseModel):
    """Structured issue classification for ticket analysis."""
    
    issue_type: Literal["billing", "technical", "account", "refund", "other"] = Field(
        description="Type of issue the customer is reporting"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Priority level based on issue severity"
    )
    description: str = Field(
        description="Concise summary of the customer issue"
    )
    keywords: list[str] = Field(
        description="Key terms extracted from the customer message",
        min_items=1,
        max_items=5
    )


class EscalationDecision(BaseModel):
    """Structured escalation logic for complex cases."""
    
    should_escalate: bool = Field(
        description="Whether this case should be escalated to human support"
    )
    escalation_reason: str = Field(
        description="If escalating, explain why. Otherwise, explain why no escalation needed."
    )
    escalation_priority: Literal["standard", "high", "urgent"] | None = Field(
        description="Priority level if escalating (null if no escalation)",
        default=None
    )
    suggested_team: str = Field(
        description="Suggested team for escalation (e.g., 'billing_team', 'technical_support', 'vip_support')"
    )


class SupportResponse(BaseModel):
    """Structured response from the customer support agent with tier awareness."""
    
    reasoning: str = Field(
        description="Your step-by-step reasoning for this decision"
    )
    action: Literal["process_refund", "escalate_to_human", "provide_info", "create_approval_ticket"] = Field(
        description="Action to take based on the request and customer tier"
    )
    confidence: float = Field(
        description="Confidence in this decision (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    message: str = Field(
        description="Professional message to send to the customer",
        max_length=500
    )
    requires_approval: bool = Field(
        description="Does this action require manager approval?"
    )
    tier_adjusted: bool = Field(
        description="Whether this response was adjusted based on customer tier"
    )


class SupportOutput(BaseModel):
    """Structured output for customer support decisions (v1.0.0)."""
    
    action: Literal["process_refund", "escalate_to_human", "provide_info", "escalate_to_senior_support"] = Field(
        description="Action to take based on tier and authority"
    )
    confidence: float = Field(
        description="Confidence score for this decision (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    message: str = Field(
        description="Clear, empathetic message to customer"
    )
    reasoning: str = Field(
        description="Explicit reasoning for the decision"
    )
    requires_approval: bool = Field(
        description="Whether manager approval is needed"
    )
    tier_adjusted: bool = Field(
        description="Whether response was customized for customer tier"
    )
