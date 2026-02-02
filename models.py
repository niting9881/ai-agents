"""
Pydantic models for structured output.
This ensures type-safe responses from the LLM.
"""

from pydantic import BaseModel, Field
from typing import Literal

class TicketRouting(BaseModel):
    """ Simple Routing Decision """
    specialist: Literal['IT', 'HR', 'Facilities','escalate'] = Field(
        description="The specialist to route the ticket to"
    )
    reasoning: str = Field(
        description="The reasoning for the routing decision to route the ticket to the selected specialist"
    )
    confidence: float = Field(description="The confidence in the routing decision", 
    ge=0.0, le=1.0)

class SupportResponse(BaseModel):
    """Structured response from the customer support agent."""
    
    reasoning: str = Field(
        description="Your step-by-step reasoning for this decision"
    )
    action: str = Field(
        description="Action to take: assist on the specialized request area|escalate_to_human|provide_info|create_approval_ticket"
    )
    confidence: float = Field(
        description="Confidence in this decision (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    message: str = Field(
        description="Message to send to the user"
    )
    requires_approval: bool = Field(
        description="Does this action require human approval?"
    )
