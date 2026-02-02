"""
Output Validator - Validates LLM outputs for security violations.
Final defense against PII leakage and system exposure.
"""

import re
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class OutputValidator:
    """Validates LLM outputs before returning to users."""
    
    # PII patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    CREDIT_CARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    
    def __init__(self, allowed_emails: List[str] = None):
        """
        Initialize validator.
        
        Args:
            allowed_emails: List of email addresses that are safe to include
        """
        self.allowed_emails = allowed_emails or []
    
    def check_for_pii_leakage(self, text: str, employee_email: str) -> Tuple[bool, List[str]]:
        """
        Check if output contains PII from other users.
        
        Args:
            text: Output text to check
            employee_email: Current user's email (allowed)
        
        Returns:
            Tuple of (has_pii, violations)
        """
        violations = []
        
        # Check for emails
        emails = re.findall(self.EMAIL_PATTERN, text)
        for email in emails:
            if email != employee_email and email not in self.allowed_emails:
                violations.append(f"Unauthorized email: {email}")
        
        # Check for phone numbers
        phones = re.findall(self.PHONE_PATTERN, text)
        if phones:
            violations.append(f"Phone number found: {phones[0]}")
        
        # Check for credit cards
        credit_cards = re.findall(self.CREDIT_CARD_PATTERN, text)
        if credit_cards:
            violations.append("Credit card pattern found")
        
        return len(violations) > 0, violations
    
    def check_for_system_exposure(self, text: str) -> bool:
        """
        Check if output exposes system internals.
        
        Args:
            text: Output text to check
        
        Returns:
            True if system exposure detected
        """
        exposure_patterns = [
            r"system prompt",
            r"instruction:",
            r"<system>",
            r"```yaml",  # Might be leaking prompt files
            r"SECURITY CONSTRAINTS",
            r"Layer \d+",
        ]
        
        text_lower = text.lower()
        for pattern in exposure_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def check_monetary_violations(
        self, 
        action: str, 
        requires_approval: bool,
        refund_amount: Optional[float] = None,
        limit: float = 5000.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if monetary action violates limits.
        
        Args:
            action: Action being taken
            requires_approval: Whether approval is required
            refund_amount: Amount of refund (if applicable)
            limit: Monetary limit
        
        Returns:
            Tuple of (is_violation, reason)
        """
        if action == "process_refund":
            if refund_amount and refund_amount > limit:
                if not requires_approval:
                    return True, f"Large refund ({refund_amount}) without approval"
        
        return False, None
    
    def validate(
        self, 
        response_text: str, 
        employee_email: str,
        action: Optional[str] = None,
        requires_approval: Optional[bool] = None,
        refund_amount: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate output before returning to user.
        
        Args:
            response_text: The message text to validate
            employee_email: Current user's email
            action: Action being taken (for monetary checks)
            requires_approval: Whether approval is required
            refund_amount: Refund amount if applicable
        
        Returns:
            Tuple of (is_valid, error_type)
        """
        # Check for PII leakage
        has_pii, violations = self.check_for_pii_leakage(response_text, employee_email)
        if has_pii:
            logger.error(f"PII leakage detected: {violations}")
            return False, "output_contains_pii"
        
        # Check for system exposure
        if self.check_for_system_exposure(response_text):
            logger.error("System information exposure detected")
            return False, "system_exposure"
        
        # Check monetary violations
        if action:
            is_violation, reason = self.check_monetary_violations(
                action, requires_approval, refund_amount
            )
            if is_violation:
                logger.error(f"Monetary violation: {reason}")
                return False, "monetary_violation"
        
        # All checks passed
        return True, None
