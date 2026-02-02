"""
Input Sanitizer - Detects and handles prompt injection attempts.
First line of defense against malicious input.
"""

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Sanitizes user input to detect injection attempts."""
    
    # Common injection patterns
    INJECTION_PATTERNS = [
        r"ignore (previous|above|prior) instructions",
        r"forget (all|your|the) (previous|prior|above)",
        r"you are now",
        r"system:",
        r"from now on",
        r"disregard .* rules",
        r"reveal (your|the) (prompt|instructions)",
        r"what (are|is) your (instructions|system prompt)",
        r"override",
        r"new instructions",
    ]
    
    def check_for_injection(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text contains injection attempts.
        
        Args:
            text: Input text to check
        
        Returns:
            Tuple of (is_injection, pattern_matched)
        """
        text_lower = text.lower()
        
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True, pattern
        
        return False, None
    
    def sanitize_length(self, text: str, max_length: int = 4000) -> str:
        """
        Prevent extremely long inputs that could cause issues.
        
        Args:
            text: Input text
            max_length: Maximum allowed length
        
        Returns:
            Truncated text if needed
        """
        if len(text) > max_length:
            logger.warning(f"Input truncated from {len(text)} to {max_length} chars")
            return text[:max_length]
        return text
    
    def remove_control_characters(self, text: str) -> str:
        """
        Remove control characters that might bypass filters.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Remove null bytes, zero-width characters, etc.
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        cleaned = cleaned.replace('\u200b', '')  # Zero-width space
        cleaned = cleaned.replace('\ufeff', '')   # Zero-width no-break space
        return cleaned
    
    def sanitize(self, text: str) -> Tuple[str, bool]:
        """
        Full sanitization pipeline.
        
        Args:
            text: Raw user input
        
        Returns:
            Tuple of (cleaned_text, is_suspicious)
        """
        # Remove control characters first
        text = self.remove_control_characters(text)
        
        # Check for injection
        is_injection, pattern = self.check_for_injection(text)
        if is_injection:
            logger.warning(f"Injection attempt detected: {pattern}")
            # Don't reject - just flag and monitor
            # Rejecting might cause false positives
        
        # Length check
        text = self.sanitize_length(text)
        
        return text, is_injection
