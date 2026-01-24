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
    
    # Common injection patterns - expanded set for better detection
    INJECTION_PATTERNS = [
        # Direct override attempts
        r"ignore (previous|above|prior|all) (instructions|rules|directions)",
        r"forget (all|your|the) (previous|prior|above)",
        r"you are now",
        r"system:",
        r"from now on",
        r"disregard .* (rules|instructions|directives)",
        r"reveal (your|the|my) (prompt|instructions|system prompt)",
        r"what (are|is) your (instructions|system prompt|instructions)",
        r"override",
        r"new instructions",
        # Context switching
        r"pretend (you are|you were|you're)",
        r"act as (if|like|though)",
        r"switch (to|into)",
        r"(start|begin|execute) new (role|task|mission)",
        # Encoding/obfuscation bypass attempts
        r"decode (this|that|the)",
        r"translate.*to (plain|clear|text)",
        r"base64",
        r"rot13",
        # Delimiter/escape sequences
        r"<system>",
        r"</system>",
        r"\$SYSTEM",
        r"\[SYSTEM\]",
        r"<!--.*-->",
        # Jailbreak variants
        r"DAN:",
        r"jailbreak",
        r"ignore (my|your) (instructions|rules)",
        r"print (instructions|prompt|directive)",
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
    
    def detect_encoding_attacks(self, text: str) -> bool:
        """
        Detect attempts to use encoding to bypass filters.
        
        Args:
            text: Input text
        
        Returns:
            True if encoding attack detected
        """
        encoding_patterns = [
            r'%[0-9a-fA-F]{2}',  # URL encoding
            r'&#\d+;',  # HTML entity encoding
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
        ]
        
        for pattern in encoding_patterns:
            if re.search(pattern, text):
                logger.warning(f"Encoding attack detected: {pattern}")
                return True
        
        return False
    
    def detect_repetition_attacks(self, text: str) -> bool:
        """
        Detect padding attacks that repeat instructions to obscure intent.
        
        Args:
            text: Input text
        
        Returns:
            True if repetition attack detected
        """
        # Check for repeated characters/phrases (common in padding attacks)
        # Look for patterns like "ignore instructions ignore instructions ignore instructions"
        lines = text.split('\n')
        if len(lines) > 10:
            # Check if many lines are very similar
            line_counts = {}
            for line in lines[:20]:  # Check first 20 lines
                line_lower = line.lower().strip()
                if line_lower:
                    line_counts[line_lower] = line_counts.get(line_lower, 0) + 1
            
            # If same line appears 3+ times, it might be padding
            if any(count >= 3 for count in line_counts.values()):
                logger.warning("Repetition/padding attack detected")
                return True
        
        return False

    def should_block(self, text: str) -> bool:
        """
        Decide whether sanitized text should be blocked outright.

        This is a conservative helper used by integration tests. It returns
        True when clear injection or encoding/repetition attacks are detected.
        """
        if not text:
            return True

        is_injection, _ = self.check_for_injection(text)
        if is_injection:
            return True

        if self.detect_encoding_attacks(text):
            return True

        if self.detect_repetition_attacks(text):
            return True

        return False
