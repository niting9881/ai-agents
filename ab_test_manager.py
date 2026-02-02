"""
A/B Test Manager - Handles variant routing for prompt testing.
Deterministic assignment ensures users get consistent variants.
"""

import hashlib
from typing import Dict, Literal, Optional


class ABTestManager:
    """
    Manages A/B tests for prompt versions.
    
    Uses deterministic hashing to ensure users always get the same variant.
    """
    
    def __init__(self):
        """Initialize with default test configuration."""
        self.active_tests: Dict[str, Dict] = {
            "customer_support": {
                "test_id": "prompt_verbosity_test_001",
                "variants": {
                    "control": {"version": "v1.0.0", "traffic": 0.5},
                    "treatment": {"version": "v1.1.0", "traffic": 0.5}
                },
                "start_date": "2024-01-15",
                "sample_size_target": 1000  # per variant
            }
        }
    
    def get_variant(self, test_id: str, user_id: str) -> str:
        """
        Deterministically assign user to variant.
        
        Args:
            test_id: Test identifier
            user_id: User identifier
        
        Returns:
            Variant name ("control" or "treatment")
        """
        # Hash user_id + test_id for stable assignment
        hash_input = f"{user_id}_{test_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Convert to 0-1 range
        normalized = (hash_value % 10000) / 10000
        
        # Get test config
        test_config = None
        for agent_name, config in self.active_tests.items():
            if config.get("test_id") == test_id:
                test_config = config
                break
        
        if not test_config:
            return "control"
        
        # Route based on traffic allocation
        threshold = test_config['variants']['control']['traffic']
        return "control" if normalized < threshold else "treatment"
    
    def get_prompt_version(self, agent_name: str, user_id: str) -> str:
        """
        Get the right prompt version for this user.
        
        Args:
            agent_name: Name of the agent (e.g., "customer_support")
            user_id: User identifier
        
        Returns:
            Prompt version string (e.g., "v1.0.0" or "current")
        """
        if agent_name not in self.active_tests:
            return "current"  # No active test
        
        test_config = self.active_tests[agent_name]
        variant = self.get_variant(test_config['test_id'], user_id)
        
        return test_config['variants'][variant]['version']
    
    def add_test(
        self,
        agent_name: str,
        test_id: str,
        control_version: str,
        treatment_version: str,
        traffic_split: float = 0.5
    ):
        """
        Add a new A/B test.
        
        Args:
            agent_name: Name of the agent
            test_id: Unique test identifier
            control_version: Version for control group
            treatment_version: Version for treatment group
            traffic_split: Fraction of traffic for control (0.0 to 1.0)
        """
        self.active_tests[agent_name] = {
            "test_id": test_id,
            "variants": {
                "control": {"version": control_version, "traffic": traffic_split},
                "treatment": {"version": treatment_version, "traffic": 1.0 - traffic_split}
            },
            "start_date": None,  # Set when test starts
            "sample_size_target": 1000
        }
    
    def disable_test(self, agent_name: str):
        """Disable an active test (fall back to 'current' version)."""
        if agent_name in self.active_tests:
            del self.active_tests[agent_name]
