"""
Prompt Manager - Handles versioning and compilation of 4-layer prompts.
This module implements file-based prompt versioning with Git.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class PromptManager:
    """Manages versioned prompts stored as YAML files."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.rollback_history = []
    
    def load_prompt(self, agent_name: str, version: str = "current") -> Dict[str, Any]:
        """
        Load a versioned prompt from disk.
        
        Args:
            agent_name: Name of the agent (e.g., "customer_support")
            version: Version to load (e.g., "v1.0.0" or "current")
        
        Returns:
            Dictionary containing prompt data
        """
        if version == "current":
            prompt_file = self.prompts_dir / agent_name / "current.yaml"
        else:
            prompt_file = self.prompts_dir / agent_name / f"{version}.yaml"
        
        if not prompt_file.exists():
            raise ValueError(f"Prompt not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)
        
        # Add metadata
        prompt_data['loaded_at'] = datetime.utcnow().isoformat()
        prompt_data['file_path'] = str(prompt_file)
        
        return prompt_data
    
    def get_version_history(self, agent_name: str) -> list[str]:
        """List all versions for an agent."""
        agent_dir = self.prompts_dir / agent_name
        
        if not agent_dir.exists():
            return []
        
        versions = []
        
        # Add 'current' if it exists
        current_file = agent_dir / "current.yaml"
        if current_file.exists():
            versions.append("current")
        
        # Add versioned files (v*.yaml)
        versions.extend([
            f.stem for f in agent_dir.glob("v*.yaml")
        ])
        
        return versions
    
    def get_version_info(self, agent_name: str, version: str) -> Dict[str, Any]:
        """
        Get metadata about a specific version.
        
        Args:
            agent_name: Name of the agent
            version: Version identifier (e.g., "v1.0.0", "current")
        
        Returns:
            Dictionary with version metadata
        """
        try:
            prompt_data = self.load_prompt(agent_name, version)
            
            return {
                'version': prompt_data.get('version', version),
                'description': prompt_data.get('description', 'No description'),
                'status': prompt_data.get('status', 'unknown'),
                'created_at': prompt_data.get('created_at', 'unknown'),
                'loaded_at': prompt_data.get('loaded_at', ''),
            }
        except Exception as e:
            return {
                'version': version,
                'description': f'Error loading version: {str(e)}',
                'status': 'error',
                'created_at': 'unknown',
            }
    
    def rollback_to_version(self, agent_name: str, version: str) -> Dict[str, Any]:
        """
        Rollback current.yaml to a specific version (< 50ms operation).
        
        Args:
            agent_name: Name of the agent
            version: Version to rollback to (e.g., "v1.0.0")
        
        Returns:
            Dictionary with rollback status and timing
        """
        import time
        import shutil
        
        start_time = time.time()
        
        try:
            # Resolve source version file
            if version == "current":
                source_file = self.prompts_dir / agent_name / "current.yaml"
            else:
                source_file = self.prompts_dir / agent_name / f"{version}.yaml"
            
            if not source_file.exists():
                return {
                    'success': False,
                    'message': f'Version not found: {version}',
                    'elapsed_ms': 0,
                }
            
            # Get current version before rollback
            current_file = self.prompts_dir / agent_name / "current.yaml"
            current_version = "unknown"
            if current_file.exists():
                try:
                    with open(current_file, 'r', encoding='utf-8') as f:
                        current_data = yaml.safe_load(f)
                        current_version = current_data.get('version', 'unknown')
                except:
                    current_version = "unknown"
            
            # Copy version file to current.yaml
            shutil.copy2(source_file, current_file)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Track in rollback history
            self.rollback_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'from_version': current_version,
                'to_version': version,
                'elapsed_ms': round(elapsed_ms, 4),
            })
            
            return {
                'success': True,
                'message': f'Rolled back from {current_version} to {version}',
                'elapsed_ms': round(elapsed_ms, 4),
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Rollback failed: {str(e)}',
                'elapsed_ms': (time.time() - start_time) * 1000,
            }
    
    def compile_prompt(self, prompt_data: Dict[str, Any], user_message: str) -> str:
        """
        Compile the 4-layer prompt into final text.
        
        Layers:
        1. Security (top) - Sandwich defense
        2. Role & Constraints
        3. Context & Examples
        4. Task (current request)
        5. Security (bottom) - Final check
        """
        
        # Layer 4 (Top): Security Guards
        security_top = prompt_data.get('security', {}).get('top_guard', '')
        
        # Layer 1: Role & Constraints
        role_section = self._format_role(prompt_data.get('role', {}))
        constraints_section = self._format_constraints(prompt_data.get('constraints', {}))
        
        # Layer 2: Context & Examples
        context_section = self._format_context(prompt_data.get('context', {}))
        examples_section = self._format_examples(prompt_data.get('examples', []))
        
        # Layer 3: Current Task
        task_section = f"""
CURRENT REQUEST:
User: {user_message}

Your task: Analyze this request and respond according to the guidelines above.
"""
        
        # Layer 4 (Bottom): Security Guards
        security_bottom = prompt_data.get('security', {}).get('bottom_guard', '')
        
        # Assemble in order (Sandwich Defense)
        full_prompt = f"""
{security_top}

{role_section}

{constraints_section}

{context_section}

{examples_section}

{task_section}

{security_bottom}
"""
        
        return full_prompt.strip()
    
    def get_rollback_history(self) -> list[Dict[str, Any]]:
        """
        Get the history of rollback operations performed.
        
        Returns:
            List of rollback operation records
        """
        return self.rollback_history
    
    def _format_role(self, role: Dict) -> str:
        """Format Layer 1: Role section."""
        if not role:
            return ""
        
        identity = role.get('identity', '')
        expertise = role.get('expertise', '')
        tone = role.get('tone', '')
        
        return f"""
ROLE:
You are: {identity}
Your expertise: {expertise}
Your tone: {tone}
"""
    
    def _format_constraints(self, constraints: Dict) -> str:
        """Format Layer 1: Constraints section."""
        if not constraints:
            return ""
        
        lines = ["CONSTRAINTS:"]
        
        if 'monetary_limit' in constraints:
            lines.append(f"- Monetary limit: {constraints['monetary_limit']}")
        
        if 'data_access' in constraints:
            lines.append(f"- Data access: {constraints['data_access']}")
        
        if 'scope' in constraints:
            lines.append(f"- Scope: {constraints['scope']}")
        
        if 'prohibited_actions' in constraints:
            lines.append("- Prohibited actions:")
            for action in constraints['prohibited_actions']:
                lines.append(f"  • {action}")
        
        return "\n".join(lines)
    
    def _format_context(self, context: Dict) -> str:
        """Format Layer 2: Context section."""
        if not context:
            return ""
        
        lines = ["CONTEXT:"]
        
        if 'company_info' in context:
            lines.append("Company Information:")
            for info in context['company_info']:
                lines.append(f"  • {info}")
        
        if 'processes' in context:
            lines.append("\nProcess Flows:")
            for process_name, steps in context['processes'].items():
                lines.append(f"  {process_name}:")
                for step in steps:
                    lines.append(f"    - {step}")
        
        return "\n".join(lines)
    
    def _format_examples(self, examples: list) -> str:
        """Format Layer 2: Examples section."""
        if not examples:
            return ""
        
        lines = ["EXAMPLES:"]
        
        for i, example in enumerate(examples, 1):
            scenario = example.get('scenario', '')
            user = example.get('user', '')
            correct_response = example.get('correct_response', {})
            
            lines.append(f"\nExample {i}: {scenario}")
            lines.append(f"  User: {user}")
            lines.append(f"  Correct Response:")
            lines.append(f"    Reasoning: {correct_response.get('reasoning', '')}")
            lines.append(f"    Action: {correct_response.get('action', '')}")
            lines.append(f"    Message: {correct_response.get('message', '')}")
        
        return "\n".join(lines)