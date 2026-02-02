#!/usr/bin/env python3
"""
Rollback Utility - 30-Second Prompt Rollback Script
Enables quick rollback between prompt versions for customer support agent.

Usage:
    python rollback_utility.py v1.0.0         # Rollback to specific version
    python rollback_utility.py --list          # List all versions
    python rollback_utility.py --info v1.1.0   # Show version info
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from prompt_manager import PromptManager


class RollbackUtility:
    """Utility for managing prompt rollbacks with minimal overhead."""
    
    def __init__(self, agent_name: str = "customer_support", prompts_dir: str = "prompts"):
        """Initialize rollback utility."""
        self.agent_name = agent_name
        self.pm = PromptManager(prompts_dir=prompts_dir)
    
    def list_versions(self):
        """List all available versions."""
        print("\n" + "="*70)
        print("AVAILABLE PROMPT VERSIONS")
        print("="*70)
        
        versions = self.pm.get_version_history(self.agent_name)
        
        if not versions:
            print("❌ No prompt versions found")
            return
        
        for version in versions:
            info = self.pm.get_version_info(self.agent_name, version)
            
            if 'error' in info:
                print(f"  ❌ {version}: {info['error']}")
            else:
                marker = "→ CURRENT" if version == "current" else ""
                print(f"\n  📄 {version} {marker}")
                print(f"     Description: {info.get('description', 'N/A')}")
                print(f"     Status: {info.get('status', 'N/A')}")
                print(f"     Created: {info.get('created_at', 'N/A')}")
                print(f"     Size: {info.get('file_size_bytes', 0)} bytes")
    
    def show_version_info(self, version: str):
        """Show detailed info about a specific version."""
        print(f"\n" + "="*70)
        print(f"VERSION INFO: {version}")
        print("="*70)
        
        info = self.pm.get_version_info(self.agent_name, version)
        
        if 'error' in info:
            print(f"❌ {info['error']}")
            return
        
        for key, value in info.items():
            if key not in ['file_path']:
                print(f"  {key:20s}: {value}")
        
        print(f"\n  File: {info.get('file_path', 'N/A')}")
    
    def rollback(self, version: str):
        """
        Perform quick rollback to specified version.
        
        This is a O(1) operation: Simply overwrites current.yaml with target version.
        """
        print(f"\n" + "="*70)
        print(f"ROLLING BACK TO: {version}")
        print("="*70)
        
        # Get current version info
        print("\n📊 Current Status:")
        try:
            current_info = self.pm.get_version_info(self.agent_name, "current")
            print(f"  Current Version: {current_info.get('version', 'unknown')}")
            print(f"  Description: {current_info.get('description', '')}")
        except:
            print("  Current Version: unknown")
        
        # Perform rollback
        print(f"\n⏱️  Performing rollback operation...")
        start = datetime.utcnow()
        
        result = self.pm.rollback_to_version(self.agent_name, version)
        
        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        
        if result['success']:
            print(f"  ✅ {result['message']}")
            print(f"  ⏱️  Elapsed: {result['elapsed_ms']:.1f}ms")
            print(f"  📅 Timestamp: {result['timestamp']}")
            
            # Show new current version
            print(f"\n✅ Rollback Complete!")
            try:
                new_info = self.pm.get_version_info(self.agent_name, "current")
                print(f"  New Current Version: {new_info.get('version', 'unknown')}")
                print(f"  Description: {new_info.get('description', '')}")
            except:
                pass
        
        else:
            print(f"  ❌ {result['message']}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
    
    def show_history(self):
        """Show rollback history."""
        print("\n" + "="*70)
        print("ROLLBACK HISTORY")
        print("="*70)
        
        history = self.pm.get_rollback_history()
        
        if not history:
            print("  No rollback operations in this session")
            return
        
        print(f"\n  Total Operations: {len(history)}\n")
        
        for i, record in enumerate(history, 1):
            print(f"  {i}. {record['timestamp']}")
            print(f"     From: {record['from_version']} → To: {record['to_version']}")
            print(f"     Time: {record['elapsed_ms']:.1f}ms")
            print(f"     Status: {record['status']}")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Prompt Rollback Utility - 30-Second Prompt Version Switching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rollback_utility.py v1.0.0        # Rollback to v1.0.0
  python rollback_utility.py --list        # List all versions
  python rollback_utility.py --info v1.1.0 # Show version details
  python rollback_utility.py --history     # Show rollback history
        """
    )
    
    parser.add_argument(
        'version',
        nargs='?',
        help='Version to rollback to (e.g., v1.0.0, v1.1.0)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available versions'
    )
    
    parser.add_argument(
        '--info',
        metavar='VERSION',
        help='Show detailed info about a specific version'
    )
    
    parser.add_argument(
        '--history',
        action='store_true',
        help='Show rollback history for this session'
    )
    
    parser.add_argument(
        '--agent',
        default='customer_support',
        help='Agent name (default: customer_support)'
    )
    
    parser.add_argument(
        '--prompts-dir',
        default='prompts',
        help='Prompts directory (default: prompts)'
    )
    
    args = parser.parse_args()
    
    # Initialize utility
    util = RollbackUtility(
        agent_name=args.agent,
        prompts_dir=args.prompts_dir
    )
    
    try:
        # Handle different commands
        if args.list:
            util.list_versions()
        
        elif args.info:
            util.show_version_info(args.info)
        
        elif args.history:
            util.show_history()
        
        elif args.version:
            util.rollback(args.version)
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
