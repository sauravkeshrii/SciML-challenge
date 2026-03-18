#!/usr/bin/env python3
"""
🎯 LAUNCHER FOR COMPLETE GEN-SHM DEMONSTRATION
Automatically activates virtual environment and runs the demo
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Get the project directory
    project_dir = Path(__file__).parent.resolve()
    
    # Check for virtual environment
    venv_activate = project_dir / 'gen-shm-env' / 'bin' / 'activate'
    
    if venv_activate.exists():
        print("✅ Found virtual environment, activating...")
        
        # Run the demo with virtual environment
        cmd = [
            'bash', '-c', 
            f'cd "{project_dir}" && source gen-shm-env/bin/activate && python complete_working_demo.py'
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running demo: {e}")
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
    else:
        print("⚠️  Virtual environment not found")
        print("🔧 Trying to run with system Python...")
        
        # Try running directly
        try:
            # Add src to path
            src_path = project_dir / 'src'
            sys.path.insert(0, str(src_path))
            
            # Run the demo
            demo_script = project_dir / 'complete_working_demo.py'
            with open(demo_script, 'r') as f:
                exec(f.read())
                
        except Exception as e:
            print(f"❌ Error running demo: {e}")
            print("💡 Try running: source gen-shm-env/bin/activate && python complete_working_demo.py")

if __name__ == "__main__":
    main()