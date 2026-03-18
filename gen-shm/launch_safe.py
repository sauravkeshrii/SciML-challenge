#!/usr/bin/env python3
"""
Gen-SHM Robust Launcher
Handles common client deployment issues automatically
"""

import sys
import os
from pathlib import Path
import subprocess

def setup_environment():
    """Setup environment and handle common issues"""
    print("🔧 Setting up Gen-SHM environment...")
    
    # Add src to path
    project_root = Path(__file__).parent.resolve()
    src_path = project_root / 'src'
    
    # Insert at beginning to take priority
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Change to project directory
    os.chdir(str(project_root))
    
    print(f"✅ Working directory: {project_root}")
    print(f"✅ Source path added: {src_path}")

def check_and_install_dependencies():
    """Check and install missing dependencies"""
    print("📦 Checking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print("🔧 Installing missing packages...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ Packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages automatically")
            print("💡 Please run: pip install torch numpy scipy")
            return False
    
    return True

def run_demo_safely(demo_name):
    """Run demo with error handling"""
    demos = {
        'showcase': 'showcase_demo.py',
        'revolutionary': 'revolutionary_demo.py',
        'direct': 'direct_working_demo.py',
        'diagnostic': 'client_diagnostic.py'
    }
    
    if demo_name not in demos:
        print(f"❌ Unknown demo: {demo_name}")
        print(f"Available demos: {list(demos.keys())}")
        return False
    
    demo_file = demos[demo_name]
    demo_path = Path(__file__).parent / demo_file
    
    if not demo_path.exists():
        print(f"❌ Demo file not found: {demo_file}")
        return False
    
    print(f"🚀 Running {demo_name} demo...")
    
    try:
        # Execute the demo file
        with open(demo_path, 'r') as f:
            exec(f.read(), {'__name__': '__main__'})
        return True
    except KeyboardInterrupt:
        print(f"\n👋 {demo_name} demo interrupted")
        return True
    except Exception as e:
        print(f"❌ Error running {demo_name} demo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main launcher function"""
    print("=" * 50)
    print("🚀 GEN-SHM ROBUST LAUNCHER")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_and_install_dependencies():
        print("❌ Dependency installation failed")
        return False
    
    # Get user choice
    print("\nAvailable options:")
    print("1. Quick Showcase Demo (text-based)")
    print("2. Revolutionary Demo (GUI - mission critical)")
    print("3. Direct Working Demo (minimal output)")
    print("4. Run Diagnostic Tool")
    print("5. Exit")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        demo_map = {
            '1': 'showcase',
            '2': 'revolutionary',
            '3': 'direct',
            '4': 'diagnostic'
        }
        
        if choice in demo_map:
            success = run_demo_safely(demo_map[choice])
            if success:
                print(f"\n✅ {demo_map[choice]} demo completed successfully!")
            else:
                print(f"\n❌ {demo_map[choice]} demo encountered issues")
        elif choice == '5':
            print("👋 Goodbye!")
            return True
        else:
            print("❌ Invalid choice")
            return False
            
    except KeyboardInterrupt:
        print("\n\n👋 Launcher interrupted")
        return True
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)