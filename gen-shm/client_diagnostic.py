#!/usr/bin/env python3
"""
Gen-SHM Client Diagnostic Tool
This script helps identify and fix common deployment issues
"""

import sys
import os
from pathlib import Path
import subprocess
import platform

def print_header():
    print("=" * 60)
    print("🚀 GEN-SHM CLIENT DIAGNOSTIC TOOL")
    print("=" * 60)

def check_python_version():
    """Check Python version compatibility"""
    print("\n🐍 Python Environment Check")
    print("-" * 30)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version compatible")
        return True

def check_virtual_environment():
    """Check if running in virtual environment"""
    print("\n🔧 Virtual Environment Check")
    print("-" * 30)
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print("✅ Running in virtual environment")
        print(f"   Environment: {sys.prefix}")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("   Recommendation: Use virtual environment for isolation")
        return False

def check_working_directory():
    """Check if in correct directory"""
    print("\n📂 Directory Structure Check")
    print("-" * 30)
    
    current_dir = Path.cwd()
    required_files = ['src', 'requirements.txt']
    
    print(f"Current directory: {current_dir}")
    
    missing_items = []
    for item in required_files:
        if not (current_dir / item).exists():
            missing_items.append(item)
    
    if missing_items:
        print(f"❌ Missing required items: {missing_items}")
        print("   Please run this script from the gen-shm root directory")
        return False
    else:
        print("✅ Required directory structure found")
        return True

def check_core_dependencies():
    """Check core Python dependencies"""
    print("\n📦 Core Dependencies Check")
    print("-" * 30)
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML')
    ]
    
    missing_deps = []
    
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError:
            print(f"❌ {display_name}: Not installed")
            missing_deps.append(module_name)
    
    return len(missing_deps) == 0

def check_pytorch():
    """Check PyTorch installation"""
    print("\n🔥 PyTorch Check")
    print("-" * 30)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available (using CPU)")
            
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_gen_shm_components():
    """Check Gen-SHM specific components"""
    print("\n⚙️  Gen-SHM Components Check")
    print("-" * 30)
    
    # Add src to path temporarily
    project_root = Path(__file__).parent
    src_path = project_root / 'src'
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    components = [
        ('src.utils.config', 'Configuration Manager'),
        ('src.models.beam_physics', 'Physics Engine'),
        ('src.models.pinn_generator', 'PINN Generator')
    ]
    
    missing_components = []
    
    for module_path, component_name in components:
        try:
            __import__(module_path)
            print(f"✅ {component_name}: Loaded")
        except ImportError as e:
            print(f"❌ {component_name}: Failed to load ({e})")
            missing_components.append(component_name)
        except Exception as e:
            print(f"⚠️  {component_name}: Loaded with warnings ({e})")
    
    return len(missing_components) == 0

def check_gui_support():
    """Check GUI/tkinter support"""
    print("\n🖥️  GUI Support Check")
    print("-" * 30)
    
    try:
        import tkinter as tk
        root = tk.Tk()
        root.destroy()
        print("✅ Tkinter GUI support available")
        return True
    except ImportError:
        print("❌ Tkinter not available")
        print("   GUI demos will not work")
        print("   Install with: sudo apt-get install python3-tk (Linux)")
        return False
    except Exception as e:
        print(f"⚠️  Tkinter available but may have issues: {e}")
        return False

def run_basic_functionality_test():
    """Run basic Gen-SHM functionality test"""
    print("\n🧪 Basic Functionality Test")
    print("-" * 30)
    
    try:
        # Import components
        from src.utils.config import Config
        from src.models.beam_physics import BeamPhysics
        import torch
        
        # Test configuration
        config = Config()
        print("✅ Configuration system working")
        
        # Test physics engine
        physics = BeamPhysics(config.config)
        print("✅ Physics engine initialized")
        
        # Test basic computation
        x = torch.tensor([0.5])
        stiffness = physics.stiffness_field(x, 0.5, 0.0)
        print(f"✅ Basic physics computation: {stiffness.item():.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def provide_recommendations(all_checks):
    """Provide recommendations based on test results"""
    print("\n📋 RECOMMENDATIONS")
    print("=" * 60)
    
    failed_checks = [name for name, passed in all_checks.items() if not passed]
    
    if not failed_checks:
        print("🎉 All checks passed! Gen-SHM is ready to use.")
        print("\n🚀 To get started:")
        print("   python showcase_demo.py")
        print("   python revolutionary_demo.py")
        return
    
    print("⚠️  Some issues detected. Here's how to fix them:")
    
    if 'python_version' in failed_checks:
        print("\n🐍 Python Version Issue:")
        print("   - Install Python 3.8 or later")
        print("   - Download from python.org or use system package manager")
    
    if 'dependencies' in failed_checks:
        print("\n📦 Missing Dependencies:")
        print("   - Run: pip install -r requirements.txt")
        print("   - Or: pip install numpy scipy matplotlib pyyaml")
    
    if 'pytorch' in failed_checks:
        print("\n🔥 PyTorch Issue:")
        print("   - Run: pip install torch torchvision torchaudio")
        print("   - For CPU only: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    
    if 'working_directory' in failed_checks:
        print("\n📂 Directory Issue:")
        print("   - Navigate to the gen-shm root directory")
        print("   - The directory should contain 'src/' and 'requirements.txt'")
    
    if 'gui_support' in failed_checks:
        print("\n🖥️  GUI Issue:")
        print("   - Install tkinter: sudo apt-get install python3-tk (Ubuntu/Debian)")
        print("   - Or use text-based demos instead")
    
    print("\n🔧 Quick Fix Commands:")
    print("   cd /path/to/gen-shm")
    print("   python3 -m venv gen-shm-client-env")
    print("   source gen-shm-client-env/bin/activate")
    print("   pip install -r requirements.txt")

def main():
    """Main diagnostic function"""
    print_header()
    
    # Store results
    results = {}
    
    # Run all checks
    results['python_version'] = check_python_version()
    results['virtual_env'] = check_virtual_environment()
    results['working_directory'] = check_working_directory()
    results['dependencies'] = check_core_dependencies()
    results['pytorch'] = check_pytorch()
    results['gui_support'] = check_gui_support()
    results['gen_shm'] = check_gen_shm_components()
    results['functionality'] = run_basic_functionality_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}")
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {check_name.replace('_', ' ').title()}")
    
    # Provide recommendations
    provide_recommendations(results)
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Diagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)