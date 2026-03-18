#!/bin/bash
#
# Gen-SHM Client Setup Script
# This script prepares the environment for running Gen-SHM on client machines
#

set -e  # Exit on any error

echo "🚀 Gen-SHM Client Setup Starting..."
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date +%H:%M:%S)] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date +%H:%M:%S)] ERROR: $1${NC}"
}

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_status "Found Python $PYTHON_VERSION"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "src" ]; then
    print_error "Please run this script from the gen-shm root directory"
    exit 1
fi

# Create virtual environment
print_status "Setting up virtual environment..."
if [ ! -d "gen-shm-client-env" ]; then
    python3 -m venv gen-shm-client-env
    print_status "Virtual environment created"
else
    print_status "Using existing virtual environment"
fi

# Activate virtual environment
source gen-shm-client-env/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies with error handling
print_status "Installing dependencies..."
if pip install -r requirements.txt; then
    print_status "Dependencies installed successfully"
else
    print_warning "Some dependencies failed to install. Trying alternative approach..."
    # Install core packages individually
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install numpy scipy matplotlib
    pip install pyyaml
    print_status "Core dependencies installed"
fi

# Test installation
print_status "Testing installation..."
if python3 -c "import torch; import numpy; print('✅ Core packages working')" 2>/dev/null; then
    print_status "Installation successful!"
else
    print_error "Installation test failed. Please check error messages above."
    exit 1
fi

# Create simple test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify Gen-SHM installation"""

import sys
from pathlib import Path

def test_installation():
    print("🧪 Testing Gen-SHM Installation...")
    
    # Test Python version
    print(f"✅ Python version: {sys.version}")
    
    # Test core imports
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    # Test Gen-SHM components
    try:
        # Add src to path
        project_root = Path(__file__).parent
        src_path = project_root / 'src'
        sys.path.insert(0, str(src_path))
        
        from src.utils.config import Config
        config = Config()
        print("✅ Gen-SHM Config loaded")
        
        from src.models.beam_physics import BeamPhysics
        physics = BeamPhysics(config.config)
        print("✅ Gen-SHM Physics engine loaded")
        
        print("\n🎉 All tests passed! Gen-SHM is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Gen-SHM component test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
EOF

chmod +x test_installation.py

# Run test
print_status "Running final verification..."
if python3 test_installation.py; then
    print_status "Setup completed successfully!"
    echo ""
    echo "==================================="
    echo "🚀 Gen-SHM is ready to use!"
    echo "==================================="
    echo "To run demonstrations:"
    echo "  source gen-shm-client-env/bin/activate"
    echo "  python showcase_demo.py"
    echo "  python revolutionary_demo.py"
    echo ""
    echo "To run tests:"
    echo "  python test_installation.py"
    echo ""
    echo "For troubleshooting, check the logs above or contact support."
else
    print_error "Setup completed with issues. Please review the error messages."
    exit 1
fi