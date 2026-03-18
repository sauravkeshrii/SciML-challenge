# 🚀 Gen-SHM Client Deployment Guide

## 📋 System Requirements

**Minimum Requirements:**
- Python 3.8 or later
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for initial setup

**Recommended Specifications:**
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with CUDA support (optional but faster)
- Linux/macOS/Windows 10+

## 🛠️ Installation Methods

### Method 1: Automated Setup (Recommended)
```bash
# Clone or extract the package
cd gen-shm

# Run the setup script
./setup_client.sh

# Activate environment and test
source gen-shm-client-env/bin/activate
python test_installation.py
```

### Method 2: Manual Installation
```bash
# Create virtual environment
python3 -m venv gen-shm-client-env
source gen-shm-client-env/bin/activate

# Install dependencies
pip install -r client_requirements.txt

# Test installation
python test_installation.py
```

## 🚨 Common Client Issues & Solutions

### 1. **Virtual Environment Problems**
**Issue**: `Permission denied` or `command not found`
**Solution**: 
```bash
# Use user flag for restricted systems
python3 -m venv gen-shm-client-env --without-pip
# Or use conda if available
conda create -n gen-shm-client python=3.9
conda activate gen-shm-client
```

### 2. **PyTorch Installation Failures**
**Issue**: CUDA/cuDNN compatibility errors
**Solution**:
```bash
# CPU-only version (works everywhere)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or let the setup script handle it automatically
```

### 3. **Import Errors**
**Issue**: `No module named 'src'` or `ModuleNotFoundError`
**Solution**: Always run from the project root directory:
```bash
cd /path/to/gen-shm
source gen-shm-client-env/bin/activate
python showcase_demo.py
```

### 4. **GUI Display Issues**
**Issue**: Tkinter not available or display problems
**Solution**:
```bash
# Install tkinter (Ubuntu/Debian)
sudo apt-get install python3-tk

# Or use headless mode
export MPLBACKEND=Agg
python showcase_demo.py  # Uses text-only output
```

### 5. **Path Configuration**
**Issue**: Python can't find modules
**Solution**: The setup script handles this automatically, but manual fix:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/gen-shm"
```

## 🧪 Testing Your Installation

Run the built-in test suite:
```bash
source gen-shm-client-env/bin/activate
python test_installation.py
```

Expected output:
```
🧪 Testing Gen-SHM Installation...
✅ Python version: 3.x.x
✅ PyTorch: 2.x.x
✅ NumPy: 1.xx.x
✅ Gen-SHM Config loaded
✅ Gen-SHM Physics engine loaded

🎉 All tests passed! Gen-SHM is ready to use.
```

## ▶️ Running Demos

### Quick Start Demo
```bash
python showcase_demo.py
```
Non-interactive demonstration showing core functionality

### Interactive Mission Demo
```bash
python revolutionary_demo.py
```
GUI-based mission-critical scenario comparison

### Command Line Interface
```bash
python direct_working_demo.py
```
Text-based output for headless environments

## 📁 Package Structure

```
gen-shm/
├── src/                    # Core source code
│   ├── models/            # Neural networks and physics
│   ├── data/              # Data generation utilities
│   └── utils/             # Configuration and helpers
├── experiments/           # Training and evaluation scripts
├── tests/                 # Test suite
├── notebooks/             # Jupyter demonstrations
├── setup_client.sh        # Automated setup script
├── test_installation.py   # Verification script
├── client_requirements.txt # Pinned dependencies
└── README_CLIENT.md       # This file
```

## 🔧 Troubleshooting

### If setup fails:
1. Check Python version: `python3 --version`
2. Ensure internet connectivity
3. Try manual installation method
4. Check system permissions

### If demos don't run:
1. Verify virtual environment is activated
2. Run from project root directory
3. Check that all tests pass
4. Review error messages carefully

### Performance Issues:
1. Close other applications
2. Ensure sufficient RAM (4GB minimum)
3. Consider using CPU-only PyTorch if GPU issues
4. Monitor system resources during execution

## 🆘 Support Information

**Common Error Messages:**

- **"ModuleNotFoundError"**: Wrong directory or environment not activated
- **"CUDA out of memory"**: Reduce batch sizes or use CPU version
- **"Display connection failed"**: Use headless mode or install tkinter
- **"Permission denied"**: Run with appropriate user privileges

**Getting Help:**
1. Run `python test_installation.py` first
2. Capture complete error output
3. Note your operating system and Python version
4. Check if you're in the correct directory

## 📈 Performance Expectations

**Typical Runtime:**
- Quick demo: 10-15 seconds
- Interactive demo: 2-3 minutes
- Full training: 2-4 hours (GPU), 6-12 hours (CPU)

**Resource Usage:**
- Memory: 2-4GB during operation
- CPU: Moderate usage
- GPU: Optional but significantly faster

## 🔒 Security Notes

- All code runs locally - no internet required after setup
- No data collection or transmission
- Virtual environment isolates dependencies
- Open source - full transparency of all components

---
*For commercial support or customization inquiries, please contact the development team.*