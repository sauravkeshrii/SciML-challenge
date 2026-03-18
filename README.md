# Gen-SHM: Physics-Informed Generative Surrogate for Drone Wing Structural Integrity

This repository implements a complete physics-informed generative surrogate model for drone wing damage detection using Physics-Informed Neural Networks (PINNs), with professional client-ready deployment.

## 🚀 Overview

The system generates synthetic vibration data for arbitrary damage scenarios in drone wings by embedding the Euler-Bernoulli beam equation into a parametric neural network. This addresses the critical lack of failure data for training structural health monitoring systems while providing mission-critical safety demonstrations.

**Professional Features:**
- ✅ Client-ready deployment with automated setup
- ✅ Cross-platform compatibility (Linux/macOS/Windows)
- ✅ Comprehensive diagnostic tools
- ✅ Multiple demonstration modes
- ✅ Robust error handling and graceful degradation

## 🌟 Key Features

### Core Technology
- **Physics-Informed Generation**: Embeds Euler-Bernoulli beam theory with spatially varying stiffness
- **Damage Parameterization**: Models crack location and severity through stiffness reduction
- **Real-time Capabilities**: Lightweight surrogate suitable for edge deployment
- **Zero-shot Detection**: Enables damage identification without destructive testing
- **Comprehensive Validation**: Physics compliance checks and SHM performance metrics

### Professional Deployment
- **Automated Client Setup**: One-command installation with `setup_client.sh`
- **Multiple Demo Modes**: Text-based, GUI, and diagnostic demonstrations
- **Cross-Platform Support**: Works on Linux, macOS, and Windows
- **Robust Error Handling**: Graceful degradation when components fail
- **Comprehensive Diagnostics**: Built-in troubleshooting tools

## 🛠️ Installation

### Quick Start (Recommended)
```bash
cd gen-shm
./setup_client.sh
```

### Manual Installation
```bash
cd gen-shm
python3 -m venv gen-shm-client-env
source gen-shm-client-env/bin/activate  # Linux/macOS
# gen-shm-client-env\Scripts\activate     # Windows
pip install -r requirements.txt
```

## ▶️ Quick Start

### Interactive Mission Demo (Most Impressive)
```bash
python revolutionary_demo.py
```
Shows mission-critical scenario comparing protected vs unprotected drones

### Quick Showcase Demo
```bash
python showcase_demo.py
```
Text-based demonstration of core functionality

### Diagnostic Tool
```bash
python client_diagnostic.py
```
Identifies and fixes common deployment issues

### Programmatic Usage
```python
# Generate synthetic vibration data for damage detection
from src.models.surrogate_model import DroneWingSurrogate

# Initialize model
surrogate = DroneWingSurrogate()

# Generate 100 samples with 20% crack at wing root
samples = surrogate.generate_samples(
    damage_level=0.2,
    damage_location=0.0,  # Root location
    num_samples=100
)
```

## 📁 Complete Project Structure

```
gen-shm/
├── src/                    # Core source code
│   ├── models/            # Neural network architectures and physics
│   ├── data/              # Data generation and preprocessing
│   ├── training/          # Training loops and optimization
│   ├── evaluation/        # Metrics and validation
│   └── utils/             # Utility functions
├── experiments/           # Main experiment scripts
├── notebooks/             # Jupyter demonstrations
├── configs/               # Configuration files
├── tests/                 # Unit and integration tests
├── demos/                 # Interactive demonstrations
│   ├── showcase_demo.py          # Text-based showcase
│   ├── revolutionary_demo.py     # Mission-critical GUI demo
│   └── direct_working_demo.py    # Minimal output demo
├── deployment/            # Client deployment tools
│   ├── setup_client.sh           # Automated setup script
│   ├── launch_safe.py            # Robust launcher
│   ├── client_diagnostic.py      # Diagnostic tool
│   └── README_CLIENT.md          # Client documentation
└── documentation/         # Additional guides
```

## Key Components

### Physics Foundation
- `beam_physics.py`: Implements Euler-Bernoulli beam equation with damage parameterization
- Physics residual computation using automatic differentiation
- Boundary condition enforcement

### Generative Architecture
- `pinn_generator.py`: Parametric PINN that learns solution operator across damage parameters
- Input: space-time coordinates (x,t) + damage parameter (d)
- Output: vibration response u(x,t)

### Training Framework
- Hybrid loss combining data fidelity and physics compliance
- Adaptive loss weighting strategies
- Multi-scale training progression

## 🧪 Running Experiments

### Main Training
```bash
python experiments/train_model.py --config configs/default.yaml
```

### Sample Generation
```bash
python experiments/generate_samples.py --damage_level 0.15 --num_samples 50
```

### Performance Evaluation
```bash
python experiments/evaluate_shm.py --model_path checkpoints/best_model.pt
```

### Client Diagnostics
```bash
python client_diagnostic.py
```

### Safe Launcher
```bash
python launch_safe.py
```
Provides menu-driven access to all demonstrations

## 📚 Documentation

- [Client Deployment Guide](README_CLIENT.md) - Complete setup and troubleshooting
- [Getting Started](GETTING_STARTED.md) - Developer setup guide
- [Implementation Details](IMPLEMENTATION_SUCCESS.md) - Technical architecture
- [Jupyter Notebooks](notebooks/) - Interactive tutorials

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📖 Citation

If you use this code in your research, please cite:

```
@article{gen-shm2026,
  title={Gen-SHM: Physics-Informed Generative Surrogate for Drone Wing Structural Integrity},
  author={Saurav Keshri},
  year={2026}
}
```

## 📄 License

MIT License

## 🆘 Support

For issues, questions, or contributions:
- Check the [Client Deployment Guide](README_CLIENT.md) first
- Run `python client_diagnostic.py` for automatic troubleshooting
- Review the [Getting Started](GETTING_STARTED.md) guide for developer questions
