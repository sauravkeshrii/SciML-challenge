# Gen-SHM: Physics-Informed Generative Surrogate for Drone Wing Structural Integrity

This repository implements a physics-informed generative surrogate model for drone wing damage detection using Physics-Informed Neural Networks (PINNs).

## Overview

The system generates synthetic vibration data for arbitrary damage scenarios in drone wings by embedding the Euler-Bernoulli beam equation into a parametric neural network. This addresses the critical lack of failure data for training structural health monitoring systems.

## Features

- **Physics-Informed Generation**: Embeds Euler-Bernoulli beam theory with spatially varying stiffness
- **Damage Parameterization**: Models crack location and severity through stiffness reduction
- **Real-time Capabilities**: Lightweight surrogate suitable for edge deployment
- **Zero-shot Detection**: Enables damage identification without destructive testing
- **Comprehensive Validation**: Physics compliance checks and SHM performance metrics

## Installation

```bash
cd gen-shm
pip install -r requirements.txt
```

## Quick Start

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

## Project Structure

```
gen-shm/
├── src/
│   ├── models/          # Neural network architectures and physics
│   ├── data/            # Data generation and preprocessing
│   ├── training/        # Training loops and optimization
│   ├── evaluation/      # Metrics and validation
│   └── utils/           # Utility functions
├── experiments/         # Main experiment scripts
├── notebooks/           # Jupyter demonstrations
├── configs/             # Configuration files
└── tests/              # Unit and integration tests
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

## Experiments

Run the main training script:
```bash
python experiments/train_model.py --config configs/default.yaml
```

Generate samples for analysis:
```bash
python experiments/generate_samples.py --damage_level 0.15 --num_samples 50
```

Evaluate SHM performance:
```bash
python experiments/evaluate_shm.py --model_path checkpoints/best_model.pt
```

## Citation

If you use this code in your research, please cite:

```
@article{gen-shm2026,
  title={Gen-SHM: Physics-Informed Generative Surrogate for Drone Wing Structural Integrity},
  author={Saurav},
  year={2026}
}
```

## License

MIT License