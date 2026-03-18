# Gen-SHM: Getting Started Guide

Welcome to Gen-SHM, a physics-informed generative surrogate for drone wing structural integrity monitoring!

## Quick Setup

### 1. Install Dependencies

```bash
cd gen-shm
pip install -r requirements.txt
```

### 2. Quick Demo

Try the interactive demo notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

Or run a quick test:
```bash
python tests/run_tests.py
```

## Basic Usage Examples

### Generate Synthetic Data (No Training Required)

```python
from src.models.surrogate_model import quick_train_and_generate
import matplotlib.pyplot as plt

# Generate 25 samples with 20% damage at wing root
samples = quick_train_and_generate(
    damage_level=0.2,      # 0.0 to 1.0 (stiffness reduction)
    damage_location=0.0,   # 0.0 to 1.0 (normalized position)
    num_samples=25
)

# Plot results
plt.figure(figsize=(12, 8))
for i in range(3):  # Plot first 3 sensors
    plt.subplot(3, 1, i+1)
    plt.plot(samples['time'], samples['acceleration'][0, i, :])
    plt.title(f'Sensor {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
plt.tight_layout()
plt.show()
```

### Full Training Workflow

```python
from src.models.surrogate_model import DroneWingSurrogate

# Initialize model
surrogate = DroneWingSurrogate()

# Train (takes 10-30 minutes depending on hardware)
history = surrogate.train(verbose=True)

# Generate samples with trained model
samples = surrogate.generate_samples(
    damage_level=0.15,
    damage_location=0.3,
    num_samples=50
)

# Save trained model
surrogate.save_model('my_trained_model.pt')

# Load trained model later
surrogate_loaded = DroneWingSurrogate(model_path='my_trained_model.pt')
```

## Command Line Usage

### Train Model
```bash
python experiments/train_model.py --epochs 500 --gpu 0
```

### Generate Samples
```bash
python experiments/generate_samples.py \
    --model_path trained_model.pt \
    --damage_level 0.25 \
    --damage_location 0.5 \
    --num_samples 100 \
    --plot
```

### Evaluate Performance
```bash
python experiments/evaluate_shm.py \
    --model_path trained_model.pt \
    --num_test_cases 50 \
    --plot_results \
    --physics_validation
```

## Project Structure

```
gen-shm/
├── src/                    # Source code
│   ├── models/            # Neural networks and physics
│   │   ├── beam_physics.py       # Euler-Bernoulli beam equations
│   │   ├── pinn_generator.py     # Physics-informed neural network
│   │   └── surrogate_model.py    # High-level interface
│   ├── data/              # Data generation and handling
│   ├── training/          # Training framework
│   ├── evaluation/        # Metrics and validation
│   └── utils/             # Utilities and helpers
├── experiments/           # Main scripts
├── notebooks/             # Jupyter demos
├── configs/               # Configuration files
├── tests/                 # Unit tests
└── requirements.txt       # Dependencies
```

## Key Features

### ✅ Physics-Informed
- Embeds Euler-Bernoulli beam theory directly into the neural network
- Ensures generated data satisfies fundamental physical laws
- Maintains energy conservation and boundary conditions

### ✅ Zero-Shot Capability
- Can generate data for damage scenarios never seen during training
- Generalizes to arbitrary damage locations and severities
- No need for extensive real-world failure data

### ✅ Production Ready
- Lightweight architecture suitable for edge deployment
- Comprehensive validation and testing framework
- Modular design for easy customization

## Configuration

The default configuration is in `configs/default.yaml`. Key parameters:

```yaml
physics:
  beam_length: 1.0        # Wing length in meters
  young_modulus: 70e9     # Material stiffness (Pa)
  density: 2700           # Material density (kg/m³)

damage:
  max_severity: 0.5       # Maximum damage (50% stiffness reduction)
  location_range: [0.1, 0.9]  # Valid damage locations

model:
  hidden_layers: 6        # Network depth
  hidden_dim: 128         # Network width
  activation: "swish"     # Activation function

training:
  epochs: 1000            # Training duration
  physics_points: 10000   # Physics constraint points
  loss_weights:           # Loss component importance
    data: 1.0
    physics: 10.0
    boundary: 100.0
```

## Common Use Cases

### 1. Damage Scenario Analysis
```python
# Compare different damage scenarios
scenarios = [
    {'level': 0.0, 'location': 0.5, 'name': 'Healthy'},
    {'level': 0.1, 'location': 0.2, 'name': 'Light_root_damage'},
    {'level': 0.2, 'location': 0.5, 'name': 'Moderate_center_damage'},
]

for scenario in scenarios:
    samples = surrogate.generate_samples(**scenario, num_samples=25)
    # Analyze frequency content, amplitude changes, etc.
```

### 2. Training Data Augmentation
```python
# Generate large synthetic datasets for ML training
large_dataset = surrogate.generate_samples(
    damage_level=0.15,
    damage_location=0.4,
    num_samples=1000,  # Generate 1000 samples
    duration=5.0,      # Longer time series
    sampling_rate=2000 # Higher sampling rate
)
```

### 3. Real-time Monitoring Simulation
```python
# Simulate continuous monitoring
for time_step in range(100):
    # Generate current state
    current_data = surrogate.generate_samples(
        damage_level=current_damage_estimate,
        damage_location=current_location_estimate,
        num_samples=1,
        duration=0.1  # Short window for real-time
    )
    
    # Process and detect changes...
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` in config
2. **Slow Training**: Reduce `physics_points` or use fewer `hidden_layers`
3. **Poor Physics Compliance**: Increase `physics` loss weight or training epochs
4. **Import Errors**: Make sure you're in the gen-shm directory and requirements are installed

### Performance Tips

- Use GPU if available (`--gpu 0` flag)
- Start with shorter training (`--epochs 100`) for testing
- Adjust `physics_points` based on your computational resources
- Monitor training progress with `--verbose` flag

## Next Steps

1. **Explore the demo notebook** to understand capabilities
2. **Experiment with different damage scenarios** 
3. **Customize configuration** for your specific application
4. **Integrate with your SHM system** using the generated data

## Need Help?

- Check the documentation in each module
- Run the tests: `python tests/run_tests.py`
- Review the demo notebook for examples
- Examine configuration options in `configs/default.yaml`

Happy coding! 🚀