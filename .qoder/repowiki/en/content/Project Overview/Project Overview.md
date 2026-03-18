# Project Overview

<cite>
**Referenced Files in This Document**
- [README.md](file://README.md)
- [GETTING_STARTED.md](file://GETTING_STARTED.md)
- [default.yaml](file://configs/default.yaml)
- [beam_physics.py](file://src/models/beam_physics.py)
- [pinn_generator.py](file://src/models/pinn_generator.py)
- [surrogate_model.py](file://src/models/surrogate_model.py)
- [train_model.py](file://experiments/train_model.py)
- [generate_samples.py](file://experiments/generate_samples.py)
- [demo.ipynb](file://notebooks/demo.ipynb)
- [trainer.py](file://src/training/trainer.py)
- [validation.py](file://src/evaluation/validation.py)
- [helpers.py](file://src/utils/helpers.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)
10. [Appendices](#appendices)

## Introduction
This project implements a Physics-Informed Generative Surrogate for drone wing structural integrity, leveraging Physics-Informed Neural Networks (PINNs) to synthesize vibration data for arbitrary damage scenarios. It integrates Euler-Bernoulli beam theory into a parametric neural network, enabling zero-shot generation of structural responses under unseen damage configurations. The system targets structural health monitoring (SHM) applications where real-world failure data is scarce, offering a data-efficient, physics-constrained approach suitable for both research exploration and production-ready edge deployments.

Key goals:
- Embed Euler-Bernoulli beam dynamics directly into the model via physics constraints.
- Parameterize damage as stiffness reduction with configurable spatial influence.
- Provide a lightweight surrogate model for real-time monitoring simulations.
- Offer comprehensive validation of physics compliance and SHM performance.

## Project Structure
The repository follows a modular layout separating models, data generation, training, evaluation, utilities, and experiments. High-level directories:
- src/models: Neural network architectures and physics engines.
- src/data: Synthetic data generation and preprocessing.
- src/training: Training loops, loss functions, and optimization.
- src/evaluation: Metrics, validation, and visualization.
- src/utils: Configuration, helpers, and logging.
- experiments: CLI scripts for training, sampling, and evaluation.
- notebooks: Interactive demos and tutorials.
- configs: YAML configuration files.
- tests: Unit and integration tests.

```mermaid
graph TB
subgraph "Experiments"
ET["train_model.py"]
EG["generate_samples.py"]
end
subgraph "Models"
SM["surrogate_model.py"]
PG["pinn_generator.py"]
BP["beam_physics.py"]
end
subgraph "Data"
SDG["data_generation.py"]
end
subgraph "Training"
TR["trainer.py"]
end
subgraph "Evaluation"
VAL["validation.py"]
end
subgraph "Utils"
CFG["config.py"]
HLP["helpers.py"]
end
ET --> SM
EG --> SM
SM --> PG
PG --> BP
SM --> SDG
TR --> PG
TR --> SDG
VAL --> PG
VAL --> BP
SM --> CFG
PG --> CFG
SDG --> CFG
ET --> CFG
EG --> CFG
SM --> HLP
PG --> HLP
SDG --> HLP
```

**Diagram sources**
- [train_model.py:1-165](file://experiments/train_model.py#L1-L165)
- [generate_samples.py:1-216](file://experiments/generate_samples.py#L1-L216)
- [surrogate_model.py:1-337](file://src/models/surrogate_model.py#L1-L337)
- [pinn_generator.py:1-352](file://src/models/pinn_generator.py#L1-L352)
- [beam_physics.py:1-300](file://src/models/beam_physics.py#L1-L300)
- [data_generation.py:1-384](file://src/data/data_generation.py#L1-L384)
- [trainer.py:1-392](file://src/training/trainer.py#L1-L392)
- [validation.py:1-376](file://src/evaluation/validation.py#L1-L376)
- [config.py:1-123](file://src/utils/config.py#L1-L123)
- [helpers.py:1-161](file://src/utils/helpers.py#L1-L161)

**Section sources**
- [README.md:41-55](file://README.md#L41-L55)
- [GETTING_STARTED.md:104-122](file://GETTING_STARTED.md#L104-L122)

## Core Components
- Surrogate Model (DroneWingSurrogate): High-level interface for training, sampling, and validation. It orchestrates the PINN generator, beam physics engine, and synthetic data generation.
- PINN Generator: Parametric neural network that predicts displacement u(x,t) conditioned on spatial, temporal coordinates, and damage parameters. It computes physics-informed losses using automatic differentiation.
- Beam Physics Engine: Implements Euler-Bernoulli beam theory with spatially varying stiffness, boundary conditions, and initial conditions. Provides physics residual computation and energy checks.
- Training Framework: Adapts loss weighting, schedules learning rates, and monitors convergence. Supports early stopping and checkpointing.
- Data Generation: Produces healthy calibration data, collocation points, and damage scenarios for training and validation.
- Evaluation and Validation: Validates governing equation satisfaction, boundary/initial conditions, energy conservation, and numerical stability.

**Section sources**
- [surrogate_model.py:15-271](file://src/models/surrogate_model.py#L15-L271)
- [pinn_generator.py:39-287](file://src/models/pinn_generator.py#L39-L287)
- [beam_physics.py:12-258](file://src/models/beam_physics.py#L12-L258)
- [trainer.py:55-339](file://src/training/trainer.py#L55-L339)
- [data_generation.py:14-318](file://src/data/data_generation.py#L14-L318)
- [validation.py:16-281](file://src/evaluation/validation.py#L16-L281)

## Architecture Overview
The system combines deep learning with physical laws through PINNs. The surrogate model initializes the PINN generator and beam physics engine, then trains the model using hybrid loss that balances data fidelity, physics compliance, and boundary/initial conditions. During inference, the model generates acceleration time histories at sensor locations for specified damage scenarios.

```mermaid
sequenceDiagram
participant User as "User"
participant Script as "train_model.py"
participant Surrogate as "DroneWingSurrogate"
participant Trainer as "PINNTrainer"
participant Data as "SyntheticDataGenerator"
participant Model as "PINNGenerator"
participant Physics as "BeamPhysics"
User->>Script : "Run training"
Script->>Surrogate : "Initialize DroneWingSurrogate"
Script->>Surrogate : "train(verbose=True)"
Surrogate->>Data : "generate_training_data()"
Data-->>Surrogate : "Training tensors"
Surrogate->>Trainer : "Initialize PINNTrainer"
Trainer->>Model : "train_epoch() with batches"
Model->>Physics : "compute_physics_loss()"
Physics-->>Model : "Residual loss"
Trainer-->>Surrogate : "Training history"
Surrogate-->>Script : "Training history"
Script-->>User : "Saved model and validation report"
```

**Diagram sources**
- [train_model.py:77-162](file://experiments/train_model.py#L77-L162)
- [surrogate_model.py:131-166](file://src/models/surrogate_model.py#L131-L166)
- [trainer.py:207-297](file://src/training/trainer.py#L207-L297)
- [data_generation.py:211-263](file://src/data/data_generation.py#L211-L263)
- [pinn_generator.py:155-185](file://src/models/pinn_generator.py#L155-L185)
- [beam_physics.py:107-150](file://src/models/beam_physics.py#L107-L150)

## Detailed Component Analysis

### Surrogate Model (DroneWingSurrogate)
The surrogate model encapsulates the end-to-end workflow:
- Training: Generates synthetic data, initializes the trainer, and runs the training loop.
- Sampling: Generates acceleration time histories for specified damage levels and locations.
- Validation: Computes physics compliance metrics across damage scenarios.
- Persistence: Saves and loads model checkpoints with configuration.

```mermaid
classDiagram
class DroneWingSurrogate {
+__init__(model_path, config_dict)
+generate_samples(damage_level, damage_location, num_samples, duration, sampling_rate) Dict
+train(training_config, verbose) Dict
+detect_damage(acceleration_data) Dict
+validate_physics_compliance(num_test_points) Dict
+save_model(path) void
+load_model(path) void
+get_model_info() Dict
}
class PINNGenerator {
+forward(x, t, damage_location, damage_severity) Tensor
+predict_displacement(x, t, damage_location, damage_severity) Tensor
+compute_physics_loss(x, t, damage_location, damage_severity) Tensor
+compute_boundary_loss(x_bound, t_bound, damage_location, damage_severity) Tensor
+compute_initial_loss(x_init, t_init, damage_location, damage_severity) Tensor
+generate_acceleration(x, t, damage_location, damage_severity) Tensor
+get_model_info() Dict
}
class BeamPhysics {
+stiffness_field(x, damage_location, damage_severity) Tensor
+physics_residual(x, t, u, damage_param) Tensor
+boundary_conditions(x, t, u) Tuple
+initial_conditions(x, t, u) Tuple
+energy_conservation_check(x, t, u, damage_param) Dict
}
DroneWingSurrogate --> PINNGenerator : "uses"
DroneWingSurrogate --> BeamPhysics : "uses"
PINNGenerator --> BeamPhysics : "uses"
```

**Diagram sources**
- [surrogate_model.py:15-271](file://src/models/surrogate_model.py#L15-L271)
- [pinn_generator.py:39-287](file://src/models/pinn_generator.py#L39-L287)
- [beam_physics.py:12-258](file://src/models/beam_physics.py#L12-L258)

**Section sources**
- [surrogate_model.py:15-271](file://src/models/surrogate_model.py#L15-L271)

### PINN Generator and Physics Constraints
The PINN generator embeds physics constraints directly into the loss function:
- Input: [x, t, damage_location, damage_severity].
- Output: displacement u(x,t).
- Physics loss: Residual of the Euler-Bernoulli beam equation computed via automatic differentiation.
- Boundary/Initial losses: Enforce boundary and initial conditions.
- Acceleration generation: Uses second-order time derivatives to produce acceleration time histories.

```mermaid
flowchart TD
Start(["Forward Pass"]) --> Inputs["Stack inputs [x, t, d_loc, d_sev]"]
Inputs --> Net["Neural Network Forward"]
Net --> U["Predict u(x,t)"]
U --> Grads["Enable gradients for x,t"]
Grads --> Residual["Compute physics residual via BeamPhysics"]
Residual --> Loss["Compute physics loss (mean squared residual)"]
Loss --> End(["Return physics loss"])
```

**Diagram sources**
- [pinn_generator.py:155-185](file://src/models/pinn_generator.py#L155-L185)
- [beam_physics.py:107-150](file://src/models/beam_physics.py#L107-L150)

**Section sources**
- [pinn_generator.py:155-239](file://src/models/pinn_generator.py#L155-L239)
- [beam_physics.py:107-200](file://src/models/beam_physics.py#L107-L200)

### Euler-Bernoulli Beam Theory Integration
The beam physics engine defines:
- Governing equation: ρA ∂²u/∂t² + c ∂u/∂t + ∂²/∂x²[EI(x;d) ∂²u/∂x²] = 0.
- Spatially varying stiffness: EI(x;d) = EI₀ · (1 − φ(x;d)), where φ is a configurable damage influence function (gaussian or step).
- Boundary conditions: Clamped, simply supported, or free at both ends.
- Initial conditions: Zero displacement and velocity at t=0.
- Energy conservation check: Computes kinetic and strain energy densities and totals.

```mermaid
flowchart TD
A["Input: x, t, u(x,t), d=[d_loc, d_sev]"] --> B["Compute stiffness_field(x, d_loc, d_sev)"]
B --> C["Compute ∂²u/∂x² and ∂²u/∂t²"]
C --> D["Compute ∂²/∂x²[EI(x;d) ∂²u/∂x²]"]
D --> E["Assemble residual = ρA ∂²u/∂t² + c ∂u/∂t + ∂²/∂x²[EI(x;d) ∂²u/∂x²]"]
E --> F["Return residual"]
```

**Diagram sources**
- [beam_physics.py:107-150](file://src/models/beam_physics.py#L107-L150)

**Section sources**
- [beam_physics.py:12-258](file://src/models/beam_physics.py#L12-L258)

### Training Framework and Loss Functions
The training framework:
- Uses a composite loss combining data fidelity, physics compliance, and boundary/initial condition penalties.
- Applies adaptive loss weighting and learning rate scheduling.
- Monitors training progress and supports early stopping.
- Provides checkpointing for model persistence.

```mermaid
sequenceDiagram
participant Loader as "DataLoader"
participant Trainer as "PINNTrainer"
participant Model as "PINNGenerator"
participant Loss as "PhysicsRegularizedLoss"
participant Opt as "Optimizer"
Loader->>Trainer : "Batch of training data"
Trainer->>Model : "Forward pass"
Model-->>Trainer : "Predictions"
Trainer->>Loss : "compute_regularized_loss(model, batch)"
Loss-->>Trainer : "Total loss"
Trainer->>Opt : "Backward and step"
Trainer-->>Loader : "Next batch"
```

**Diagram sources**
- [trainer.py:127-180](file://src/training/trainer.py#L127-L180)
- [trainer.py:207-297](file://src/training/trainer.py#L207-L297)

**Section sources**
- [trainer.py:55-339](file://src/training/trainer.py#L55-L339)

### Data Generation and Collocation Points
Synthetic data generation produces:
- Healthy calibration data with sparse sensor measurements and excitation signals.
- Collocation points for physics loss: interior, boundary, and initial condition points.
- Damage scenarios with randomized locations and severities within configured bounds.

```mermaid
flowchart TD
Start(["Generate Training Data"]) --> Healthy["Generate healthy calibration data"]
Healthy --> Colloc["Generate collocation points"]
Colloc --> Damage["Generate damage scenarios"]
Damage --> Pack["Package training tensors"]
Pack --> End(["Return training_data"])
```

**Diagram sources**
- [data_generation.py:211-263](file://src/data/data_generation.py#L211-L263)

**Section sources**
- [data_generation.py:14-318](file://src/data/data_generation.py#L14-L318)

### Evaluation and Physics Validation
Validation routines assess:
- Governing equation satisfaction across damage scenarios.
- Boundary and initial condition compliance.
- Energy conservation properties.
- Numerical stability over extended simulations.

```mermaid
flowchart TD
Start(["Run Validation Suite"]) --> GE["Validate governing equation"]
GE --> BC["Validate boundary conditions"]
BC --> IC["Validate initial conditions"]
IC --> EC["Validate energy conservation"]
EC --> NS["Validate numerical stability"]
NS --> Report["Generate validation report"]
Report --> End(["Return results"])
```

**Diagram sources**
- [validation.py:250-281](file://src/evaluation/validation.py#L250-L281)

**Section sources**
- [validation.py:16-376](file://src/evaluation/validation.py#L16-L376)

## Dependency Analysis
The system exhibits clear separation of concerns:
- Surrogate model depends on PINN generator and beam physics.
- PINN generator depends on beam physics for residual computation.
- Training framework depends on data generation and loss functions.
- Experiments depend on surrogate model and evaluation utilities.
- Utilities provide shared helpers for device selection, normalization, and derivative computation.

```mermaid
graph TB
SM["surrogate_model.py"] --> PG["pinn_generator.py"]
SM --> BP["beam_physics.py"]
PG --> BP
TR["trainer.py"] --> PG
TR --> SDG["data_generation.py"]
ET["train_model.py"] --> SM
EG["generate_samples.py"] --> SM
VAL["validation.py"] --> PG
VAL --> BP
HLP["helpers.py"] --> PG
HLP --> TR
HLP --> SDG
```

**Diagram sources**
- [surrogate_model.py:1-337](file://src/models/surrogate_model.py#L1-L337)
- [pinn_generator.py:1-352](file://src/models/pinn_generator.py#L1-L352)
- [beam_physics.py:1-300](file://src/models/beam_physics.py#L1-L300)
- [trainer.py:1-392](file://src/training/trainer.py#L1-L392)
- [data_generation.py:1-384](file://src/data/data_generation.py#L1-L384)
- [train_model.py:1-165](file://experiments/train_model.py#L1-L165)
- [generate_samples.py:1-216](file://experiments/generate_samples.py#L1-L216)
- [validation.py:1-376](file://src/evaluation/validation.py#L1-L376)
- [helpers.py:1-161](file://src/utils/helpers.py#L1-L161)

**Section sources**
- [surrogate_model.py:1-337](file://src/models/surrogate_model.py#L1-L337)
- [pinn_generator.py:1-352](file://src/models/pinn_generator.py#L1-L352)
- [beam_physics.py:1-300](file://src/models/beam_physics.py#L1-L300)
- [trainer.py:1-392](file://src/training/trainer.py#L1-L392)
- [data_generation.py:1-384](file://src/data/data_generation.py#L1-L384)
- [train_model.py:1-165](file://experiments/train_model.py#L1-L165)
- [generate_samples.py:1-216](file://experiments/generate_samples.py#L1-L216)
- [validation.py:1-376](file://src/evaluation/validation.py#L1-L376)
- [helpers.py:1-161](file://src/utils/helpers.py#L1-L161)

## Performance Considerations
- Collocation points: The number of physics points, boundary points, and initial condition points directly impacts training cost and accuracy. Tune these counts according to computational budget.
- Network architecture: Depth and width influence capacity and speed. Dropout is disabled for physics problems to preserve smoothness.
- Adaptive weighting: The system adjusts loss weights dynamically to balance data fidelity and physics compliance.
- Numerical stability: Gradient clipping and careful derivative computation prevent blow-ups.
- Early stopping: Prevents overfitting and reduces training time.
- GPU utilization: Prefer GPU when available; adjust batch size accordingly.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
Common issues and remedies:
- CUDA out of memory: Reduce batch size or number of collocation points.
- Slow training: Decrease physics points, use fewer hidden layers, or switch to CPU.
- Poor physics compliance: Increase physics loss weight or training epochs; verify boundary conditions.
- Import errors: Ensure working directory is gen-shm and dependencies are installed.

Performance tips:
- Use GPU with the --gpu flag.
- Start with fewer epochs for testing.
- Adjust collocation point counts based on resources.
- Monitor progress with --verbose.

**Section sources**
- [GETTING_STARTED.md:212-226](file://GETTING_STARTED.md#L212-L226)

## Conclusion
Gen-SHM delivers a robust, physics-informed surrogate for drone wing structural health monitoring. By integrating Euler-Bernoulli beam theory into a parametric PINN, it enables zero-shot generation of vibration data across arbitrary damage scenarios, with strong validation guarantees. The modular design supports both research experimentation and production deployment, offering a practical pathway to address the scarcity of failure data in SHM applications.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### Configuration Options (YAML)
Key configuration categories and representative parameters:
- physics: beam_length, beam_width, beam_height, young_modulus, density, damping_coefficient, boundary_conditions.
- damage: min_severity, max_severity, location_range, damage_function.
- model: input_dim, output_dim, hidden_layers, hidden_dim, activation, dropout_rate.
- training: epochs, batch_size, learning_rate, optimizer, lr_scheduler, loss_weights, physics_points, boundary_points, initial_condition_points.
- data: spatial_points, temporal_points, sensor_locations, noise_level, frequency_range.
- paths: data_dir, checkpoints_dir, logs_dir, results_dir, experiments_dir.
- advanced: multiscale_training, scale_epochs, max_scales, adaptive_weighting, weight_adaptation_rate, l2_regularization, physics_regularization, gradient_clipping, numerical_tolerance.
- visualization: plot_training_progress, save_plots, plot_frequency, style.
- logging: level, format, save_to_file, console_output.

**Section sources**
- [default.yaml:4-100](file://configs/default.yaml#L4-L100)

### Usage Patterns
- Research: Explore damage scenarios, adjust configuration, and validate physics compliance.
- Production: Train on synthetic data, save checkpoints, and generate samples for downstream SHM pipelines.

Practical examples:
- Damage scenario analysis: Compare healthy versus damaged vibration signatures across multiple scenarios.
- Training data augmentation: Generate large synthetic datasets with varied damage severities and locations.
- Real-time monitoring simulation: Continuously generate short windows of acceleration data for change detection.

**Section sources**
- [GETTING_STARTED.md:169-210](file://GETTING_STARTED.md#L169-L210)
- [demo.ipynb:119-173](file://notebooks/demo.ipynb#L119-L173)