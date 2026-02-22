# 🧠 Gen-SHM: Physics-Informed Generative Surrogate for Drone Wing Structural Integrity

**Track 1: Scientific ML + Generative AI**  
*Saurav*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![Physics-Informed](https://img.shields.io/badge/Physics--Informed-NN-success)](#)

> **Gen-SHM** is a physics-informed generative surrogate that creates realistic, physically consistent synthetic vibration data for drone wing damage detection. By embedding the Euler-Bernoulli beam equation into a parametric neural network, it generates high-fidelity acceleration time histories for arbitrary damage scenarios—solving the data scarcity problem in structural health monitoring (SHM).

---

## 🚀 Why Gen-SHM?

- **Zero-shot damage detection** — Drones can recognize unseen failure patterns without destructive testing.
- **Physics-grounded generation** — Unlike GANs or diffusion models, our outputs obey conservation laws (energy, momentum).
- **Massive synthetic datasets** — Generate 100k+ labeled vibration signals from sparse healthy-state calibration data.
- **Edge-ready** — Train lightweight classifiers (e.g., 1D CNNs) for onboard SHM.

---

## 🔥 Key Features

- **Physics-Informed Neural Network (PINN)**  
  Solves the dynamic Euler-Bernoulli beam equation with a parametric damage field.
- **Continuous damage parameterization**  
  Model arbitrary crack locations and severities via stiffness reduction \( EI(x;d) = EI_0(1 - d \cdot \phi(x)) \).
- **Generative at scale**  
  Sample countless vibration signals for any damage scenario by varying implicit seeds.
- **Automatic differentiation**  
  Physics residuals computed directly from network outputs—no numerical solvers needed.

---

## 🧠 How It Works

### 1. Governing Equation (Euler-Bernoulli Beam)

\[
\rho A\frac{\partial^2u}{\partial t^2} + c\frac{\partial u}{\partial t} + \frac{\partial^2}{\partial x^2}\left(EI(x;d)\frac{\partial^2u}{\partial x^2}\right) = 0
\]

where  
- \( u(x,t) \) : vertical displacement  
- \( EI(x;d) \) : stiffness field with damage \( d \)  
- \( \rho A \) : mass per unit length  
- \( c \) : damping coefficient  

### 2. Neural Surrogate

The network \( \mathcal{G}(x,t;d) \) directly outputs the displacement:

\[
\hat{u}(x,t) = \mathcal{G}(x,t;d)
\]

### 3. Training Loss

\[
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}
\]

- **Data loss** \( \mathcal{L}_{\text{data}} \) : matches healthy-state (\(d=0\)) calibration data  
- **Physics loss** \( \mathcal{L}_{\text{physics}} \) : enforces PDE residual for any \(d\)

Once trained, \( \mathcal{G} \) becomes a **generative engine** for arbitrary damage parameters.

---

## 📁 Dataset & Calibration

- **Foundation**: Euler-Bernoulli theory (no training data required for damage states)
- **Calibration**: 3–5 accelerometers on a healthy wing → identifies baseline \(EI_0\), \(c\)
- **Output**: Synthetic time histories at sensor locations for any \(d\) (location + severity)

> Example: Generate 100 signals for a 20% crack at the root → 100 physically consistent acceleration traces.


## ⚙️ Installation
git clone https://github.com/yourusername/Gen-SHM.git
cd Gen-SHM
pip install -r requirements.txt

## 🧪 Usage
1. Calibrate healthy baseline:
from genshm import Calibrator

cal = Calibrator(sensor_positions=[0.2, 0.5, 0.8])
healthy_params = cal.fit(healthy_data)

2. Train physics-informed generator:
from genshm import Generator

gen = Generator(healthy_params)
gen.train(epochs=10000, lambda_physics=0.1)

3. Generate synthetic damage data

signals = gen.generate(
    damage_location=0.1,
    damage_severity=0.2,
    num_samples=100,
    sensor_positions=[0.2, 0.5, 0.8]
)

## 📊 Results

    ✅ Physical consistency: PDE residuals < 1e-4 on test damage scenarios

    ✅ Data efficiency: Trained with only 3 healthy-state sensors

    ✅ Zero-shot generalization: Detects unseen damage patterns with >95% accuracy (via downstream CNN)

## 🎥 Demo Video

A demonstration of the physics-informed generative surrogate and vibration signal generation:

▶️ [Watch Demo Video](https://drive.google.com/file/d/1y3yWWa5sb_bPDzF9LiHO9Ym5AvlRvYvv/view?usp=sharing)

## 👨‍🔬 Author

Saurav
Track 1: Scientific ML + Generative AI
Challenge: Vizuara Ai Labs
- [Linkedin](https://www.linkedin.com/in/saurav-keshri-9352691b0/)

## 📜 License

MIT © 2025 Saurav
text

