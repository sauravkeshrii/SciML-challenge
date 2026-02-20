# Gen-SHM: Physics-Informed Generative Surrogate for Drone Wing Integrity

> **"Generating physically valid vibration data for any damage scenario—without crashing a single drone."**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![JAX](https://img.shields.io/badge/JAX-Accelerated-red?style=for-the-badge&logo=google)
![Equinox](https://img.shields.io/badge/Equinox-Neural%20Networks-green?style=for-the-badge)
![Diffrax](https://img.shields.io/badge/Diffrax-Solvers-orange?style=for-the-badge)
![Track](https://img.shields.io/badge/Track-Scientific%20ML%20%2B%20Generative%20AI-purple?style=for-the-badge)

**Submission Type:** Thinking + Modeling Challenge  
**Focus:** Structural Health Monitoring (SHM) via Generative Physics

---

## 🚁 Problem Statement

Developing robust damage detection algorithms for drone wings requires massive datasets of vibration responses under various damage conditions (e.g., micro‑cracks, delamination). Obtaining this data experimentally is impossible—**we cannot crash 10,000 drones to train one AI**. Traditional Finite Element Analysis (FEA) is too computationally expensive for real‑time edge deployment.

**The Challenge:** We need a **Generative Surrogate Model** that can instantly "hallucinate" physically accurate sensor data for any theoretical damage level, allowing us to train onboard health‑monitoring systems without destroying hardware.

---

## 🎯 Why This Problem Matters

- **Safety Critical:** Invisible fatigue cracks in composite wings are the leading cause of mid‑air failure in delivery drones.
- **Data Scarcity:** Real‑world failure data is sparse. Standard Generative AI (GANs/Diffusion) hallucinate realistic‑looking images of signals but fail to respect the conservation of energy and momentum required for valid engineering data.
- **Impact:** A physics‑grounded generative model unlocks **"Zero‑Shot" damage detection**—training drones to recognize failures they have never seen in real life.

---

## 📊 Dataset & Physics Source

- **Physics Foundation:** The **Euler‑Bernoulli Beam Theory** (Solid Mechanics), serving as the ground truth for structural vibration.
- **Input Data:** Sparse accelerometer readings from a healthy drone (for calibration).
- **Generated Output:** A synthetic dataset of 100,000+ acceleration time‑histories $\ddot{w}(t)$ corresponding to varying stiffness reductions ($E_{\text{damaged}}$).

---

## 🧠 Proposed Modeling Approach: The Physics‑Informed Generative Surrogate

We propose a **Parametric Physics‑Informed Neural Network (Parametric PINN)** acting as a generative engine. Unlike a standard solver that finds one solution, this network learns the solution operator across a continuous space of damage parameters.

### A. The Governing Equation (The "Famous" Equation)

We model the drone wing as a cantilever beam with spatially varying stiffness. The network must satisfy the **Dynamic Euler‑Bernoulli Equation**:

$$
\rho A(x) \frac{\partial^2 w}{\partial t^2} + \frac{\partial^2}{\partial x^2}\left( E(x,\xi) I(x) \frac{\partial^2 w}{\partial x^2} \right) + C \frac{\partial w}{\partial t} = F(x,t)
$$

Where:
- $w(x,t)$: Vertical displacement (vibration).
- $E(x,\xi)$: Stiffness field parameterized by damage variable $\xi$ (e.g., $\xi = 0.1$ implies 10% stiffness loss at a specific node).
- $\rho A$: Mass density per unit length.
- $C$: Viscous damping coefficient.

### B. The Generative Architecture

We treat the PINN as a generator $G_\theta(x,t,\xi)$.

- **Input:** Space‑time coordinates $(x,t)$ and a sampled damage parameter $\xi$ (drawn from a probability distribution).
- **Output:** The predicted vibration response $w_{\text{pred}}$.

### C. The SciML Loss Function

The model is trained not just to fit data, but to minimize the **Physics Residual**. This forces the generated data to obey Newton’s laws.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{PDE}}
$$

1. **Data Loss (Calibration):** Ensures the model matches the "healthy" drone baseline.
   $$
   \mathcal{L}_{\text{data}} = \frac{1}{N} \sum |w_{\text{pred}} - w_{\text{measured}}|^2
   $$

2. **Physics Residual (The "Truth"):** Penalizes the network if the generated vibration violates the Euler‑Bernoulli equation.
   $$
   \mathcal{L}_{\text{PDE}} = \frac{1}{M} \sum \left\| \rho A \frac{\partial^2 w_{\text{pred}}}{\partial t^2} + \frac{\partial^2}{\partial x^2}\left( E(x,\xi) I \frac{\partial^2 w_{\text{pred}}}{\partial x^2} \right) \right\|^2
   $$

---

## 🔬 Scientific Assumptions

- **Linear Elasticity:** The wing material behaves linearly up to the point of failure (Hooke’s Law applies).
- **Euler‑Bernoulli Validity:** The wing’s length‑to‑thickness ratio is high (>10:1), making shear deformation negligible (ignoring Timoshenko effects for computational efficiency).
- **Damage as Stiffness Reduction:** Structural damage is modeled purely as a local reduction in Young’s Modulus $E(x)$, not as a geometric change (mesh breakage).

---

## 🏁 Expected Outcome

A **Physics‑Verified Generative Tool** that outputs clean, labeled vibration datasets.

- **Input:** "Generate 100 samples of a wing with a 20% crack at the root."
- **Output:** 100 distinct, physically valid time‑series sensor readings.
- **Application:** These synthetic datasets will train a lightweight edge‑classifier (CNN) to detect damage with >95% accuracy, even for damage scenarios that have never occurred in flight testing.

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install dependencies (we recommend a virtual environment):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
