# GPU-Accelerated Computing and Machine Learning for Economic Modeling

<div align="center">

![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900.svg)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)

**Master's Thesis Research**  
*Exploring GPU accelerated computing and machine learning methods for solving dynamic economic models*

[Overview](#-overview) • [Research Goals](#-research-goals) • [Current Progress](#-current-progress) • [Methods](#-methods) • [Getting Started](#-getting-started)

</div>

---

## 📋 Overview

This repository contains the implementation and analysis code for my master's thesis on **GPU-Accelerated Computing and Machine Learning for Economic Modeling**. The research investigates the computational efficiency gains from GPU acceleration and reinforcement learning techniques applied to solving complex dynamic economic models, with applications to macroeconomic policy analysis.

### 🎯 Research Objectives

1. **Benchmark traditional vs. accelerated methods** for solving dynamic stochastic economic models
2. **Implement GPU-accelerated algorithms** to handle high-dimensional state spaces efficiently
3. **Apply reinforcement learning (Q-learning)** as an alternative to traditional value function iteration
4. **Extend methodology to pension systems modeling** with demographic and policy complexity
5. **Explore neural network architectures** for approximating policy and value functions in economic contexts

### 🔬 Research Questions

- How much speedup can GPU acceleration provide for value function iteration in economic models?
- Can reinforcement learning algorithms converge to accurate policy functions in economic applications?
- What are the precision trade-offs between `float` and `double` in economic computations?
- How can these methods scale to overlapping generations (OLG) pension models with realistic complexity?

---

## 📊 Current Progress

### ✅ Completed: Neoclassical Growth Model

The **neoclassical growth model** serves as a benchmark case for method comparison. I have implemented and analyzed three solution approaches:

#### 1. **Value Function Iteration (VFI)** - Traditional CPU Implementation
- Standard dynamic programming approach
- Bellman equation iteration until convergence
- Baseline for performance comparison

#### 2. **GPU-Accelerated Value Function Iteration**
- CUDA-based parallel implementation
- Significant speedup for large state spaces (grid sizes 1,000–10,000+)
- Precision analysis comparing `float` vs. `double` arithmetic

#### 3. **Q-Learning (Reinforcement Learning)**
- Model-free approach using exploration-exploitation strategy
- Configurable hyperparameters:
  - Grid size (`n_k`): 145–10,000 points
  - Exploration probability with exponential decay
  - Learning rate and convergence tolerance
- Converges to near-optimal policies without explicit model knowledge

**Model Specification:**
```
Household maximization:
  max E[Σ β^t log(c_t)]
  subject to: k_{t+1} = z·k_t^α + (1-δ)k_t - c_t

Parameters:
  α = 0.5   (capital share)
  β = 0.96  (discount factor)
  δ = 0.025 (depreciation rate)
  z = 1.0   (productivity)
```

### 🚧 In Progress: Pension System Model

Currently developing an **overlapping generations (OLG) model** with pension systems, incorporating:

- **Life-cycle optimization** for multiple cohorts
- **Pension contributions and benefits** with policy rules
- **Government budget constraints** linking contributions to payouts
- **Aggregate capital and labor markets** with general equilibrium

**Planned Extensions:**
- Multi-period pension reform analysis
- Demographic shocks (aging population scenarios)
- Neural network policy approximators for high-dimensional state spaces

---

## 🛠️ Methods & Implementation

### Project Structure

```
masters-thesis/
├── neoclassical-model/
│   ├── exe-time-measurements/     # Performance benchmarking
│   │   ├── rl.cpp                 # Q-learning with timing
│   │   ├── rl2.cpp                # Alternative RL implementation
│   │   └── vfi.cpp                # VFI benchmarks (CPU & GPU)
│   └── plots/                     # Visualization and analysis
│       ├── rl-plots.cpp           # RL results generator
│       ├── plots.py               # Unified plotting script
│       └── plot.py                # VFI visualization
├── pension-model/
│   └── model.cpp                  # OLG pension model (WIP)
└── README.md
```

### Computational Techniques

#### Value Function Iteration (VFI)
```cpp
// Bellman operator
V_{n+1}(k) = max_{k'} { log(f(k) - k') + β·V_n(k') }

// GPU parallelization across state space grid
// Each thread computes optimal k' for different k
```

#### Q-Learning Algorithm
```cpp
// State-action value update
Q(k, k') ← (1-α)·Q(k, k') + α·[r(k,k') + β·max_{k''} Q(k', k'')]

// ε-greedy exploration
action = (rand() < ε) ? random_action : argmax Q(k, ·)
```

#### Performance Metrics
- **Convergence speed**: Iterations to ε-tolerance
- **Wall-clock time**: Real execution time
- **CPU cycles**: Processor cycle counts (Windows `QueryProcessCycleTime`)
- **Accuracy**: Distance from analytical/VFI steady state
- **Precision**: `float` vs. `double` comparison

---

## 📈 Results & Visualization

The Python plotting scripts generate comprehensive visualizations:

- **Value functions** across state space
- **Policy functions** (optimal savings/capital decisions)
- **Convergence paths** (iteration vs. error)
- **Consumption and savings rates**
- **Execution time comparisons** (CPU, GPU, RL)
- **Steady-state analysis** (fixed points, crossing points)

Example outputs saved to `out/plots/`:
```
rl_145_value_function.png
rl_145_policy_function.png
rl_145_convergence.png
cpu_145_execution_time.png
```

---

## 🚀 Getting Started

### Prerequisites

**For C++ (VFI/RL):**
- **C++17 compiler** (MSVC, GCC, or Clang)
- **CUDA Toolkit** (for GPU implementations)
- **Windows** (for CPU cycle timing; Linux support can use `clock_gettime`)

**For Python (plotting):**
```bash
pip install pandas numpy matplotlib seaborn
```

### Compilation Instructions

Each C++ file contains compilation commands in header comments:

**Reinforcement Learning (Release, Double Precision):**
```bash
cl /std:c++17 /O2 /EHsc /DUSE_DOUBLE=1 rl-plots.cpp /Fe:rl-plots-double.exe
```

**Reinforcement Learning (Release, Float Precision):**
```bash
cl /std:c++17 /O2 /EHsc rl-plots.cpp /Fe:rl-plots-float.exe
```

**Debug Builds:**
```bash
cl /std:c++17 /Zi /Od /EHsc /DUSE_DOUBLE=1 rl-plots.cpp /Fe:rl-plots-double-d.exe
```

### Running Experiments

**Example: Q-learning with 1,000 grid points**
```bash
./rl-plots-double.exe 1000 50000 0.99 0.999 0.95 38 20 0.001 5
```

**Parameters:**
1. `n_k` = 1000 (grid size)
2. `steps_per_iter` = 50000
3. `exploration_prob` = 0.99 (initial ε)
4. `explr_multiplier` = 0.999 (ε decay)
5. `learning_rate` = 0.95 (α)
6. `iters` = 38 (max iterations)
7. `explr_limits` = 20
8. `epsilon` = 0.001 (convergence tolerance)
9. `save_every` = 5 (snapshot frequency)

**Example: Large-scale experiment (10,000 grid)**
```bash
./rl-plots-double.exe 10000 125000 0.99 0.999 0.99 458 200 0.0006 5
```

### Generating Visualizations

```bash
cd neoclassical-model/plots
python plots.py
```

This will:
- Detect all result CSV files in `out/data/`
- Generate plots for each grid size and method
- Create summary statistics in `out/plots/`

---

## 📚 Theoretical Background

### Dynamic Programming Framework

Economic agents solve:
```
V(s) = max_{a∈A(s)} { u(s,a) + β·E[V(s')|s,a] }
```

Where:
- `s` = state (capital stock)
- `a` = action (consumption/saving)
- `u(·)` = utility function
- `β` = discount factor
- `V(·)` = value function

### Reinforcement Learning Connection

Q-learning reformulates this as:
```
Q(s,a) = r(s,a) + β·E[max_{a'} Q(s',a')]
```

---

## 🔬 Upcoming Work

### Phase 2: Pension Model Extensions

1. **Model Complexity**
   - Multi-generation overlapping structure
   - Pension contribution rates (τ)
   - Retirement benefits linked to wage history
   - Government budget balance

2. **Solution Methods**
   - Extend VFI to 2D/3D state spaces
   - GPU acceleration critical

3. **Policy Experiments**
 

### Phase 3: Neural Network Approximation


---




<div align="center">

**⭐ If you find this research interesting, please consider starring the repository!**

*Last Updated: February 2026*

</div>
