# 🔬 Optimization Algorithms on the Rastrigin Function

Implementing and comparing 6 classical and nature-inspired optimization algorithms from scratch in Python — benchmarked on the Rastrigin function with a linear equality constraint.

---

## 📌 Table of Contents
- [Problem Definition](#problem-definition)
- [Constraint](#constraint)
- [Algorithms Implemented](#algorithms-implemented)
  - [Random Search](#1-random-search)
  - [Gradient Descent](#2-gradient-descent)
  - [Simulated Annealing](#3-simulated-annealing)
  - [Genetic Algorithm](#4-genetic-algorithm)
  - [Bacterial Foraging (Chemotaxis)](#5-bacterial-foraging-chemotaxis)
  - [Particle Swarm Optimization](#6-particle-swarm-optimization)
- [Constraint Handling](#constraint-handling)
- [Results & Comparison](#results--comparison)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## 📐 Problem Definition

The **Rastrigin function** is a standard non-convex benchmark in optimization. It is highly multimodal — meaning it has a large number of local minima — making it a difficult test for any optimizer.

$$f(x, y) = 20 + x^2 + y^2 - 10\left(\cos(2\pi x) + \cos(2\pi y)\right)$$

**Search space:** $x, y \in [-5.12,\ 5.12]$

**Global minimum:** $f(0, 0) = 0$

```python
def function(x, y):
    return 20 + x**2 + y**2 - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))
```

The surface looks like this — hundreds of valleys, only one of which is the true global minimum:

```
High values  →  yellow/green peaks
Low values   →  dark purple valleys
Global min   →  deepest purple point at (0, 0)
```

<img width="1536" height="754" alt="RF_3d" src="https://github.com/user-attachments/assets/b44fffda-eabf-4150-8eda-cfe68087d406" />


---

## 🔗 Constraint

All algorithms are tested under the **equality constraint**:

$$x - y = 0 \quad \Longleftrightarrow \quad x = y$$

This is enforced using the **penalty method** — adding a penalty term to the objective whenever the constraint is violated:

```python
penalty_weight = 100

def penalized(x, y):
    return function(x, y) + penalty_weight * (x - y)**2
```

The penalty term $(x - y)^2 = 0$ only when $x = y$, which steers all algorithms toward the constraint line. The constrained global minimum remains at **(0, 0)** since it satisfies both $x = y$ and $f(0,0) = 0$.

> ⚠️ Tuning `penalty_weight` is critical:
> - Too low → constraint is ignored
> - Too high → objective is ignored, algorithm only satisfies constraint

<img width="640" height="480" alt="RF_ct" src="https://github.com/user-attachments/assets/cf3511e2-37c3-4441-9411-510f265ada45" />

---

## ⚙️ Algorithms Implemented

### 1. Random Search

The simplest possible optimizer. Randomly samples points in the search space and keeps track of the best one found. No memory, no direction, no learning.

**How it works:**
```
repeat N times:
    sample random (x, y) from bounds
    if f(x, y) < best → update best
```

**Role here:** Serves as a performance baseline. Shows how much structure and memory actually matter in the other methods.

**Weakness:** Does not scale — performance degrades quickly in higher dimensions.

<img width="640" height="480" alt="RS_ct" src="https://github.com/user-attachments/assets/1035270c-7cec-4ce0-99f2-1b050edf6ef3" />

---

### 2. Gradient Descent

Follows the negative gradient of the function to move downhill at each step.

**Update rule:**
$$x_{t+1} = x_t - \alpha \cdot \nabla f(x_t)$$

Since the Rastrigin function is not easily differentiable analytically, numerical gradients are used:

```python
def numerical_gradient(x, y, h=1e-5):
    df_dx = (function(x+h, y) - function(x-h, y)) / (2*h)
    df_dy = (function(x, y+h) - function(x, y-h)) / (2*h)
    return df_dx, df_dy
```

---

#### 📉 Adaptive Learning Rate

A fixed learning rate $\alpha$ causes two problems:
- **Too large** → overshoots the minimum, diverges
- **Too small** → converges extremely slowly

Adaptive learning rate methods adjust $\alpha$ automatically at each step based on the history of gradients.

**Decay Schedule** — shrink $\alpha$ over time regardless of gradient:
$$\alpha_t = \frac{\alpha_0}{1 + \text{decay} \times t}$$

```python
def decayed_lr(alpha_0, decay, t):
    return alpha_0 / (1 + decay * t)
```

> Simple and predictable. Good when you know roughly how many iterations you need.

<img width="640" height="480" alt="GD_ct" src="https://github.com/user-attachments/assets/33f8c3ad-da5b-4c79-b00c-d8664965e434" />

---

### 3. Simulated Annealing

Inspired by the metallurgical process of heating and slowly cooling metal to reduce defects. Allows **uphill moves** with a probability controlled by temperature.

**Acceptance rule (Boltzmann criterion):**
$$P(\text{accept}) = e^{-\Delta f / T}$$

- High $T$ → accepts bad moves freely → wide exploration  
- Low $T$ → rarely accepts bad moves → fine-tuning

**Cooling schedule:**
$$T_{t+1} = T_t \times \rho$$

**Key parameters:**

| Parameter | Description | Recommended |
|---|---|---|
| `T_start` | Initial temperature | `1000–5000` |
| `cooling_rate` | Rate of temperature decay | `0.95–0.999` |
| `step_size` | Neighbor perturbation size | `0.1–1.0` |

**Enhancement used:** Adaptive step size — step shrinks as temperature drops:
```python
step_size = max(2.0 * (T / T_start), 0.01)
```
<img width="640" height="480" alt="SA_ct" src="https://github.com/user-attachments/assets/b3b26191-3aee-41f3-a4ea-d41195a51087" />

---

### 4. Genetic Algorithm

Inspired by Darwinian evolution. Maintains a **population** of candidate solutions and evolves them over generations through selection, crossover, and mutation.

**Pipeline per generation:**
```
Evaluate fitness → Tournament Selection → BLX-α Crossover → Gaussian Mutation → Elitism
```

**Key operators:**

- **Tournament Selection** — randomly pick k individuals, keep the best
- **BLX-α Crossover** — blend parent genes in an extended interval:
$$\text{child} \in [\min(g_1, g_2) - \alpha d,\ \max(g_1, g_2) + \alpha d]$$
- **Gaussian Mutation** — add noise: $g \mathrel{+}= \mathcal{N}(0, \sigma)$
- **Elitism** — always carry the top N individuals to the next generation

**Key parameters:**

| Parameter | Description | Recommended |
|---|---|---|
| `pop_size` | Number of individuals | `50–200` |
| `crossover_rate` | Probability of crossover | `0.6–0.9` |
| `mutation_rate` | Probability of mutation per gene | `0.05–0.2` |
| `elite_count` | Number of elites preserved | `1–5` |

<img width="640" height="480" alt="GA_ct" src="https://github.com/user-attachments/assets/f453e24b-d963-431f-bbb3-0e6c5addbb97" />

---

### 5. Bacterial Foraging (Chemotaxis)

Inspired by how *E. coli* bacteria navigate chemical gradients in search of nutrients. Part of the **Bacterial Foraging Optimization (BFO)** algorithm.

**Each bacterium performs:**

```
TUMBLE → pick a random unit direction vector
SWIM   → keep moving in that direction while fitness improves
         stop after n_swim steps or if fitness worsens
```

**Full BFO lifecycle:**
```
for each Reproduction cycle:
    for each Elimination cycle:
        for each Chemotaxis step:
            all bacteria: tumble → swim
    Elimination: randomly teleport some bacteria (prevents stagnation)
Reproduction: best half survive and clone themselves
```

**Key parameters:**

| Parameter | Description | Recommended |
|---|---|---|
| `n_bacteria` | Swarm size | `20–50` |
| `step_size` | Swim step length | `0.05–0.5` |
| `n_swim` | Max swim steps per tumble | `5–15` |
| `elim_prob` | Elimination probability | `0.1–0.3` |

<img width="640" height="480" alt="CT_ct" src="https://github.com/user-attachments/assets/9d99de9b-0b83-489c-8d25-4a330b120891" />

---

### 6. Particle Swarm Optimization

Inspired by the collective behaviour of bird flocks or fish schools. Each particle has a **position**, **velocity**, **personal best**, and access to the **global best**.

**Velocity update rule:**
$$v_{t+1} = \underbrace{w \cdot v_t}_{\text{inertia}} + \underbrace{c_1 r_1 (p_{\text{best}} - x_t)}_{\text{cognitive}} + \underbrace{c_2 r_2 (g_{\text{best}} - x_t)}_{\text{social}}$$

**Position update:**
$$x_{t+1} = x_t + v_{t+1}$$

- $w$ — inertia weight: how much old velocity is preserved  
- $c_1$ — cognitive weight: trust in personal experience  
- $c_2$ — social weight: trust in swarm knowledge  
- $r_1, r_2$ — random scalars in $[0, 1]$

**Key parameters:**

| Parameter | Description | Recommended |
|---|---|---|
| `n_particles` | Swarm size | `20–50` |
| `w` | Inertia weight | `0.4–0.9` |
| `c1` | Cognitive weight | `1.5–2.5` |
| `c2` | Social weight | `1.5–2.5` |
| `v_max` | Max velocity clamp | `0.2–1.0` |

<img width="640" height="480" alt="PS_ct" src="https://github.com/user-attachments/assets/8a00e059-b2ee-4cae-a603-6ae0b5453749" />

---

## 📊 Results & Comparison

| Algorithm | Converges to Global Min? | Speed | Diversity | Constraint Handling |
|---|---|---|---|---|
| Random Search | Rarely | Slow | High | Weak |
| Gradient Descent | Sometimes | Fast | None | Moderate |
| Simulated Annealing | Often | Moderate | Low | Good |
| Genetic Algorithm | Most reliable | Fast | High | Good |
| Bacterial Foraging | Often | Moderate | Medium | Good |
| PSO | Most reliably | **Fastest** | Medium | Good |

**Key observations:**
- **PSO** converged fastest — particles share global information instantly via `gbest`
- **GA** maintained the best population diversity — least likely to get permanently stuck
- **SA** was most sensitive to hyperparameters — cooling rate and step size need careful tuning
- **Chemotaxis** was most biologically interpretable — tumble/swim maps directly to real bacteria behaviour
- **All methods** required careful `penalty_weight` tuning to properly respect the `x = y` constraint
- **Gradient Descent** consistently failed on Rastrigin — trapped in the first local minimum it found

---

## 🗂️ Project Structure

```
.
├── Function.py              # Rastrigin function definition
├── RandomSearch.py         # Random Search implementation
├── GradientDescent.py      # Gradient Descent implementation
├── SimulatedAnnealing.py   # Simulated Annealing implementation
├── GeneticAlgorithm.py     # Genetic Algorithm implementation
├── Chemotaxis.py            # Bacterial Foraging / Chemotaxis implementation
├── ParticleSwarm.py                   # Particle Swarm Optimization implementation
└── README.md
```

---

## 📦 Requirements

```
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```

---

## ▶️ How to Run

Each algorithm is self-contained. Run any file directly:

```bash
python RandomSearch.py
python GradientDescent.py
python SimulatedAnnealing.py
python GeneticAlgorithm.py
python Chemotaxis.py
python ParticleSwarm.py
```

Each script will:
1. Run the optimizer on the Rastrigin function with constraint `x = y`
2. Print progress and the best solution found
3. Display a 3D surface plot, contour plot, convergence curve, and constraint violation history


## 📚 References

- Rastrigin, L. A. (1974). *Systems of Extremal Control*
- Kennedy, J. & Eberhart, R. (1995). *Particle Swarm Optimization*
- Passino, K. M. (2002). *Biomimicry of Bacterial Foraging for Distributed Optimization*
- Kirkpatrick, S. et al. (1983). *Optimization by Simulated Annealing*
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*
