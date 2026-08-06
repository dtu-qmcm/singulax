# Singulax

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python->=3.12-blue)
![JAX](https://img.shields.io/badge/JAX-0.10-orange)
![Status](https://img.shields.io/badge/status-in--progress-orange)

Singulax solves the fed-batch bioreactor problem *"how should I feed substrate over time to
maximise product at harvest?"* Because the feed rate enters the dynamics linearly, the optimal
policy is a sequence of arcs — full feed, no feed, and **singular** arcs on which the classical
optimality condition degenerates. Those singular arcs are where the interesting behaviour lives
(the feed tracks the culture so that substrate is held at its productivity-optimal level), and
they are also what makes the problem numerically awkward.

The approach here is a **two-stage hybrid direct → indirect method**: a robust low-accuracy
direct solve discovers the arc structure and seeds the adjoints, then an indirect PMP shooting
solve refines everything to machine precision. Every derivative in the pipeline — costate
equations, Lie brackets, the singular feed law, sparse NLP Jacobians — is obtained by automatic
differentiation, so switching to a new kinetic model means writing three rate functions and
nothing else.

Built on [JAX](https://github.com/google/jax), [diffrax](https://github.com/patrick-kidger/diffrax),
[optimistix](https://github.com/patrick-kidger/optimistix) and IPOPT.

## Installation

Requires Python ≥ 3.12.

```sh
pip install singulax
```

## Usage

### As a library

Singulax exposes the building blocks of the optimal control problem: the state and costate
dynamics, the switching function $\phi$, and the singular feed law derived from $\ddot{\phi}=0$.

```python
import jax
jax.config.update("jax_enable_x64", True)   # required: the shooting residuals need float64

import jax.numpy as jnp
from singulax.substrate_product_inhibition import get_phi, get_Fsin, vector_field

S_F, V_max, F_min, F_max = 150.0, 10.0, 0.0, 2.0

x   = jnp.array([10.0, 5.0, 2.0, 3.0])      # [XV, SV, PV, V]
lam = jnp.array([1.0, 0.05, 1.0, -7.5])     # costate
y   = jnp.concatenate([x, lam])

print("phi    =", get_phi(y))                                # switching function
print("F_sing =", get_Fsin(y, (S_F, V_max, F_min, F_max)))   # singular feed
print("ydot   =", vector_field(0.0, y, (S_F, 1.0)))          # augmented state-costate RHS
```

Enabling `jax_enable_x64` is not optional, and it must happen **before** any array is created.
In float32 the shooting residuals cannot reach their convergence tolerance.

Each inhibition case is a self-contained subpackage:

| import | model |
|---|---|
| `singulax.substrate_inhibition` | Haldane kinetics $\mu(S)$ — has a closed-form singular feed |
| `singulax.substrate_product_inhibition` | Shin & Lim ethanol kinetics $\mu(S,P)$ |
| `singulax.collocation` | Stage-1 direct transcription, shared by both cases |

Case-specific names (`mu`, `sigma`, `get_phi`, `get_Fsin`, `vector_field`, …) exist in **both**
case subpackages and are deliberately not re-exported at the top level — import them from the
relevant subpackage so it is always clear which model you are using.

### Running the case studies

The `scripts/` tree holds one runnable script per control structure. Each is self-contained,
prints its Stage-1 and Stage-2 diagnostics, and writes PNG figures next to itself:

```sh
uv run python scripts/substrate_and_product_inhibition/hybrid_spi.py
uv run python scripts/substrate_inhibition/singular_arc/hybrid_approach/hybrid_si_sing.py
```

Scripts under `analytical_sol/` and `analytical_and_pmp/` finish in seconds; the
`hybrid_approach/` ones invoke IPOPT and take roughly one to five minutes.

---
## References

1. Foroozandeh, Z., Shamsi, M., & de Pinho, M. d. R. (2018). A Hybrid Direct–Indirect Approach for
   Solving the Singular Optimal Control Problems of Finite and Infinite Order.
   *Iranian Journal of Science and Technology, Transactions A: Science*, **42**(3), 1545–1554.
   [doi:10.1007/s40995-017-0176-2](https://doi.org/10.1007/s40995-017-0176-2)
   — the two-stage hybrid method and the cart-pole benchmark.
2. Shin, H. S., & Lim, H. C. *Fed-Batch Cultures*. Example 13.E.3 (ethanol from glucose by
   *S. cerevisiae*) — the substrate + product inhibition kinetics and the reference
   $PV(t_f) = 764.47$ g.
