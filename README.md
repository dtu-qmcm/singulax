# Singulax

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python->=3.12-blue)
![JAX](https://img.shields.io/badge/JAX-0.10-orange)
![Status](https://img.shields.io/badge/status-in--progress-orange)

**Optimal feeding policies for fed-batch fermentation, computed from Pontryagin's Maximum
Principle — including the singular arcs where the maximum principle on its own determines
nothing.**

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
