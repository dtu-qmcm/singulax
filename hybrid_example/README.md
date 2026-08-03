## Cart-Pole Optimal Control Problem

Picture a cart that slides on a horizontal track with a pendulum hinged on top of it. You are allowed to push the cart to the left or right with a force $u$, which serves as the only actuator. 
Thus, the pole only moves because the cart accelerates underneath its pivot. 

The goal is to bring the cart-pole system from an initial state to a terminal state with minimum control efforts. The initial state is the stable stationary point (the cart at a stop with the pole vertically down), and the terminal state is the unstable stationary state (the cart at a stop but with the pole vertically up).

This example simulates **Illustrative Example 1 (A Finite Order Singular Problem)** from [Foroozandeh et al., 2018](https://doi.org/10.1007/s40995-017-0176-2).

### System Dynamics

The system is defined by four state variables $x = [x_1, x_2, x_3, x_4]^T$:
* $x_1$: Pendulum angle
* $x_2$: Pendulum angular velocity
* $x_3$: Cart position along the track
* $x_4$: Cart velocity

The equations of motion $\dot{x} = f(x, u)$ are given by:

$$
\begin{aligned}
\frac{dx_1}{dt} &= x_2 \\
\frac{dx_2}{dt} &= F_2 \\
\frac{dx_3}{dt} &= x_4 \\
\frac{dx_4}{dt} &= F_4
\end{aligned}
$$

where:

$$
F_2 = -\frac{\epsilon x_2^2 c s + s + cu}{D}, \quad F_4 = \frac{\epsilon s (x_2^2 + c) + u}{D}
$$

and the auxiliary expressions and parameters are:

$$
c = \cos(x_1), \quad s = \sin(x_1), \quad D = 1 - \epsilon c^2, \quad \epsilon = 0.5, \quad t_f = 4.912
$$

### Initial Conditions & Constraints

* **Initial States:** $x(0) = [0, 0, 0, 0]^T$
* **Control Bounds:** $-1 \le u \le +1$

### Objective Function

The objective is to minimize the terminal state error:

$$
\min_{u} J(u) = \frac{1}{2} (x(t_f) - x_f)^T (x(t_f) - x_f)
$$

---

## Solution Methodology

To solve this problem, a two-stage hybrid method is used:

1. **Stage 1:** Direct Euler Method
2. **Stage 2:** Adaptive Indirect Shooting Method

### Why a Two-Stage Approach?
The **Euler method** is robust and provides a quick approximate solution, which is sufficient for detecting the structure of the optimal control profile and generating an initial guess for the adjoint (co-state) variables. 
However, direct Euler discretization yields low accuracy. 

To resolve this, **Stage 2** initializes the **adaptive shooting method** using the output from Stage 1 to yield a highly accurate solution.

---

## Running the Code

To execute the example, run:

```bash
python hybrid_example.py
```

It prints the Stage-1 and Stage-2 results to the console and exports two figures:
* `hybrid_example_stage1.png`
* `hybrid_example_stage2.png`

---

## Code Walkthrough

The script `hybrid_example.py` is structured into four main blocks following the hybrid direct–indirect methodology:

### 1. Problem Definition
* `F_calc(x, u)` / `states_ode(x, u)`: Defines the cart-pole equations of motion $\dot{x} = f(x, u)$.
* **Problem Setup:** Parameters ($N, t_f, x_0, x_f$) and decision-vector layout $z = [x_0, \dots, x_N, u_0, \dots, u_{N-1}]$ are configured with control bounds $|u| \le 1$.

### 2. Stage 1 — Direct Method (Forward-Euler Transcription + IPOPT)
* `objective(z)`: Evaluates the terminal cost $\frac{1}{2} \|x(t_f) - x_f\|^2$.
* `constraints(z)`: Computes the forward-Euler defects $x_{k+1} - x_k - h f(x_k, u_k) = 0$ to enforce system dynamics.
* `jac_defects(z)`: Computes the sparse (COO format) Jacobian of the defect constraints. Since each defect only depends on $x_k, x_{k+1}, u_k$, only those non-zero block structures are evaluated.
* `minimize_ipopt(...)`: Solves the Nonlinear Program (NLP) and extracts:
  * **Costate Seed (`lam = -mult_g`):** The constraint multipliers serve as discrete costates (with a negative sign due to minimization).
  * **Control Structure Detection:** Evaluated via `detect_switching_times(U_opt, tf)`.

> **Control Structure Analysis (`detect_switching_times`):**
> Labels each control sample as $+1$, $-1$, or interior, applies median filtering to remove 1–2 point numerical flicker, and classifies each segment:
> * A **long interior segment** represents a singular arc (two switches: entry and exit).
> * A **short interior segment** represents a standard bang–bang transition (one switch, located with sub-grid accuracy via zero-crossing interpolation).

### 3. Stage 2 — Indirect Method (PMP Multiple Shooting)
With the control structure established as `arc_u = [+1, singular, -1, +1, -1]`, the optimal control problem is recast as a Boundary Value Problem (BVP):
* `switching_fn(y)`: Calculates the switching function $\sigma$.
* `hamiltonian` & `costate_ode`: Evaluates the costate dynamics $\dot{\lambda} = -\frac{\partial H}{\partial x}$.
* `u_sing(x, l, t)`: Evaluates the singular control law derived from $\ddot{\sigma} = 0$.
* `vector_field` / `vector_field_singular`: Integrates the augmented state–costate system $y = [x, \lambda]^T$ on bang arcs (fixed $u = \pm 1$) or singular arcs ($u = u_{\text{sing}}$).
* `solve_and_get_residuals__ms(guess, args)`: Constructs the residual vector for the decision vector:
  $$\zeta = [y_0, \dots, y_5 \mid t_1, \dots, t_4]$$
  *(6 nodes $\times$ 8 states/costates + 4 switching times = 52 unknowns)*. The residual evaluates:
  1. 5 continuity defects across internal nodes.
  2. Initial boundary condition ($x(0) = 0$).
  3. Terminal boundary condition / transversality condition ($\lambda(t_f) = x(t_f) - x_f$).
  4. Switching conditions ($\sigma = 0$ at each interior switch time).
* `optx.LevenbergMarquardt`: Solves the 52-dimensional system of nonlinear equations, jointly refining node states and switching times.

### 4. Plotting
Reconstructs dense state–costate trajectories from the converged solution and renders the Stage-1 and Stage-2 diagnostic plots.
