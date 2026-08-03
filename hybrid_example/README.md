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
