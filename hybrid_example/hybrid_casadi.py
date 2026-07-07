import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- Problem Setup ---
N = 1000
t_f = 4.912
h = t_f / N
xf_target = [np.pi, 0.0, 0.0, 0.0]

# --- Create CasADi Opti Stack ---
opti = ca.Opti()

# Variables: States (N+1 points) and Controls (N points)
X = opti.variable(4, N + 1)
U = opti.variable(1, N)

# --- Dynamics & Constraints ---
# Initial conditions
opti.subject_to(X[:, 0] == [0, 0, 0, 0])

# Forward Euler Dynamics 
for k in range(N):
    c = ca.cos(X[0, k])
    s = ca.sin(X[0, k])
    e = 0.5
    D = 1 - e * c**2
    
    F2 = -((e * X[1, k]**2 * c + s + c * U[0, k]) / D)
    F4 = -((e * s * (X[1, k]**2 + c) + U[0, k]) / D)
    
    dx = ca.vertcat(X[1, k], F2, X[3, k], -F4)
    
    # Collocation constraint: x_{k+1} = x_k + h * dx
    opti.subject_to(X[:, k+1] == X[:, k] + h * dx)

# Control Bounds
opti.subject_to(opti.bounded(-1.0, U, 1.0))

# --- Objective ---
# Terminal state error
diff = X[:, -1] - xf_target
opti.minimize(0.5 * ca.dot(diff, diff))

# --- Initial Guess ---
# Straight line interpolation for states, zeros for control
for i in range(4):
    opti.set_initial(X[i, :], np.linspace(0, xf_target[i], N + 1))
opti.set_initial(U, 0.0)

# --- Solve ---
# CasADi handles the sparsity automatically here!
opti.solver('ipopt', {'print_time': False}, {'max_iter': 3000, 'tol': 1e-6})

print(f"Starting CasADi IPOPT optimization with N={N}...")
try:
    sol = opti.solve()
    X_opt = sol.value(X)
    U_opt = sol.value(U)
    print("Optimization Status: SUCCESS")
except Exception as e:
    print("Optimization failed. Pulling best current values.")
    X_opt = opti.debug.value(X)
    U_opt = opti.debug.value(U)

# --- Plotting ---
U_plot = np.array(U_opt).flatten() 
t_controls = np.linspace(0, t_f, N)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t_states = np.linspace(0, t_f, N + 1)

ax1.plot(t_states, X_opt[0, :], label='x1')
ax1.plot(t_states, X_opt[1, :], label='x2')
ax1.plot(t_states, X_opt[2, :], label='x3')
ax1.plot(t_states, X_opt[3, :], label='x4')
ax1.set_xlabel('t')
ax1.set_title(f'Stage 1: States (CasADi, N={N})')
ax1.legend()

ax2.plot(t_controls, U_plot)
ax2.axhline( 1, color='k', linestyle='--', linewidth=0.8)
ax2.axhline(-1, color='k', linestyle='--', linewidth=0.8)
ax2.set_xlabel('t')
ax2.set_ylabel('u')
ax2.set_title('Stage 1: Optimal Control')

plt.tight_layout()
plt.show()
