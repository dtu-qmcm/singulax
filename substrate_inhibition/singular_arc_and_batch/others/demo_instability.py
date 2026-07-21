"""
Demonstration of why forward integration of the batch-phase costate ODE explodes.

The simplified model is:
    dλ₂/dt = k·λ₂ − c,    λ₂(T) = 0

This is a stand-alone script; it only needs numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

k = 641.0    # eigenvalue of the batch-phase costate ODE
c = 41.25    # constant term
T = 0.6      # length of the batch phase (hours)

y_eq  = c / k                     # equilibrium: dλ₂/dt = 0  →  λ₂* ≈ 0.0643
y0_exact = y_eq * (1 - np.exp(-k * T))   # true λ₂ at t=0 (satisfies λ₂(T)=0)

# ── 1. The true initial condition is indistinguishable from the equilibrium ──
print("=" * 60)
print("STEP 1: true λ₂(0) vs equilibrium λ₂*")
print("=" * 60)
print(f"  λ₂*          = {y_eq:.16f}")
print(f"  true λ₂(0)   = {y0_exact:.16f}")
print(f"  difference   = {y_eq - y0_exact:.3e}")
print(f"  float64 eps  = {np.finfo(float).eps:.3e}")
print()
print(f"  Are they equal in float64?  {y0_exact == y_eq}")
print()
print("  → The true initial condition is 10^-167 below the equilibrium.")
print("    Float64 has only ~16 significant digits, so the computer")
print("    CANNOT represent the difference.  Both look identical.")

# ── 2. Forward Euler: first step introduces machine-epsilon error ────────────
print()
print("=" * 60)
print("STEP 2: first-step error and its amplification")
print("=" * 60)

def exact_solution(t):
    """Analytical solution satisfying λ₂(T) = 0."""
    return y_eq * (1 - np.exp(-k * (T - t)))

# Simulate with simple forward Euler so the error mechanism is transparent
dt = 0.001
t = 0.0
y = y0_exact   # start from the "exact" value (which is y_eq in float64)

first_step_error = None
t_arr, y_num, y_true, err = [], [], [], []

while t <= T + 1e-10:
    y_ex = exact_solution(t)
    e = abs(y - y_ex)
    t_arr.append(t)
    y_num.append(y)
    y_true.append(y_ex)
    err.append(e)
    if t > 0 and first_step_error is None:
        first_step_error = e
    y = y + dt * (k * y - c)    # Euler step
    t += dt

t_arr  = np.array(t_arr)
y_num  = np.array(y_num)
y_true = np.array(y_true)
err    = np.array(err)

print(f"  First-step error ε₀ ≈ {first_step_error:.3e}")
print(f"  Predicted final error = ε₀ × e^(k×T)")
print(f"                       = {first_step_error:.2e} × {np.exp(k*T):.2e}")
print(f"                       = {first_step_error * np.exp(k*T):.2e}")
print(f"  Actual   final error = {err[-1]:.2e}")

# ── 3. Effect of tightening tolerance (reducing ε₀) ─────────────────────────
print()
print("=" * 60)
print("STEP 3: does tightening tolerance help?")
print("=" * 60)
print(f"{'ε₀ (initial error)':>22} | {'error at t=T':>15} | {'useful?':>8}")
print("-" * 54)
for eps in [1e-4, 1e-8, 1e-12, 1e-16, 1e-32, 1e-100, 1e-167]:
    final = eps * np.exp(k * T)
    useful = "yes" if final < 1e-3 else "no"
    print(f"{eps:>22.0e} | {final:>15.2e} | {useful:>8}")
print()
print("  → You would need ε₀ < 10^-167 to suppress the explosion.")
print(f"    Float64 machine epsilon is {np.finfo(float).eps:.0e}.")
print("    This is 151 orders of magnitude short.")

# ── 4. Backward integration: starts from y(T)=0 exactly, errors are DAMPED ─
print()
print("=" * 60)
print("STEP 4: backward integration — why it works")
print("=" * 60)

# Backward Euler (same step size)
dt_b = 0.001
t_b = T
y_b = 0.0    # terminal condition: EXACT

t_b_arr, y_b_arr, y_b_true, err_b = [], [], [], []
while t_b >= -1e-10:
    y_ex = exact_solution(t_b)
    t_b_arr.append(t_b)
    y_b_arr.append(y_b)
    y_b_true.append(y_ex)
    err_b.append(abs(y_b - y_ex))
    y_b = y_b - dt_b * (k * y_b - c)    # Euler step backward
    t_b -= dt_b

t_b_arr = np.array(t_b_arr[::-1])
y_b_arr = np.array(y_b_arr[::-1])
err_b   = np.array(err_b[::-1])

print(f"  Backward integration gives λ₂(0) = {y_b_arr[0]:.8f}")
print(f"  True                    λ₂(0) = {y0_exact:.8f}")
print(f"  Max error over full trajectory = {max(err_b):.2e}")
print()
print("  → Backward integration stays accurate because perturbations")
print("    shrink as e^{-k·Δt} instead of growing as e^{+k·Δt}.")

# ── 5. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle("Why forward integration explodes\n"
             r"($d\lambda_2/dt = 641\,\lambda_2 - 41.25$, batch phase model)",
             fontsize=14)

# Panel 1: trajectories
ax = axes[0]
ax.plot(t_arr, y_true, 'k-',  lw=2,   label='True solution')
ax.plot(t_arr, y_num,  'r--', lw=1.5, label='Forward (numerical)')
ax.plot(t_b_arr, y_b_arr, 'b:', lw=1.5, label='Backward (reversed)')
ax.axhline(y_eq, color='gray', linestyle='-.', label=f'Equilibrium λ₂* = {y_eq:.3f}')
ax.set_ylabel(r'$\lambda_2$')
ax.set_title('Trajectories')
ax.legend()
ax.set_xlim(0, T)
ax.set_ylim(-0.05, 0.15)
ax.grid(True)

# Panel 2: forward error (log scale)
ax = axes[1]
safe = err > 0
ax.semilogy(t_arr[safe], err[safe], 'r-', lw=1.5, label='Forward error')
ax.semilogy(t_arr[safe], first_step_error * np.exp(k * t_arr[safe]),
            'k--', lw=1, label=r'$\varepsilon_0 \cdot e^{641\,t}$ (predicted)')
ax.set_ylabel('|error|')
ax.set_title('Forward integration: error grows as $e^{641t}$')
ax.legend()
ax.set_xlim(0, T)
ax.grid(True)

# Panel 3: backward error (log scale)
ax = axes[2]
safe_b = err_b > 0
ax.semilogy(t_b_arr[safe_b], err_b[safe_b], 'b-', lw=1.5, label='Backward error')
ax.set_ylabel('|error|')
ax.set_xlabel('Time (h)')
ax.set_title('Backward integration: error stays small')
ax.legend()
ax.set_xlim(0, T)
ax.grid(True)

plt.tight_layout()
plt.savefig("demo_instability.png", dpi=120, bbox_inches='tight')
plt.show()
print("\nPlot saved to demo_instability.png")
