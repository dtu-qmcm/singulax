"""
Forward single-shooting comparison across ODE solvers.

The backward integration (already known to be stable) gives us exact initial
conditions.  We then try every available solver for the forward direction and
report how well each reproduces the terminal costate λ₂(t_end) = 0.

If a solver succeeds, λ₂(t_end) should be ≈ 0.
If it fails (the underlying IVP is unstable), the value explodes regardless.
"""

import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx
import matplotlib.pyplot as plt
import os
from diffrax import ODETerm, SaveAt, diffeqsolve, PIDController
from scipy.integrate import solve_ivp

from singulax.substrate_inhibition.odes import vector_field
from singulax.substrate_inhibition.analytical import states_analytical_sol, volume_cond
from singulax.substrate_inhibition.shooting import loss_fn_batch
from singulax.substrate_inhibition.Fsin import get_Fsin_unconstrained

jax.config.update("jax_enable_x64", True)

# ── Parameters ────────────────────────────────────────────────────────────────
Ks, Ki = 0.1, 100.0
S0     = jnp.sqrt(Ks * Ki)
y0     = jnp.array([10.0, jnp.log(S0), 0.0, 1.0])
S_F    = 150.0
V_max  = 4.0
F_max  = 2.0
F_min  = 0.0
t_end  = 9.0

# ── Backward shooting (same as main script) ───────────────────────────────────
print("Setting up: backward shooting ...")
root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event       = diffrax.Event(volume_cond, root_finder)

sol_event = diffeqsolve(
    ODETerm(states_analytical_sol), diffrax.Kvaerno5(),
    t0=0, t1=t_end, dt0=0.01, y0=y0, event=event,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    args=(S_F, V_max)
)
t_switch = float(sol_event.ts.squeeze())

sol_sing_ana = diffeqsolve(
    ODETerm(states_analytical_sol), diffrax.Kvaerno5(),
    t0=0, t1=t_switch, dt0=0.01, y0=y0,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(0.0, t_switch, 300)), args=(S_F, V_max)
)
sol_batch_ana = diffeqsolve(
    ODETerm(states_analytical_sol), diffrax.Kvaerno5(),
    t0=t_switch, t1=t_end, dt0=0.01, y0=sol_sing_ana.ys[-1],
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(t_switch, t_end, 300)),
    args=(S_F, V_max), max_steps=int(1e6)
)
x_tf = sol_batch_ana.ys[-1]

opt_bwd = optx.minimise(
    fn=loss_fn_batch, solver=optx.BFGS(rtol=1e-5, atol=1e-6), y0=-0.1,
    args=(t_end, y0, S_F, x_tf, t_switch, V_max, F_min, F_max), max_steps=500
)
l4_tf    = opt_bwd.value
y_tf_opt = jnp.array([x_tf[0], x_tf[1], x_tf[2], x_tf[3], 0.0, 0.0, 1.0, l4_tf])

# Backward batch integration → y_at_tswitch (exact starting point for forward)
sol_bwd_batch = diffeqsolve(
    ODETerm(lambda t, y, args: vector_field(t, y, (S_F, 0.0))),
    diffrax.Kvaerno5(), t0=t_end, t1=t_switch, dt0=-0.01, y0=y_tf_opt,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    max_steps=int(1e5), saveat=SaveAt(ts=jnp.linspace(t_end, t_switch, 500))
)
y_at_tswitch = sol_bwd_batch.ys[-1]

# Backward singular arc → y_at_t0
sol_bwd_sing = diffeqsolve(
    ODETerm(lambda t, y, args: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))))),
    diffrax.Kvaerno5(), t0=t_switch, t1=0.0, dt0=-0.01, y0=y_at_tswitch,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    max_steps=int(1e5), saveat=SaveAt(ts=jnp.linspace(t_switch, 0.0, 500))
)
y_at_t0 = sol_bwd_sing.ys[-1]

# Backward reference in forward time order
t_ref_sing  = np.array(sol_bwd_sing.ts[::-1])
y_ref_sing  = np.array(sol_bwd_sing.ys[::-1])
t_ref_batch = np.array(sol_bwd_batch.ts[::-1])
y_ref_batch = np.array(sol_bwd_batch.ys[::-1])
t_ref = np.concatenate([t_ref_sing, t_ref_batch])
y_ref = np.concatenate([y_ref_sing, y_ref_batch], axis=0)

print(f"  t_switch = {t_switch:.4f} h,  l4_tf = {float(l4_tf):.6f}")
print(f"  λ₂(t=0) from backward = {float(y_at_t0[5]):.8f}")
print(f"  λ₂(t_switch) from backward = {float(y_at_tswitch[5]):.8f}\n")

# ── numpy vector field for scipy solvers ──────────────────────────────────────
_vf_batch_jit = jax.jit(lambda y: vector_field(0.0, jnp.array(y), (S_F, 0.0)))
_vf_sing_jit  = jax.jit(lambda y: vector_field(0.0, jnp.array(y), (S_F, get_Fsin_unconstrained(jnp.array(y), (S_F, V_max, F_min, F_max)))))

def vf_batch_np(t, y): return np.array(_vf_batch_jit(y))
def vf_sing_np(t, y):  return np.array(_vf_sing_jit(y))

# Combined vector field: singular arc + batch (for full trajectory)
_vf_full_jit = jax.jit(lambda t, y: jnp.where(
    t < t_switch,
    vector_field(t, jnp.array(y), (S_F, get_Fsin_unconstrained(jnp.array(y), (S_F, V_max, F_min, F_max)))),
    vector_field(t, jnp.array(y), (S_F, 0.0))
))
def vf_full_np(t, y): return np.array(_vf_full_jit(t, y))

# ── Helper: run one scipy forward integration ─────────────────────────────────
SAVE_PTS = 500

def run_scipy(method, rtol, atol, phase='batch'):
    if phase == 'batch':
        vf, t0, t1, y_start = vf_batch_np, t_switch, t_end, np.array(y_at_tswitch)
    else:
        vf, t0, t1, y_start = vf_full_np, 0.0, t_end, np.array(y_at_t0)
    t_eval = np.linspace(t0, t1, SAVE_PTS)
    try:
        sol = solve_ivp(vf, [t0, t1], y_start, method=method,
                        rtol=rtol, atol=atol, t_eval=t_eval, dense_output=False)
        l2_end = float(sol.y[5, -1])
        return sol.t, sol.y.T, l2_end, sol.nfev
    except Exception as e:
        return None, None, float('nan'), 0

# Helper: run one diffrax forward integration (batch phase only)
def run_diffrax(solver, rtol, atol):
    try:
        sol = diffeqsolve(
            ODETerm(lambda t, y, args: vector_field(t, y, (S_F, 0.0))),
            solver, t0=t_switch, t1=t_end, dt0=0.001, y0=y_at_tswitch,
            stepsize_controller=PIDController(rtol=rtol, atol=atol),
            max_steps=int(1e6),
            saveat=SaveAt(ts=jnp.linspace(t_switch, t_end, SAVE_PTS))
        )
        l2_end = float(sol.ys[-1, 5])
        return np.array(sol.ts), np.array(sol.ys), l2_end
    except Exception as e:
        return None, None, float('nan')

# ── Run comparisons ───────────────────────────────────────────────────────────
print("=" * 74)
print(f"{'Solver / library':30s}  {'rtol':>8}  {'λ₂(t_end)':>14}  {'|error|':>12}")
print("-" * 74)

results = {}   # label → (t, y, l2_end)

# --- Diffrax solvers (batch phase: t_switch → t_end) ---
diffrax_solvers = [
    ("Kvaerno5  [diffrax]",  diffrax.Kvaerno5(),  1e-8),
    ("Kvaerno3  [diffrax]",  diffrax.Kvaerno3(),  1e-8),
    ("Tsit5     [diffrax]",  diffrax.Tsit5(),     1e-8),
    ("Dopri5    [diffrax]",  diffrax.Dopri5(),    1e-8),
    ("Dopri8    [diffrax]",  diffrax.Dopri8(),    1e-8),
    ("Kvaerno5  [diffrax] tight",  diffrax.Kvaerno5(), 1e-12),
    ("Kvaerno3  [diffrax] tight",  diffrax.Kvaerno3(), 1e-12),
]

for label, solver, rtol in diffrax_solvers:
    t_s, y_s, l2 = run_diffrax(solver, rtol=rtol, atol=rtol * 1e-2)
    results[label] = (t_s, y_s, l2)
    print(f"  {label:30s}  {rtol:>8.0e}  {l2:>14.4e}  {abs(l2):>12.3e}")

print()

# --- Scipy solvers (batch phase: t_switch → t_end) ---
scipy_batch = [
    ("RK45      [scipy]",   "RK45",   1e-8,  1e-10),
    ("RK23      [scipy]",   "RK23",   1e-8,  1e-10),
    ("DOP853    [scipy]",   "DOP853", 1e-8,  1e-10),
    ("Radau     [scipy]",   "Radau",  1e-8,  1e-10),
    ("BDF       [scipy]",   "BDF",    1e-8,  1e-10),
    ("LSODA     [scipy]",   "LSODA",  1e-8,  1e-10),
    ("Radau     [scipy] tight", "Radau", 1e-12, 1e-14),
    ("BDF       [scipy] tight", "BDF",   1e-12, 1e-14),
]

for label, method, rtol, atol in scipy_batch:
    t_s, y_s, l2, nfev = run_scipy(method, rtol, atol, phase='batch')
    results[label] = (t_s, y_s, l2)
    print(f"  {label:30s}  {rtol:>8.0e}  {l2:>14.4e}  {abs(l2):>12.3e}")

print()

# --- Full trajectory (0 → t_end) with scipy ---
print("Full trajectory (0 → t_end):")
full_traj = [
    ("Radau     [scipy] full",  "Radau",  1e-8,  1e-10),
    ("BDF       [scipy] full",  "BDF",    1e-8,  1e-10),
    ("Radau     [scipy] full tight", "Radau", 1e-12, 1e-14),
]
for label, method, rtol, atol in full_traj:
    t_s, y_s, l2, nfev = run_scipy(method, rtol, atol, phase='full')
    results[label] = (t_s, y_s, l2)
    print(f"  {label:30s}  {rtol:>8.0e}  {l2:>14.4e}  {abs(l2):>12.3e}")

print("=" * 74)
print("  λ₂(t_end) expected = 0.0")
print("  λ₂(t_end) backward = 0.0  (starts from exact terminal condition)")

# ── Plot: λ₂ trajectories for a selection of solvers ─────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Forward single shooting — solver comparison\n'
             'All start from the backward-reference initial condition at t_switch', fontsize=13)

# Panel 1: full range (shows explosion)
ax = axes[0]
ax.plot(t_ref_batch, y_ref_batch[:, 5], 'k-', lw=2.5, label='Backward (reference)', zorder=6)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
plot_solvers = ["Kvaerno5  [diffrax]", "Radau     [scipy]",
                "BDF       [scipy]",   "Tsit5     [diffrax]",
                "Kvaerno5  [diffrax] tight"]
for col, lbl in zip(colors, plot_solvers):
    t_s, y_s, l2 = results.get(lbl, (None, None, None))
    if t_s is not None:
        clipped = np.clip(y_s[:, 5], -1.0, 1.0)
        ax.plot(t_s, clipped, '--', color=col, lw=1.4,
                label=f'{lbl.split("[")[0].strip()} (λ₂(end)={l2:.2e})')

ax.set_ylabel(r'$\lambda_2$ (clipped to [−1, 1])')
ax.set_title(r'$\lambda_2$ — all solvers eventually diverge from the backward reference')
ax.set_xlim(t_switch, t_end)
ax.legend(fontsize=8); ax.grid(True)

# Panel 2: early part of batch phase (before explosion)
ax = axes[1]
t_mask = t_ref_batch <= t_switch + 0.15
ax.plot(t_ref_batch[t_mask], y_ref_batch[t_mask, 5], 'k-', lw=2.5,
        label='Backward (reference)', zorder=6)
for col, lbl in zip(colors, plot_solvers):
    t_s, y_s, l2 = results.get(lbl, (None, None, None))
    if t_s is not None:
        mask = t_s <= t_switch + 0.15
        ax.plot(t_s[mask], y_s[mask, 5], '--', color=col, lw=1.4,
                label=lbl.split("[")[0].strip())

ax.set_ylabel(r'$\lambda_2$')
ax.set_xlabel('Time (h)')
ax.set_title('Early batch phase — solvers agree before the instability dominates')
ax.set_xlim(t_switch, t_switch + 0.15)
ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(script_dir, 'output_forward_single_shooting.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {out}")
