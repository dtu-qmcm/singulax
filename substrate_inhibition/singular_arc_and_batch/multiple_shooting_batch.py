"""
Multiple shooting for the batch-phase costate forward integration.

The batch-phase costate ODE has a positive eigenvalue (~641/h), making
single forward integration unstable.  Multiple shooting divides the
interval [t_switch, t_end] into N short segments.  Each segment is
forward-integrated independently, limiting the error amplification to
e^{641 * (T/N)} instead of e^{641 * T}.

The interior nodes are initialised from the backward solution.
We then check how well each forward segment reproduces the next node,
and compare the assembled forward trajectory to the backward reference.
"""

import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx
import matplotlib.pyplot as plt
import numpy as np
import os
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, PIDController

from scipy.integrate import solve_ivp
from fedopt.odes import vector_field
from fedopt.si_analytical import states_analytical_sol, volume_cond
from fedopt.shooting import loss_fn_batch

jax.config.update("jax_enable_x64", True)

# ── Problem parameters (identical to main script) ────────────────────────────
Ks, Ki  = 0.1, 100.0
S0      = jnp.sqrt(Ks * Ki)
y0      = jnp.array([10.0, jnp.log(S0), 0.0, 1.0])
S_F     = 150.0
V_max   = 4.0
F_max   = 2.0
F_min   = 0.0
t_end   = 9.0

# ── Analytical solution: find t_switch and x_tf ──────────────────────────────
print("Step 1 — analytical solution (finding t_switch and x_tf) ...")
root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event       = diffrax.Event(volume_cond, root_finder)

sol_event = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=0, t1=t_end, dt0=0.01, y0=y0, event=event,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    args=(S_F, V_max)
)
t_switch = float(sol_event.ts.squeeze())
print(f"  t_switch = {t_switch:.6f} h")

sol_sing = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=0, t1=t_switch, dt0=0.01, y0=y0,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(0.0, t_switch, 300)),
    args=(S_F, V_max)
)
sol_batch_ana = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=t_switch, t1=t_end, dt0=0.01, y0=sol_sing.ys[-1],
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(t_switch, t_end, 300)),
    args=(S_F, V_max), max_steps=int(1e6)
)
x_tf = sol_batch_ana.ys[-1]

# ── Backward shooting: find optimal l4_tf ────────────────────────────────────
print("Step 2 — backward shooting (finding l4_tf) ...")
opt = optx.minimise(
    fn=loss_fn_batch, solver=optx.BFGS(rtol=1e-5, atol=1e-6), y0=-0.1,
    args=(t_end, y0, S_F, x_tf, t_switch, V_max, F_min, F_max), max_steps=500
)
l4_tf     = opt.value
y_tf_opt  = jnp.array([x_tf[0], x_tf[1], x_tf[2], x_tf[3], 0.0, 0.0, 1.0, l4_tf])
print(f"  l4_tf = {float(l4_tf):.8f}")

# ── Backward batch integration (reference) ───────────────────────────────────
print("Step 3 — backward batch integration (reference) ...")
REF_PTS = 600
sol_bwd = diffeqsolve(
    terms=diffrax.ODETerm(lambda t, y, args: vector_field(t, y, (S_F, 0.0))),
    solver=Kvaerno5(), t0=t_end, t1=t_switch, dt0=-0.01, y0=y_tf_opt,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    max_steps=int(1e5),
    saveat=SaveAt(ts=jnp.linspace(t_end, t_switch, REF_PTS))
)
y_at_tswitch = sol_bwd.ys[-1]         # costate+state at t_switch
t_bwd = np.array(sol_bwd.ts[::-1])    # forward time order
y_bwd = np.array(sol_bwd.ys[::-1])    # forward time order

# ── Helper: numpy vector field (for scipy, avoids JAX recompilation per segment)
_vf_jit = jax.jit(lambda y: vector_field(0.0, jnp.array(y), (S_F, 0.0)))

def _vf_np(t, y):
    return np.array(_vf_jit(y))

# ── Helper: forward-integrate one batch segment with scipy ───────────────────
def fwd_segment_scipy(y_start, t0, t1, n_save=50):
    t_eval = np.linspace(t0, t1, n_save)
    sol = solve_ivp(_vf_np, [t0, t1], np.array(y_start),
                    method='RK45', rtol=1e-8, atol=1e-10,
                    t_eval=t_eval, dense_output=False)
    return sol.t, sol.y.T, sol.y[:, -1]   # (times, trajectory, endpoint)

# ── Single forward integration (N=1): should explode ─────────────────────────
print("\nStep 4 — single forward integration (N=1) ...")
_, y_fwd1, y_end1 = fwd_segment_scipy(np.array(y_at_tswitch), t_switch, t_end, n_save=REF_PTS)
l2_end_single = float(y_end1[5])
print(f"  λ₂ at t_end (single fwd) : {l2_end_single:.3e}")
print(f"  λ₂ at t_end (expected)   : 0.000e+00")

# ── Multiple shooting ─────────────────────────────────────────────────────────
def run_multiple_shooting(N):
    """
    Divide [t_switch, t_end] into N segments.
    Interior nodes are initialised from the backward solution.
    Returns:
        t_fwd  : concatenated time array (forward order)
        y_fwd  : concatenated state+costate array
        gap    : max absolute discontinuity at each junction (quality check)
        l2_end : λ₂ value at t_end
    """
    seg_ts = np.linspace(t_switch, t_end, N + 1)

    # Initialise nodes from the backward reference (interpolated at seg_ts)
    nodes = np.zeros((N + 1, 8))
    nodes[0] = np.array(y_at_tswitch)
    nodes[N] = np.array(y_tf_opt)
    for k in range(1, N):
        idx = int(np.argmin(np.abs(t_bwd - seg_ts[k])))
        nodes[k] = y_bwd[idx]

    # Forward-integrate each segment using scipy (no JAX recompilation per segment)
    all_ts, all_ys = [], []
    gaps = []
    n_save_seg = max(10, 60 // N)
    for k in range(N):
        ts_k, ys_k, y_end_k = fwd_segment_scipy(nodes[k], seg_ts[k], seg_ts[k + 1],
                                                  n_save=n_save_seg)
        all_ts.append(ts_k)
        all_ys.append(ys_k)
        gap = np.max(np.abs(y_end_k - nodes[k + 1]))
        gaps.append(gap)

    t_fwd = np.concatenate(all_ts)
    y_fwd = np.concatenate(all_ys, axis=0)
    return t_fwd, y_fwd, max(gaps), float(y_fwd[-1, 5])

# Run for a range of N values
N_list = [1, 5, 20, 50, 100, 200]

print("\n" + "=" * 72)
print(f"{'N':>5} | {'amplif/seg':>12} | {'max junction gap':>18} | {'λ₂(t_end) fwd':>16} | {'err vs 0':>10}")
print("-" * 72)

ms_results = {}
for N in N_list:
    dt_seg  = (t_end - t_switch) / N
    amplif  = np.exp(641.0 * dt_seg)
    t_f, y_f, gap, l2_end = run_multiple_shooting(N)
    ms_results[N] = (t_f, y_f)
    err = abs(l2_end)
    print(f"{N:>5} | {amplif:>12.3e} | {gap:>18.3e} | {l2_end:>16.6e} | {err:>10.3e}")

print("=" * 72)
print("  amplif/seg = e^(641 × segment_length):  amplification within each segment")
print("  junction gap: max |y_end_of_seg - node_init|  (0 = perfectly consistent)")
print("  λ₂(t_end) should be 0.0 (terminal condition)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 14))
fig.suptitle(
    "Multiple shooting — batch-phase costate forward integration\n"
    "Each segment uses the backward solution as its initial node",
    fontsize=13
)

colors_N = {1: 'red', 20: 'darkorange', 100: 'royalblue', 200: 'green'}
labels_N = {
    1:   'N=1   (single shot — explodes)',
    20:  'N=20  segments',
    100: 'N=100 segments',
    200: 'N=200 segments',
}

# Panel 1 — λ₂ (linear scale, clipped to see the stable solutions)
ax = axes[0]
ax.plot(t_bwd, y_bwd[:, 5], 'k-', lw=2.5, label='Backward (reference)', zorder=6)
for N, col in colors_N.items():
    t_f, y_f = ms_results[N]
    lam2 = np.clip(y_f[:, 5], -0.5, 0.5)    # clip extreme explosion for display
    ax.plot(t_f, lam2, '--', color=col, lw=1.5, label=labels_N[N], alpha=0.9)
ax.set_ylabel(r'$\lambda_2$  (clipped to [−0.5, 0.5])')
ax.set_title(r'$\lambda_2$ — forward multiple shooting vs backward reference')
ax.legend(fontsize=9); ax.grid(True)
ax.set_xlim(t_switch, t_end)

# Panel 2 — λ₁
ax = axes[1]
ax.plot(t_bwd, y_bwd[:, 4], 'k-', lw=2.5, label='Backward (reference)', zorder=6)
for N, col in colors_N.items():
    t_f, y_f = ms_results[N]
    ax.plot(t_f, y_f[:, 4], '--', color=col, lw=1.5, label=labels_N[N], alpha=0.9)
ax.set_ylabel(r'$\lambda_1$')
ax.set_title(r'$\lambda_1$ — forward multiple shooting vs backward reference')
ax.set_xlabel('Time (h)')
ax.legend(fontsize=9); ax.grid(True)
ax.set_xlim(t_switch, t_end)

# Panel 3 — terminal λ₂ error vs N (log scale, using already-computed results)
ax = axes[2]
N_done  = sorted(ms_results.keys())
errs    = [abs(ms_results[N][1][-1, 5]) for N in N_done]
amplifs_done = [np.exp(641.0 * (t_end - t_switch) / N) for N in N_done]

ax.semilogy(N_done, errs, 'ko-', lw=2, markersize=6, label='|λ₂(t_end)|')
ax.axhline(1e-8, color='steelblue', linestyle='--', lw=1.5, label='rtol = 1e-8')
ax.set_xlabel('Number of shooting segments  N')
ax.set_ylabel(r'$|\lambda_2(t_\mathrm{end})|$  (should → 0)')
ax.set_title('Terminal error vs number of segments\n'
             '(amplification per segment = $e^{641 \\cdot T/N}$)')
ax.legend(); ax.grid(True, which='both')

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(script_dir, 'output_multiple_shooting.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {out}")
