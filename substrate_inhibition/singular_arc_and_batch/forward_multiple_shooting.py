"""
Full forward simulation of the PMP optimal trajectory using multiple shooting.

The backward shooting solution is used to initialise the shooting nodes.
Multiple shooting limits the per-segment error amplification to e^{λ·T/N}
instead of e^{λ·T} for the full interval.

Singular arc (0 → t_switch):  N_sing  = 200 segments, F = F_sin(y)
Batch phase  (t_switch → t_end): N_batch = 100 segments, F = 0
"""

import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx
import matplotlib.pyplot as plt
import os
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, PIDController
from scipy.integrate import solve_ivp

from fedopt.odes import vector_field
from fedopt.si_analytical import states_analytical_sol, volume_cond
from fedopt.shooting import loss_fn_batch
from fedopt.Fsin import get_Fsin_unconstrained

jax.config.update("jax_enable_x64", True)

Ks, Ki  = 0.1, 100.0
S0      = jnp.sqrt(Ks * Ki)
y0      = jnp.array([10.0, jnp.log(S0), 0.0, 1.0])
S_F     = 150.0
V_max   = 4.0
F_max   = 2.0
F_min   = 0.0
t_end   = 9.0

@jax.jit
def _rhs_sing(t, y_jax):
    F = get_Fsin_unconstrained(y_jax, (S_F, V_max, F_min, F_max))
    return vector_field(t, y_jax, (S_F, F))

@jax.jit
def _rhs_batch(t, y_jax):
    return vector_field(t, y_jax, (S_F, 0.0))

def vf_sing_np(t, y):
    return np.array(_rhs_sing(t, jnp.array(y)))

def vf_batch_np(t, y):
    return np.array(_rhs_batch(t, jnp.array(y)))

# ── Helper: forward-integrate one segment with scipy ─────────────────────────
def fwd_segment(vf_np, y_start, t0, t1, n_save=6):
    sol = solve_ivp(vf_np, [t0, t1], np.array(y_start),
                    method='RK45', rtol=1e-8, atol=1e-10,
                    t_eval=np.linspace(t0, t1, n_save))
    return sol.t, sol.y.T

# ── Step 1: analytical solution → t_switch and x_tf ─────────────────────────
print("Step 1 — analytical solution ...")
root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event       = diffrax.Event(volume_cond, root_finder)

sol_event = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=0, t1=t_end, dt0=0.01, y0=y0, event=event,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    args=(S_F, V_max)
)
t_switch = float(sol_event.ts.squeeze())
print(f"  t_switch = {t_switch:.4f} h,  batch phase = {t_end - t_switch:.4f} h")

sol_sing_ana = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=0, t1=t_switch, dt0=0.01, y0=y0,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(0.0, t_switch, 300)),
    args=(S_F, V_max)
)
sol_batch_ana = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=t_switch, t1=t_end, dt0=0.01, y0=sol_sing_ana.ys[-1],
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(t_switch, t_end, 300)),
    args=(S_F, V_max), max_steps=int(1e6)
)
x_tf = sol_batch_ana.ys[-1]

# ── Step 2: backward shooting → l4_tf ────────────────────────────────────────
print("Step 2 — backward shooting ...")
opt = optx.minimise(
    fn=loss_fn_batch, solver=optx.BFGS(rtol=1e-5, atol=1e-6), y0=-0.1,
    args=(t_end, y0, S_F, x_tf, t_switch, V_max, F_min, F_max), max_steps=500
)
l4_tf    = opt.value
y_tf_opt = jnp.array([x_tf[0], x_tf[1], x_tf[2], x_tf[3], 0.0, 0.0, 1.0, l4_tf])
print(f"  l4_tf = {float(l4_tf):.6f}")

# ── Step 3: backward integration (reference trajectories) ────────────────────
print("Step 3 — backward integration ...")
REF_PTS = 600

# Batch phase: t_end → t_switch
sol_bwd_batch = diffeqsolve(
    ODETerm(lambda t, y, args: vector_field(t, y, (S_F, 0.0))),
    Kvaerno5(), t0=t_end, t1=t_switch, dt0=-0.01, y0=y_tf_opt,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    max_steps=int(1e5),
    saveat=SaveAt(ts=jnp.linspace(t_end, t_switch, REF_PTS))
)
y_at_tswitch = sol_bwd_batch.ys[-1]

# Singular arc: t_switch → 0
sol_bwd_sing = diffeqsolve(
    ODETerm(lambda t, y, args: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))))),
    Kvaerno5(), t0=t_switch, t1=0.0, dt0=-0.01, y0=y_at_tswitch,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    max_steps=int(1e5),
    saveat=SaveAt(ts=jnp.linspace(t_switch, 0.0, REF_PTS))
)
y_at_t0 = sol_bwd_sing.ys[-1]

# Reverse both to forward time order
t_ref_sing  = np.array(sol_bwd_sing.ts[::-1])   # 0 → t_switch
y_ref_sing  = np.array(sol_bwd_sing.ys[::-1])
t_ref_batch = np.array(sol_bwd_batch.ts[::-1])  # t_switch → t_end
y_ref_batch = np.array(sol_bwd_batch.ys[::-1])

print(f"  λ₁(t=0)       = {float(y_at_t0[4]):.6e}")
print(f"  λ₂(t=0)       = {float(y_at_t0[5]):.6e}")
print(f"  λ₂(t_switch)  = {float(y_at_tswitch[5]):.6e}")
print(f"  λ₂(t_end)     = {float(y_tf_opt[5]):.6e}  (terminal condition)")

# ── Step 4: multiple shooting — singular arc (0 → t_switch) ──────────────────
N_sing      = 200
print(f"\nStep 4 — multiple shooting: singular arc (N={N_sing}) ...")
seg_ts_sing = np.linspace(0.0, t_switch, N_sing + 1)

nodes_sing         = np.zeros((N_sing + 1, 8))
nodes_sing[0]      = np.array(y_at_t0)
nodes_sing[N_sing] = np.array(y_at_tswitch)
for k in range(1, N_sing):
    idx = int(np.argmin(np.abs(t_ref_sing - seg_ts_sing[k])))
    nodes_sing[k] = y_ref_sing[idx]

_ts_sing, _ys_sing = [], []
for k in range(N_sing):
    ts_k, ys_k = fwd_segment(vf_sing_np, nodes_sing[k],
                              seg_ts_sing[k], seg_ts_sing[k + 1])
    _ts_sing.append(ts_k)
    _ys_sing.append(ys_k)

t_ms_sing = np.concatenate(_ts_sing)
y_ms_sing = np.concatenate(_ys_sing, axis=0)

gap_sing = np.max(np.abs(y_ms_sing[-1] - np.array(y_at_tswitch)))
print(f"  Junction error at t_switch (max over state+costate): {gap_sing:.3e}")
print(f"  λ₁(t_switch) MS={y_ms_sing[-1, 4]:.6e}  ref={float(y_at_tswitch[4]):.6e}")
print(f"  λ₂(t_switch) MS={y_ms_sing[-1, 5]:.6e}  ref={float(y_at_tswitch[5]):.6e}")

# ── Step 5: multiple shooting — batch phase (t_switch → t_end) ───────────────
N_batch      = 100
print(f"\nStep 5 — multiple shooting: batch phase (N={N_batch}) ...")
seg_ts_batch = np.linspace(t_switch, t_end, N_batch + 1)

nodes_batch          = np.zeros((N_batch + 1, 8))
nodes_batch[0]       = np.array(y_at_tswitch)
nodes_batch[N_batch] = np.array(y_tf_opt)
for k in range(1, N_batch):
    idx = int(np.argmin(np.abs(t_ref_batch - seg_ts_batch[k])))
    nodes_batch[k] = y_ref_batch[idx]

_ts_batch, _ys_batch = [], []
for k in range(N_batch):
    ts_k, ys_k = fwd_segment(vf_batch_np, nodes_batch[k],
                              seg_ts_batch[k], seg_ts_batch[k + 1])
    _ts_batch.append(ts_k)
    _ys_batch.append(ys_k)

t_ms_batch = np.concatenate(_ts_batch)
y_ms_batch = np.concatenate(_ys_batch, axis=0)

gap_batch = np.max(np.abs(y_ms_batch[-1] - np.array(y_tf_opt)))
print(f"  Junction error at t_end (max over state+costate):    {gap_batch:.3e}")
print(f"  λ₂(t_end) MS={y_ms_batch[-1, 5]:.6e}  expected=0.0")

# ── Assemble full trajectory ──────────────────────────────────────────────────
t_ms_full = np.concatenate([t_ms_sing, t_ms_batch])
y_ms_full = np.concatenate([y_ms_sing, y_ms_batch], axis=0)

t_ref_full = np.concatenate([t_ref_sing, t_ref_batch])
y_ref_full = np.concatenate([y_ref_sing, y_ref_batch], axis=0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(12, 14))
fig.suptitle(
    f'Full forward simulation — multiple shooting\n'
    f'Singular arc: N={N_sing},  Batch: N={N_batch}',
    fontsize=14
)

plot_cfg = [
    (4, r'$\lambda_1$'),
    (5, r'$\lambda_2$'),
    (6, r'$\lambda_3$'),
    (7, r'$\lambda_4$'),
]

for ax, (idx, lbl) in zip(axes, plot_cfg):
    ax.plot(t_ref_full, y_ref_full[:, idx], 'k-', lw=2.5,
            label=f'{lbl} backward (reference)', zorder=6)
    ax.plot(t_ms_full,  y_ms_full[:,  idx], 'r--', lw=1.5,
            label=f'{lbl} forward MS')
    ax.axvline(t_switch, color='steelblue', linestyle=':', lw=1.2,
               label='t_switch')
    ax.set_ylabel(lbl, fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True)

axes[-1].set_xlabel('Time (h)', fontsize=11)

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(script_dir, 'output_forward_ms_full.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {out}")
