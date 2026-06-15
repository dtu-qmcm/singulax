"""
Full forward simulation using proper multiple shooting.

Workflow (mirrors the boc.multiple_shooting pattern):
 1. Define shooting nodes in time
 2. Initialise all node values from the backward reference solution
 3. Minimise continuity residuals with BFGS  (optx.minimise)
 4. Reconstruct the full trajectory by integrating forward segment by segment

The combined vector field switches automatically between the singular arc
(F = F_sin(y)) and the batch phase (F = 0) at t_switch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import optimistix as optx
import matplotlib.pyplot as plt
import os
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, PIDController

from fedopt.odes import vector_field
from fedopt.si_analytical import states_analytical_sol, volume_cond
from fedopt.shooting import loss_fn_batch
from fedopt.Fsin import get_Fsin_unconstrained

jax.config.update("jax_enable_x64", True)

# ── Parameters ────────────────────────────────────────────────────────────────
Ks, Ki     = 0.1, 100.0
S0         = jnp.sqrt(Ks * Ki)
y0         = jnp.array([10.0, jnp.log(S0), 0.0, 1.0])
S_F        = 150.0
V_max      = 4.0
F_max      = 2.0
F_min      = 0.0
t_end      = 9.0
num_states = 8

# ── Step 1: analytical solution → t_switch, x_tf ─────────────────────────────
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
print(f"  t_switch = {t_switch:.4f} h")

sol_sing_ana = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=0, t1=t_switch, dt0=0.01, y0=y0,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(0.0, t_switch, 300)), args=(S_F, V_max)
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
opt_bwd = optx.minimise(
    fn=loss_fn_batch, solver=optx.BFGS(rtol=1e-5, atol=1e-6), y0=-0.1,
    args=(t_end, y0, S_F, x_tf, t_switch, V_max, F_min, F_max), max_steps=500
)
l4_tf    = opt_bwd.value
y_tf_opt = jnp.array([x_tf[0], x_tf[1], x_tf[2], x_tf[3], 0.0, 0.0, 1.0, l4_tf])
print(f"  l4_tf = {float(l4_tf):.6f}")

# ── Step 3: backward integration → reference trajectories ────────────────────
print("Step 3 — backward integration (reference) ...")
REF_PTS = 600

sol_bwd_batch = diffeqsolve(
    ODETerm(lambda t, y, args: vector_field(t, y, (S_F, 0.0))),
    Kvaerno5(), t0=t_end, t1=t_switch, dt0=-0.01, y0=y_tf_opt,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    max_steps=int(1e5), saveat=SaveAt(ts=jnp.linspace(t_end, t_switch, REF_PTS))
)
y_at_tswitch = sol_bwd_batch.ys[-1]

sol_bwd_sing = diffeqsolve(
    ODETerm(lambda t, y, args: vector_field(t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))))),
    Kvaerno5(), t0=t_switch, t1=0.0, dt0=-0.01, y0=y_at_tswitch,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
    max_steps=int(1e5), saveat=SaveAt(ts=jnp.linspace(t_switch, 0.0, REF_PTS))
)
y_at_t0 = sol_bwd_sing.ys[-1]

t_ref_sing  = np.array(sol_bwd_sing.ts[::-1])
y_ref_sing  = np.array(sol_bwd_sing.ys[::-1])
t_ref_batch = np.array(sol_bwd_batch.ts[::-1])
y_ref_batch = np.array(sol_bwd_batch.ys[::-1])

# ── Step 4: shooting node layout ─────────────────────────────────────────────
# Singular arc: 5 segments  (low instability — few nodes suffice)
# Batch phase:  50 segments (per-segment amplification e^{641 × 0.012} ≈ 2300;
#               still manageable given the near-exact backward initialisation)
sing_nodes  = np.linspace(0.0, t_switch, 6)
batch_nodes = np.linspace(t_switch, t_end, 51)[1:]   # drop duplicate t_switch
shooting_nodes = np.concatenate([sing_nodes, batch_nodes])

num_segments = len(shooting_nodes) - 1   # 55
num_nodes    = len(shooting_nodes)        # 56

dt_batch = (t_end - t_switch) / 50
print(f"\nShooting nodes: {num_nodes}  ({num_segments} segments)")
print(f"  Singular arc  : 5 segments × {t_switch/5:.2f} h")
print(f"  Batch phase   : 50 segments × {dt_batch:.4f} h  (amplif/seg ≈ {np.exp(641*dt_batch):.1e})")

# ── Step 5: initialise node values from the backward reference ────────────────
params0 = np.zeros((num_nodes, num_states))
params0[0]  = np.array(y_at_t0)
params0[-1] = np.array(y_tf_opt)

for k in range(1, num_nodes - 1):
    tk = shooting_nodes[k]
    if tk <= t_switch + 1e-12:
        idx = int(np.argmin(np.abs(t_ref_sing - tk)))
        params0[k] = y_ref_sing[idx]
    else:
        idx = int(np.argmin(np.abs(t_ref_batch - tk)))
        params0[k] = y_ref_batch[idx]

params0_flat = jnp.array(params0.flatten())
print(f"\nparams0 shape: {params0.shape}  →  flat dim = {len(params0_flat)}")

# ── Step 6: combined vector field and continuity loss ────────────────────────
# t_switch is a Python float here — captured as a static constant in the
# closure so that JAX sees it as concrete at trace time.
_t_switch = t_switch
_t_pairs  = [(float(shooting_nodes[k]), float(shooting_nodes[k + 1]))
             for k in range(num_segments)]

def vf_full(t, y, args):
    """Single vector field for the whole trajectory; switches at t_switch."""
    F_sin = get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max))
    F     = jnp.where(t < _t_switch, F_sin, 0.0)
    return vector_field(t, y, (S_F, F))

def loss_fn_ms(params, args):
    """Sum of squared continuity gaps + boundary condition penalties."""
    y_ic, y_tc = args
    nodes = params.reshape(num_nodes, num_states)

    # Fixed boundary conditions (initial + terminal)
    total = 1e4 * jnp.sum((nodes[0]  - y_ic) ** 2)
    total = total + 1e4 * jnp.sum((nodes[-1] - y_tc) ** 2)

    # Continuity at each segment junction
    for k in range(num_segments):
        t0, t1 = _t_pairs[k]
        sol = diffeqsolve(
            ODETerm(vf_full), Kvaerno5(),
            t0=t0, t1=t1, dt0=0.01, y0=nodes[k],
            stepsize_controller=PIDController(rtol=1e-6, atol=1e-8),
            max_steps=int(2e4), saveat=SaveAt(t1=True), args=()
        )
        total = total + jnp.sum((sol.ys[-1] - nodes[k + 1]) ** 2)

    return total

# ── Step 7: BFGS optimisation ─────────────────────────────────────────────────
print("\nStep 4 — BFGS optimisation (continuity minimisation) ...")
print("  (first call triggers JAX compilation — may take a moment)")
ms_args = (jnp.array(y_at_t0), jnp.array(y_tf_opt))

initial_loss = float(loss_fn_ms(params0_flat, ms_args))
print(f"  Initial loss : {initial_loss:.4e}")

opt_ms = optx.minimise(
    fn=loss_fn_ms,
    solver=optx.BFGS(rtol=1e-6, atol=1e-8),
    y0=params0_flat,
    args=ms_args,
    max_steps=200
)
params_star = opt_ms.value
final_loss  = float(loss_fn_ms(params_star, ms_args))
print(f"  Final loss   : {final_loss:.4e}")
print(f"  Solver stats : {opt_ms.stats}")

# ── Step 8: reconstruct trajectory from optimal nodes ────────────────────────
print("\nStep 5 — reconstructing trajectory ...")
optimal_nodes = np.array(params_star).reshape(num_nodes, num_states)

all_ts, all_ys = [], []
for i in range(num_segments):
    t0_i   = float(shooting_nodes[i])
    t1_i   = float(shooting_nodes[i + 1])
    t_plot = jnp.linspace(t0_i, t1_i, 10)

    sol_seg = diffeqsolve(
        ODETerm(vf_full), Kvaerno5(),
        t0=t0_i, t1=t1_i, dt0=0.01, y0=jnp.array(optimal_nodes[i]),
        stepsize_controller=PIDController(rtol=1e-8, atol=1e-10),
        max_steps=int(2e4), saveat=SaveAt(ts=t_plot), args=()
    )
    all_ts.append(np.array(sol_seg.ts))
    all_ys.append(np.array(sol_seg.ys))

t_full = np.concatenate(all_ts)
y_full = np.concatenate(all_ys, axis=0)

t_ref_full = np.concatenate([t_ref_sing, t_ref_batch])
y_ref_full = np.concatenate([y_ref_sing, y_ref_batch], axis=0)

print(f"  λ₂(t_end) forward = {y_full[-1, 5]:.4e}  (expected 0)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(12, 14))
fig.suptitle(
    f'Full forward simulation — multiple shooting (BFGS optimised)\n'
    f'{num_nodes} nodes  |  5 singular-arc + 50 batch segments',
    fontsize=14
)

costate_cfg = [(4, r'$\lambda_1$'), (5, r'$\lambda_2$'),
               (6, r'$\lambda_3$'), (7, r'$\lambda_4$')]

for ax, (idx, lbl) in zip(axes, costate_cfg):
    ax.plot(t_ref_full, y_ref_full[:, idx], 'k-', lw=2.5,
            label=f'{lbl} backward ref', zorder=6)
    ax.plot(t_full,     y_full[:,     idx], 'r--', lw=1.5,
            label=f'{lbl} forward MS')
    ax.scatter(shooting_nodes, optimal_nodes[:, idx],
               s=15, color='steelblue', zorder=7, label='shooting nodes')
    ax.axvline(t_switch, color='gray', linestyle=':', lw=1.2)
    ax.set_ylabel(lbl, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True)

axes[-1].set_xlabel('Time (h)', fontsize=11)
plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(script_dir, 'output_forward_ms_full.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {out}")
