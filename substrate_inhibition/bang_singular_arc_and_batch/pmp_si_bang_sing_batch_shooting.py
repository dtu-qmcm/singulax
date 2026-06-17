"""
pmp_si_bang_sing_batch_shooting.py

Two-variable shooting for the Bang → Singular arc → Batch control structure.
S0 < S★, so the optimal feed is: F_max  [0, t1]  →  F_sin  [t1, t2]  →  0  [t2, t_end]

Unknowns:  t1    (bang → singular switching time)
           l4_tf (terminal costate λ₄(t_end))

Conditions: φ(t1) = 0   (smooth switch bang → singular)
            φ(t2) = 0   (smooth switch singular → batch)

Algorithm (repeated until residuals → 0):
  1. Forward states:  bang [0,t1] → singular arc [t1,t2] → batch [t2,t_end]
     t2 is found by event detection (V = V_max)
  2. Build terminal costate:  y_tf = [x_tf, 0, 0, 1, l4_tf]
  3. Backward costates:  batch [t_end,t2] → singular [t2,t1]
  4. Residuals = [φ(t1), φ(t2)]
  5. scipy.optimize.fsolve iterates using finite-difference Jacobian
"""

import diffrax
import os
import optimistix as optx
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, PIDController
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import fsolve
from fedopt.si_analytical import (states_analytical_sol,
                                   states_analytical_sol_bang, volume_cond)
from fedopt.odes import get_phi, vector_field
from fedopt.Fsin import get_Fsin_unconstrained

jax.config.update("jax_enable_x64", True)

# ── Problem parameters (identical to pmp_si_bang_sing_batch.py) ───────────────
Ks, Ki  = 0.1, 100.0
S0      = 1e-6
F_max   = 2.0
F_min   = 0.0
y0      = jnp.array([10.0, jnp.log(S0), 0.0, 1.0])   # [XV, log(SV), PV, V]
S_F     = 150.0
V_max   = 4.0
t_end   = 9.0

S_star  = jnp.sqrt(Ks * Ki)

# ── Core residuals function ───────────────────────────────────────────────────
def solve_and_get_residuals(params, args):
    """
    params = jnp.array([t1, l4_tf])
    Returns jnp.array([φ(t1), φ(t2)]) — both zero at the optimum.

    t_full (= t2) is recomputed from scratch inside for each params,
    so the optimiser sees the correct sensitivity.
    """
    t1, l4_tf = params[0], params[1]
    S_F_, V_max_, F_min_, F_max_, t_end_ = args

    pid    = PIDController(rtol=1e-8,  atol=1e-8,  dtmin=1e-10)
    pid_bw = PIDController(rtol=1e-10, atol=1e-10)
    root_finder = optx.Newton(1e-6, 1e-6, optx.rms_norm)

    # ── Forward: bang arc [0, t1] ─────────────────────────────────────────────
    sol_bang = diffeqsolve(
        ODETerm(states_analytical_sol_bang), Kvaerno5(),
        t0=0.0, t1=t1, dt0=0.01, y0=y0,
        stepsize_controller=pid, saveat=SaveAt(t1=True),
        args=(S_F_, F_max_)
    )
    y_at_t1 = sol_bang.ys[-1]

    # ── t2: when V hits V_max during the singular arc ─────────────────────────
    sol_event = diffeqsolve(
        ODETerm(states_analytical_sol), Kvaerno5(),
        t0=t1, t1=t_end_, dt0=0.01, y0=y_at_t1,
        event=diffrax.Event(volume_cond, root_finder),
        stepsize_controller=pid,
        args=(S_F_, V_max_)
    )
    t2       = sol_event.ts.squeeze()
    y_at_t2  = sol_event.ys[-1]          # state at the event time

    # ── Forward: batch [t2, t_end] ────────────────────────────────────────────
    sol_batch = diffeqsolve(
        ODETerm(states_analytical_sol), Kvaerno5(),
        t0=t2, t1=t_end_, dt0=0.01, y0=y_at_t2,
        stepsize_controller=pid, saveat=SaveAt(t1=True),
        args=(S_F_, V_max_), max_steps=int(1e6)
    )
    x_tf = sol_batch.ys[-1]

    # ── Terminal costate ──────────────────────────────────────────────────────
    y_tf = jnp.array([x_tf[0], x_tf[1], x_tf[2], x_tf[3],
                      0.0, 0.0, 1.0, l4_tf])

    # ── Backward: batch [t_end, t2] ───────────────────────────────────────────
    sol_bwd_b = diffeqsolve(
        ODETerm(lambda t, y, args: vector_field(t, y, (S_F_, 0.0))),
        Kvaerno5(), t0=t_end_, t1=t2, dt0=-0.01, y0=y_tf,
        stepsize_controller=pid_bw, saveat=SaveAt(t1=True),
        max_steps=int(1e5)
    )
    y_at_t2_bwd = sol_bwd_b.ys[-1]

    # ── Backward: singular arc [t2, t1] ──────────────────────────────────────
    sol_bwd_s = diffeqsolve(
        ODETerm(lambda t, y, args: vector_field(
            t, y, (S_F_, get_Fsin_unconstrained(y, (S_F_, V_max_, F_min_, F_max_)))
        )),
        Kvaerno5(), t0=t2, t1=t1, dt0=-0.01, y0=y_at_t2_bwd,
        stepsize_controller=pid_bw, saveat=SaveAt(t1=True),
        max_steps=int(1e5)
    )
    y_at_t1_bwd = sol_bwd_s.ys[-1]

    return jnp.array([get_phi(y_at_t1_bwd), get_phi(y_at_t2_bwd)])


# ── Initial guess and problem args ───────────────────────────────────────────
t1_guess    = 0.01
l4_tf_guess = -16.0
opt_args    = (S_F, V_max, F_min, F_max, t_end)
params0     = jnp.array([t1_guess, l4_tf_guess])

print("=" * 55)
print("  Bang → Singular → Batch  two-variable shooting")
print("=" * 55)
print(f"  Initial guess:  t1 = {t1_guess}   l4_tf = {l4_tf_guess}")
res0 = solve_and_get_residuals(params0, opt_args)
print(f"  Initial residuals:  φ(t1) = {float(res0[0]):.4e}   φ(t2) = {float(res0[1]):.4e}")
print()

# ── Root-finding via scipy.optimize.fsolve ────────────────────────────────────
# fsolve uses a finite-difference Jacobian — avoids differentiating through
# diffrax.Event, which causes NaN/inf in any JAX-autodiff-based optimizer.
def residuals_np(params_np):
    params = jnp.array(params_np)
    res    = solve_and_get_residuals(params, opt_args)
    np_res = np.array(res)
    print(f"    fsolve iter:  t1 = {params_np[0]:.6f}  l4_tf = {params_np[1]:.6f}"
          f"  →  φ(t1) = {np_res[0]:.3e}  φ(t2) = {np_res[1]:.3e}")
    return np_res

print("  Running fsolve ...")
sol_fsolve, info, ier, msg = fsolve(
    residuals_np,
    [t1_guess, l4_tf_guess],
    full_output=True,
    xtol=1e-8
)
print(f"\n  fsolve exit flag = {ier}  ({msg})")

t1_opt    = float(sol_fsolve[0])
l4_tf_opt = float(sol_fsolve[1])

print(f"  Optimal:  t1 = {t1_opt:.6f} h   l4_tf = {l4_tf_opt:.6f}")
res_f = solve_and_get_residuals(jnp.array(sol_fsolve), opt_args)
print(f"  Final residuals:  φ(t1) = {float(res_f[0]):.2e}   φ(t2) = {float(res_f[1]):.2e}")

# ── Reconstruct full trajectory with dense saving ─────────────────────────────
print("\n  Reconstructing trajectory ...")
t1_f = jnp.array(t1_opt)

sol_bang_f = diffeqsolve(
    ODETerm(states_analytical_sol_bang), Kvaerno5(),
    t0=0.0, t1=t1_f, dt0=0.01, y0=y0,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(0.0, t1_f, 300)), args=(S_F, F_max)
)
y_t1_f = sol_bang_f.ys[-1]

# t2 via event detection with optimal t1
sol_ev_f = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=t1_f, t1=t_end, dt0=0.01, y0=y_t1_f,
    event=diffrax.Event(volume_cond, optx.Newton(1e-6, 1e-6, optx.rms_norm)),
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    args=(S_F, V_max)
)
t2_f   = float(sol_ev_f.ts.squeeze())
y_t2_f = sol_ev_f.ys[-1]

sol_sing_f = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=t1_f, t1=t2_f, dt0=0.01, y0=y_t1_f,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(t1_f, t2_f, 400)),
    args=(S_F, V_max), max_steps=int(1e6)
)

sol_batch_f = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=t2_f, t1=t_end, dt0=0.01, y0=y_t2_f,
    stepsize_controller=PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=SaveAt(ts=jnp.linspace(t2_f, t_end, 300)),
    args=(S_F, V_max), max_steps=int(1e6)
)
x_tf_f = sol_batch_f.ys[-1]

# Backward costates
y_tf_f = jnp.array([x_tf_f[0], x_tf_f[1], x_tf_f[2], x_tf_f[3],
                     0.0, 0.0, 1.0, l4_tf_opt])

sol_bwd_b_f = diffeqsolve(
    ODETerm(lambda t, y, args: vector_field(t, y, (S_F, 0.0))), Kvaerno5(),
    t0=t_end, t1=t2_f, dt0=-0.01, y0=y_tf_f,
    stepsize_controller=PIDController(rtol=1e-10, atol=1e-10),
    saveat=SaveAt(ts=jnp.linspace(t_end, t2_f, 300)), max_steps=int(1e5)
)
sol_bwd_s_f = diffeqsolve(
    ODETerm(lambda t, y, args: vector_field(
        t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max)))
    )), Kvaerno5(),
    t0=t2_f, t1=t1_f, dt0=-0.01, y0=sol_bwd_b_f.ys[-1],
    stepsize_controller=PIDController(rtol=1e-10, atol=1e-10),
    saveat=SaveAt(ts=jnp.linspace(t2_f, t1_f, 400)), max_steps=int(1e5)
)
sol_bwd_bang_f = diffeqsolve(
    ODETerm(lambda t, y, args: vector_field(t, y, (S_F, F_max))), Kvaerno5(),
    t0=t1_f, t1=0.0, dt0=-0.01, y0=sol_bwd_s_f.ys[-1],
    stepsize_controller=PIDController(rtol=1e-10, atol=1e-10),
    saveat=SaveAt(ts=jnp.linspace(t1_f, 0.0, 300)), max_steps=int(1e5)
)

# Concatenate in forward-time order
t_fwd = np.concatenate([np.array(sol_bang_f.ts),
                         np.array(sol_sing_f.ts),
                         np.array(sol_batch_f.ts)])
y_fwd = np.concatenate([np.array(sol_bang_f.ys),
                         np.array(sol_sing_f.ys),
                         np.array(sol_batch_f.ys)])

t_bwd = np.concatenate([np.array(sol_bwd_bang_f.ts[::-1]),
                         np.array(sol_bwd_s_f.ts[::-1]),
                         np.array(sol_bwd_b_f.ts[::-1])])
y_bwd = np.concatenate([np.array(sol_bwd_bang_f.ys[::-1]),
                         np.array(sol_bwd_s_f.ys[::-1]),
                         np.array(sol_bwd_b_f.ys[::-1])])

XV_fwd = y_fwd[:, 0]
S_fwd  = np.exp(y_fwd[:, 1]) / y_fwd[:, 3]
PV_fwd = y_fwd[:, 2]
V_fwd  = y_fwd[:, 3]
phi_arr = np.array([float(get_phi(jnp.array(y_bwd[i]))) for i in range(len(t_bwd))])

print(f"\n  t1 (bang → singular) = {t1_opt:.4f} h")
print(f"  t2 (singular → batch) = {t2_f:.4f} h")
print(f"  XV(t_end) = {float(x_tf_f[0]):.4f} g")
print(f"  PV(t_end) = {float(x_tf_f[2]):.4f} g")
print(f"  S(t_end)  = {np.exp(float(x_tf_f[1]))/float(x_tf_f[3]):.4f} g/L")
print(f"  V(t_end)  = {float(x_tf_f[3]):.4f} L")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(13, 12))
fig.suptitle(
    f'PMP  —  Bang → Singular arc → Batch   (S₀ = {S0} g/L < S★ = {float(S_star):.3f} g/L)\n'
    f't₁ = {t1_opt:.4f} h   t₂ = {t2_f:.4f} h   λ₄(tf) = {l4_tf_opt:.4f}',
    fontsize=12
)

def shade(ax):
    ax.axvspan(0,      t1_opt, alpha=0.10, color='red',   label='bang')
    ax.axvspan(t1_opt, t2_f,   alpha=0.10, color='green', label='singular')
    ax.axvspan(t2_f,   t_end,  alpha=0.10, color='blue',  label='batch')
    ax.axvline(t1_opt, color='gray', ls='--', lw=0.9)
    ax.axvline(t2_f,   color='gray', ls='--', lw=0.9)

ax = axes[0, 0]
ax.plot(t_fwd, XV_fwd, 'r-', lw=2)
shade(ax); ax.set_ylabel('XV  (g)'); ax.set_title('Biomass × Volume')
ax.legend(fontsize=7, loc='upper left'); ax.grid(True)

ax = axes[0, 1]
ax.plot(t_fwd, S_fwd, 'b-', lw=2)
ax.axhline(float(S_star), color='k', ls=':', lw=1.5, label=f'S★ = {float(S_star):.3f} g/L')
shade(ax); ax.set_ylabel('S  (g/L)'); ax.set_title('Substrate concentration')
ax.legend(fontsize=7); ax.grid(True)

ax = axes[1, 0]
ax.plot(t_fwd, PV_fwd, 'g-', lw=2)
shade(ax); ax.set_ylabel('PV  (g)'); ax.set_title('Product × Volume')
ax.legend(fontsize=7, loc='upper left'); ax.grid(True)

ax = axes[1, 1]
ax.plot(t_fwd, V_fwd, 'm-', lw=2)
ax.axhline(V_max, color='k', ls=':', lw=1.5, label=f'V_max = {V_max} L')
shade(ax); ax.set_ylabel('V  (L)'); ax.set_title('Volume')
ax.legend(fontsize=7); ax.grid(True)

ax = axes[2, 0]
ax.plot(t_bwd, phi_arr, 'k-', lw=2)
ax.axhline(0, color='r', ls='--', lw=1.2, label='φ = 0')
ax.axvline(t1_opt, color='gray', ls='--', lw=0.9, label=f't₁ = {t1_opt:.3f} h')
ax.axvline(t2_f,   color='gray', ls=':',  lw=0.9, label=f't₂ = {t2_f:.3f} h')
ax.set_ylabel('φ = λ₂·S_F + λ₄'); ax.set_title('Switching function')
ax.set_xlabel('Time (h)'); ax.legend(fontsize=7); ax.grid(True)

ax = axes[2, 1]
ax.plot(t_bwd, y_bwd[:, 5], label='λ₂', lw=2)
ax.plot(t_bwd, y_bwd[:, 7], label='λ₄', lw=2)
ax.axvline(t1_opt, color='gray', ls='--', lw=0.9)
ax.axvline(t2_f,   color='gray', ls=':',  lw=0.9)
ax.set_ylabel('Costate'); ax.set_title('λ₂ and λ₄')
ax.set_xlabel('Time (h)'); ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(script_dir, 'output_shooting_bang_sing_batch.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to {out}")
