import diffrax
import os
import optimistix as optx
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, PIDController
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from fedopt.si_analytical import states_analytical_sol, calculate_real_feed, states_analytical_sol_bang, S_star_cond

jax.config.update("jax_enable_x64", True)

Ks = 0.1
Ki = 100
S0 = 1e-6   # initial substrate concentration, must be < S* = sqrt(Ks*Ki) ≈ 3.16
F_max = 2.0
y0 = jnp.array([10, jnp.log(S0 * 1.0), 0., 1.])
S_F = 150
V_max = 4
t_end = 9.0

root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event = diffrax.Event(S_star_cond, root_finder)

# Run the BANG phase to find t_switch (when S reaches S*)
sol = diffeqsolve(
    ODETerm(states_analytical_sol_bang),
    Kvaerno5(),
    t0=0, t1=t_end, dt0=0.01, y0=y0,
    event=event,
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    args=(S_F, F_max)
)
t_switch = sol.ts.squeeze()

print(f"Switching time:", t_switch)

# Solve ODE in two phases
solution_to_tswitch = diffeqsolve(
    ODETerm(states_analytical_sol_bang),
    Kvaerno5(),
    t0=0, t1=t_switch, dt0=0.01, y0=y0,
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_switch, 500)),
    args=(S_F, F_max)
)
y_at_switch = solution_to_tswitch.ys[-1]

solution_to_end = diffeqsolve(ODETerm(states_analytical_sol),
                              Kvaerno5(),
                              stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
                              t0=t_switch, t1=t_end, dt0=0.01,
                              y0=y_at_switch,
                              saveat= diffrax.SaveAt(ts=jnp.linspace(t_switch, t_end, 500)),
                              args=(S_F, V_max),
                              max_steps=int(1e6)                              
                              )

t_max = jnp.concatenate([solution_to_tswitch.ts, solution_to_end.ts])
y_max = jnp.concatenate([solution_to_tswitch.ys, solution_to_end.ys])

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 26
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'gray'
fig, axes = plt.subplots(3, 1, figsize=(14, 15))
fig.suptitle('Analytical solution for singular arc and batch phase', fontsize=28)

# Plot State variables
axes[0].plot(t_max, y_max[:,0], label='(XV)', color='royalblue')
axes[0].plot(t_max, jnp.exp(y_max[:,1]), label='(SV)', color='orange')
axes[0].plot(t_max, y_max[:,2], label='(PV)', color='green')
axes[0].plot(t_max, y_max[:,3], label='V', color='purple')
axes[0].set_ylabel('State (g or V)')
axes[0].set_xlabel('Time (h)')
axes[0].set_title('State variables')
axes[0].grid(True)
axes[0].legend()

# Plot S and P concentration
axes[1].plot(t_max, jnp.array(jnp.exp(y_max[:,1])) / jnp.array(y_max[:,3]), label='S', color='orange')
axes[1].plot(t_max, jnp.array(y_max[:,2]) / jnp.array(y_max[:,3]), label='P', color='green')
axes[1].set_ylabel('Concentration (g/L)')
axes[1].set_xlabel('Time (h)')
axes[1].set_title('S and P concentration')
axes[1].grid(True)
axes[1].legend()

# Plot feed: F_max during bang phase, singular feed during singular arc
F_bang = jnp.full(solution_to_tswitch.ts.shape, F_max)
F_sing = jax.vmap(lambda y: calculate_real_feed(jnp.array([y[0], jnp.exp(y[1]), y[2], y[3]]), (S_F, V_max, Ks, Ki)))(solution_to_end.ys)
F_actual = jnp.concatenate([F_bang, F_sing])
axes[2].plot(t_max, F_actual, label='Feed Rate', color='royalblue')
axes[2].set_ylabel('Feed rate (L/h)')
axes[2].set_xlabel('Time (h)')
axes[2].set_title('Feed rate')
axes[2].grid(True)
axes[2].legend()

# Print value of final states
print("Final XV value", y_max[-1, 0])
print("Final SV value", jnp.exp(y_max[-1, 1]))
print("Final PV value", y_max[-1, 2])
print("Final V value", y_max[-1, 3])

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "output_ana_si_bang_sing_batch.png"
output_path = os.path.join(script_dir, file_name)
plt.savefig(output_path, facecolor='white')
plt.show()
