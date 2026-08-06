"""
Script to solve the batch phase as a preliminary step.  
"""
import optimistix as optx
import jax
import diffrax
import jax.numpy as jnp
from diffrax import ODETerm, Kvaerno5, SaveAt, diffeqsolve, Dopri5
import numpy as np
import matplotlib.pyplot as plt
from singulax.substrate_inhibition.analytical import states_analytical_sol, calculate_real_feed, volume_cond
from singulax.substrate_inhibition.shooting import loss_fn_batch

jax.config.update("jax_enable_x64", True)

Ks = 0.1
Ki = 100
S0 = jnp.sqrt(Ks*Ki)
S_F = 150
V_max = 4.0
F_max = 2.0
F_min = 0.0
t_end = 9.0

# Get x_switch directly from forward analytical solution
y0 = jnp.array([10, jnp.log(S0), 0., 1.])

sol_ana = diffeqsolve(
    ODETerm(states_analytical_sol), Kvaerno5(),
    t0=0, t1=t_end, dt0=0.01, y0=y0,
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    args=(S_F, V_max),
    saveat=SaveAt(ts=jnp.linspace(0, t_end, 2000))
)

XV_tf = sol_ana.ys[-1, 0]
SV_tf = sol_ana.ys[-1, 1]
PV_tf = sol_ana.ys[-1, 2]
V_tf = sol_ana.ys[-1, 3]

x_tf =  jnp.array([XV_tf, SV_tf, PV_tf, V_tf])
print("Final states:", jnp.array([XV_tf, jnp.exp(SV_tf), PV_tf, V_tf]))

get_Fs = jax.vmap(lambda y: calculate_real_feed(y[:4],(S_F, V_max, Ks, Ki)))
Fs = get_Fs(sol_ana.ys)

root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
event = diffrax.Event(volume_cond, root_finder)

# Run the SINGULAR phase to find t_switch
sol = diffeqsolve(
    ODETerm(states_analytical_sol), 
    Kvaerno5(),
    t0=0, t1=t_end, dt0=0.01, y0=y0,
    event=event,
    stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-10),
    args=(S_F, V_max)
)
t_switch = sol.ts.squeeze()

print(f"Switching time:", t_switch)

# Singular arc phase
if __name__ == "__main__":
    print("=== Optimal Control Optimization for Singular Arc ===\n")
    l4_tf_guess = -0.1
    print("Final guess l4tf = ",l4_tf_guess)

    solver = optx.BFGS(rtol=1e-5, atol=1e-6)
    print("\nStarting optimization")
    solution = optx.minimise(
        fn = loss_fn_batch,
        solver=solver,
        y0=l4_tf_guess,
        args=(t_end, y0, S_F, x_tf, t_switch, V_max, F_min, F_max),
        max_steps=500
    )

    if solution is not None:
        optimal_guess = solution.value
        print("Optimal solution found!:", optimal_guess)
