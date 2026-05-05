import optimistix as optx
import os
import jax
import diffrax
import jax.numpy as jnp
from diffrax import ODETerm, Dopri5, SaveAt, diffeqsolve
import numpy as np
import matplotlib.pyplot as plt
from fedopt.si_analytical import states_analytical_sol
from fedopt.shooting import loss_fn
from fedopt.odes import vector_field, get_phi
from fedopt.Fsin import get_Fsin

jax.config.update("jax_enable_x64", True)

Ks = 0.1
Ki = 100
S0 = jnp.sqrt(Ks*Ki)
S_F = 150
V_max = 5.0
F_max = 2.0
F_min = 0.0
t_end = 9.0

# Get x_switch directly from forward analytical solution
y0 = jnp.array([10, jnp.log(S0), 0., 1.])
x0 = y0[:4]
l0 = y0[4:]
sol_ana = diffeqsolve(
    ODETerm(states_analytical_sol), Dopri5(),
    t0=0, t1=t_end, dt0=0.01, y0=y0,
    args=(S_F, V_max),
    saveat=SaveAt(ts=jnp.linspace(0, t_end, 2000))
)

XV_tf = sol_ana.ys[-1, 0]
SV_tf = sol_ana.ys[-1, 1]
PV_tf = sol_ana.ys[-1, 2]
V_tf = sol_ana.ys[-1, 3]

x_tf =  jnp.array([XV_tf, SV_tf, PV_tf, V_tf])
l_tf = jnp.array([0.0, 0.0, 1.0, 0.0])


# Singular arc phase
if __name__ == "__main__":
    print("=== Optimal Control Optimization for Singular Arc ===\n")

    final_guess = x_tf
    #final_guess = jnp.array([125, 13, 57, 3])
    print("Final guess:", final_guess)

    solver = optx.BFGS(rtol=1e-5, atol=1e-6)
    print("\nStarting optimization")
    solution = optx.minimise(
        fn = loss_fn,
        solver=solver,
        y0=final_guess,
        args=(t_end, x0, S_F, V_max, F_min, F_max),
        max_steps=500,
        throw=False
    )

    if solution is not None:
        optimal_xtf = solution.value

        ytf_opt = jnp.concatenate([optimal_xtf, l_tf])

        # Simulate backward to get optimal l0 and x0
        full_sol_backward = diffeqsolve(
        terms=diffrax.ODETerm(lambda t, y, args: vector_field(t, y, (S_F, get_Fsin(y, (S_F, V_max, F_min, F_max))))),
        solver=diffrax.Dopri5(),
        t0=t_end,
        t1=0.0,
        dt0=-0.01,
        y0=ytf_opt,
        saveat=diffrax.SaveAt(ts=jnp.linspace(t_end, 0.0, 500))
        )
        y_at_t0 = full_sol_backward.ys[-1] 
        x0_found = y_at_t0[:4]
        l0_found = y_at_t0[4:]

        # Simulate forward for plotting
        y0_forward = jnp.concatenate([x0_found, l0_found])
        ts = jnp.linspace(0.0, t_end, 500)
        sol = diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: vector_field(t, y, (S_F, get_Fsin(y, (S_F, V_max, F_min, F_max))))),
            solver=Dopri5(),
            t0=0.0, t1=t_end, dt0=0.01,
            y0=y0_forward,
            saveat=SaveAt(ts=ts)
        )
        t_plot   = sol.ts         
        y_plot   = sol.ys

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
        fig, axes = plt.subplots(4, 1, figsize=(14, 15))
        fig.suptitle('PMP shooting during singular arc for S=S*', fontsize=28)

        # Plot State variables
        axes[0].plot(t_plot, y_plot[:,0], label='(XV)', color='black', linestyle='--')
        axes[0].plot(t_plot, y_plot[:,1], label='(SV)', color='black', marker='s', linestyle='-', markevery=50)
        axes[0].plot(t_plot, y_plot[:,2], label='(PV)', color='black', linestyle='-')
        axes[0].plot(t_plot, y_plot[:,3], label='V', color='black', marker='o', linestyle='-', markevery=50)
        axes[0].set_ylabel('State (g or V)')
        axes[0].set_xlabel('Time (h)')
        axes[0].set_title('State variables')
        axes[0].grid(True)
        axes[0].legend()

        # Plot S and P concentration
        axes[1].plot(t_plot, jnp.array(y_plot[:,1]) / jnp.array(y_plot[:,3]), label='S', color='black', linestyle='-')
        axes[1].plot(t_plot, jnp.array(y_plot[:,2]) / jnp.array(y_plot[:,3]), label='P', color='black', linestyle='--')
        axes[1].set_ylabel('Concentration (g/L)')
        axes[1].set_xlabel('Time (h)')
        axes[1].set_title('S and P concentration')
        axes[1].grid(True)
        axes[1].legend()

        F_actual = jax.vmap(lambda y: get_Fsin(y, (S_F, V_max, F_min, F_max)))(y_plot)
        phi_plot = jax.vmap(get_phi)(y_plot)
        axes[2].plot(t_plot, F_actual, label='Feed Rate', color='black')
        axes[2].set_ylabel('Feed rate (L/h)')
        axes[2].set_xlabel('Time (h)')
        axes[2].set_title('Feed rate')
        axes[2].grid(True)

        axes[2].plot(t_plot, phi_plot, color='black', label='$\phi(t)$', linestyle = '--')
        axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_title('Switching Function')
        axes[2].legend()

        axes[3].plot(t_plot, y_plot[:,4], label='l1', color='black', linestyle='--')
        axes[3].plot(t_plot, y_plot[:,5], label='l2', color='black', marker = 's', linestyle='-', markevery=50)
        axes[3].plot(t_plot, y_plot[:,6], label='l3', color='black', linestyle='-')
        axes[3].plot(t_plot, y_plot[:,7], label='l4', color='black', marker = 'o', linestyle='-', markevery=50)
        axes[3].set_ylabel('Costates')
        axes[3].set_xlabel('Time (h)')
        axes[3].set_title('Adjoints variables')
        axes[3].grid(True)
        axes[3].legend()

        print("Final XV value", y_plot[-1, 0])
        print("Final SV value", jnp.exp(y_plot[-1, 1]))
        print("Final PV value", y_plot[-1, 2])
        print("Final V value", y_plot[-1, 3])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = "output_pmp_si_sing.png"
        output_path = os.path.join(script_dir, file_name)
        plt.savefig(output_path)