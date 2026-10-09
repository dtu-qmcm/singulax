import jax
import jax.numpy as jnp
import diffrax
from singulax.substrate_product_inhibition.odes import vector_field, get_phi, vf_states, params_from
from singulax.substrate_product_inhibition.Fsin import get_Fsin_unconstrained


def switch_times_from(raw, tf):
    """The ONE definition of how the shooting unknowns become ordered switch times.

    Levenberg-Marquardt is unconstrained, so it proposes negative, unordered and
    out-of-horizon times; this maps whatever it proposes onto
    0 < t1 <= ... <= t_M <= tf.

    The t_M <= tf ceiling is load-bearing, not cosmetic: the gaps accumulate past
    the first clip's bound (raw [30, 20, 35] -> cumsum [30, 30, 45]), and a t_M
    beyond tf makes the batch integrate backwards, which diffrax rejects outright.
    """
    raw = jnp.clip(jnp.asarray(raw), 1e-2, tf - 1e-2)
    gaps = jnp.maximum(jnp.diff(jnp.concatenate([jnp.array([0.0]), raw])), 0.0)
    return jnp.clip(jnp.cumsum(gaps), 1e-2, tf)


def solve_and_get_residuals_ms(guess, args):
    """Multiple-shooting residuals.

    guess = [ node_0, ..., node_{M}  (each 8 = [x(4), lambda(4)]),  t1, ..., t_M ]
            with M = num_segments, so n_nodes = M+1 nodes at times [0, t1, ..., t_M].
    args  = (num_segments, S_F, V_max, F_min, F_max, tf, x0, arc_u, params)
            arc_u[i] = control on segment i:  a float F for a bang,  None for the singular arc.

    The batch [t_M, tf] is NOT integrated as a segment (its costate blows up); instead it enters as
    lambda(t_M) = d PV(tf)/d x(t_M).

    Residuals (all in physical coordinates, so no l_to_orig):
      - continuity defects on each segment
      - x(0) = x0
      - lambda_{XV,SV,PV}(t_M) = grad PV(tf)    (transversality on the free states)
      - V(t_M) = V_max                          (tank full -> singular exit; replaces the lambda_V
                                                 transversality, since V is pinned at its bound)
      - phi = 0 at each interior switch node
      - phidot = 0 at the singular-entry node   (tangential entry)
    """
    num_segments, S_F, V_max, F_min, F_max, tf, x0, arc_u = args[:8]
    params = params_from(args, 8)
    n_nodes = num_segments + 1
    nodes = guess[:n_nodes * 8].reshape(n_nodes, 8)
    switch_times = switch_times_from(guess[n_nodes * 8:], tf)            # [t1, ..., t_M]
    shooting_nodes = jnp.concatenate([jnp.array([0.0]), switch_times])   # [0, t1, ..., t_M]

    sc = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmin=1e-12)
    vf_singular = lambda t, y, a: vector_field(
        t, y, (S_F, get_Fsin_unconstrained(y, (S_F, V_max, F_min, F_max, params)), params))

    # ---- continuity defects on the bang / bang / singular segments ----
    defects = []
    for i in range(num_segments):
        u = arc_u[i]
        vfield = vf_singular if u is None else vector_field
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vfield), diffrax.Tsit5(),
            t0=shooting_nodes[i], t1=shooting_nodes[i + 1], y0=nodes[i], dt0=0.01,
            args=(S_F, u, params),                                       # tuple, not a bare scalar
            saveat=diffrax.SaveAt(t1=True), stepsize_controller=sc,
            max_steps=100000, throw=False, adjoint=diffrax.ForwardMode())
        y_end = sol.ys[-1]
        y_end = jnp.where(jnp.any(~jnp.isfinite(y_end)), jnp.full(8, 1e6), y_end)
        defects.append(y_end - nodes[i + 1])                             # match the next node
    defects = jnp.concatenate(defects)

    # ---- batch as a terminal cost:  lambda(t_M) = d PV(tf) / d x(t_M) ----
    x_end, lam_end = nodes[-1][:4], nodes[-1][4:]
    t_M = shooting_nodes[-1]

    def batch_PV(x_sw):
        s = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, x, a: vf_states(t, x, (S_F, 0.0, params))), diffrax.Tsit5(),
            t0=t_M, t1=tf, y0=x_sw, dt0=0.01, saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=sc, max_steps=100000, throw=False, adjoint=diffrax.DirectAdjoint())
        return s.ys[-1][2]                                               # PV(tf)
    grad_PV = jax.grad(batch_PV)(x_end)
    terminal = lam_end[:3] - grad_PV[:3]                                 # transversality on free states XV, SV, PV
    v_cond = jnp.array([x_end[3] - V_max])                               # V(t_M) = V_max pins the singular exit

    # ---- boundary + switch conditions ----
    bc_start = nodes[0][:4] - x0                                         # x(0) = x0
    phi = jnp.array([get_phi(nodes[k], S_F) for k in range(1, n_nodes - 1)])  # phi = 0 at interior switches
    yd_entry = vector_field(shooting_nodes[-1 - 1], nodes[-2], (S_F, 0.0, params))   # phidot at the singular-entry node
    phidot_entry = jnp.array([get_phi(yd_entry, S_F)])   # tangential entry: phidot = 0

    return jnp.concatenate([defects, bc_start, terminal, v_cond, phi, phidot_entry])
