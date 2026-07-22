"""Generic Stage-1 direct-collocation machinery for control-affine fed-batch OCPs.

The state RHS is a REQUIRED parameter (`CollocParams.f`, signature `f(x, F, S_F)`): each caller
passes the dynamics it wants. Use the log(SV) RHS (`fedopt.odes.states_ode`) for cases where the
substrate stays away from 0 (e.g. the pure singular arc), and a PHYSICAL-SV RHS when the substrate
exhausts (SV -> 0, e.g. singular-arc+batch) -- in log(SV) the forward-Euler NLP can "consume more
substrate than exists" (SV = exp() is never negative) and returns an infeasible seed; physical SV
+ an explicit SV >= 0 bound (set by the caller) blocks that.

jitted call sites should CLOSE OVER `p` (its int fields are used as array shapes):
    p = CollocParams(N=N, nx=nx, h=h, nX=nX, nz=nz, S_F=S_F, f=states_ode)
    obj_jit  = jax.jit(lambda z: objective(z, p))
    grad_jit = jax.jit(jax.grad(lambda z: objective(z, p)))
    con_jit  = jax.jit(lambda z: constraints(z, p))
    jac      = lambda z: jac_defects(z, p)          # not jitted (returns a COO matrix)
"""
from typing import NamedTuple, Callable
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import coo_array


class CollocParams(NamedTuple):
    N: int          # number of intervals
    nx: int         # number of states
    h: float        # step size tf / N
    nX: int         # (N+1)*nx  -- length of the flattened state block
    nz: int         # nX + N    -- length of the full decision vector
    S_F: float      # feed substrate concentration
    f: Callable     # state RHS f(x, F, S_F): log(SV) or physical-SV (see module docstring)


def unpack(z, p):
    X = z[:p.nX].reshape(p.N + 1, p.nx)
    U = z[p.nX:]
    return X, U


def objective(z, p):
    X, _ = unpack(z, p)
    return -X[-1, 2]                                 # minimise -PV(tf) == maximise PV(tf)


# equality constraints: forward-Euler defects d_k = x_{k+1} - x_k - h f(x_k, F_k)
def constraints(z, p):
    X, U = unpack(z, p)
    f = jax.vmap(p.f, in_axes=(0, 0, None))(X[:-1], U, p.S_F)          # (N, nx)
    return (X[1:] - X[:-1] - p.h * f).ravel()                         # (nx*N,)


@lru_cache(maxsize=None)
def _make_blocks(states_ode_fn):
    """Per-step Jacobian blocks d(defect_k)/d(x_k, x_{k+1}, F_k), built once per distinct RHS
    (lru_cache keyed by the function object -> no re-jit across jac_defects calls)."""
    def _single_defect(xk, xk1, uk, h, S_F):
        return xk1 - xk - h * states_ode_fn(xk, uk, S_F)
    return jax.jit(jax.vmap(jax.jacrev(_single_defect, argnums=(0, 1, 2)),
                            in_axes=(0, 0, 0, None, None)))


def jac_defects(z, p):
    X, U = unpack(jnp.asarray(z), p)
    A, B, C = _make_blocks(p.f)(X[:-1], X[1:], U, p.h, p.S_F)

    # static COO sparsity pattern: each defect d_k touches only x_k, x_{k+1}, F_k
    k = np.arange(p.N)[:, None, None]
    i = np.arange(p.nx)[None, :, None]
    j = np.arange(p.nx)[None, None, :]
    rowsAB = np.broadcast_to(p.nx * k + i, (p.N, p.nx, p.nx)).ravel()
    colsA = np.broadcast_to(p.nx * k + j, (p.N, p.nx, p.nx)).ravel()             # cols of x_k
    colsB = np.broadcast_to(p.nx * (k + 1) + j, (p.N, p.nx, p.nx)).ravel()       # cols of x_{k+1}
    rowsC = (p.nx * np.arange(p.N)[:, None] + np.arange(p.nx)[None, :]).ravel()
    colsC = np.broadcast_to(p.nX + np.arange(p.N)[:, None], (p.N, p.nx)).ravel()  # col of F_k
    rows = np.concatenate([rowsAB, rowsAB, rowsC])
    cols = np.concatenate([colsA, colsB, colsC])

    data = np.concatenate([np.asarray(A).ravel(), np.asarray(B).ravel(), np.asarray(C).ravel()])
    return coo_array((data, (rows, cols)), shape=(p.nx * p.N, p.nz))
