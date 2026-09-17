"""Specific rates -- growth (mu), substrate uptake (sigma) and product formation (pi)
-- for the substrate+product inhibition case. From Shin & Lim, "Fed-Batch Cultures",
example 13.E.3 (ethanol from glucose by S. cerevisiae). Each is a function of the
substrate S and product P concentrations.
"""
import jax
from typing import NamedTuple
from jaxtyping import Array, Float, ScalarLike
import jax.numpy as jnp


class KineticParams(NamedTuple):
    """The seven kinetic constants, in the order used by the sampling scripts.

    Defaults are the Shin & Lim values, so `KineticParams()` is the literature
    model and `KineticParams(*row)` builds one from a row of a sampled design.
    It is a NamedTuple, hence a JAX pytree: it can be passed through `jit`,
    `vmap` and diffrax `args` without special handling.
    """
    mu_max: ScalarLike = 0.408      # max specific growth rate [1/h]
    k_p: ScalarLike = 16.0          # ethanol that halves growth [g/L]
    k_s: ScalarLike = 0.22          # half-saturation glucose, growth [g/L]
    Y_xs: ScalarLike = 0.1          # biomass yield [g cell/g glucose]
    pi_max: ScalarLike = 1.0        # max specific productivity [g EtOH/(g cell h)]
    k_p_pi: ScalarLike = 71.5       # ethanol that halves productivity [g/L]
    k_s_pi: ScalarLike = 0.44       # half-saturation glucose, production [g/L]


DEFAULT_PARAMS = KineticParams()


# SUBSTRATE-PRODUCT INHIBITION
def mu(S: ScalarLike, P: ScalarLike, mu_max=0.408, K_p=16, K_s=0.22) -> ScalarLike:
    """Growth rate function.

    Taken from textbook "Fed Batch Cultures" by Shin and Lim (specifically example 13.E.3 "ethanol from glucose by s. cerevisiae").

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.

    Returns:
        float: Specific growth rate of the microorganism.
    """
    return mu_max * S / ((1 + P / K_p) * (K_s + S))


def sigma(S: ScalarLike, P: ScalarLike, Y_xs=0.1, mu_max=0.408, K_p=16, K_s=0.22) -> ScalarLike:
    """Substrate consumption function.

    Taken from textbook "Fed Batch Cultures" by Shin and Lim (specifically example 13.E.3 "ethanol from glucose by s. cerevisiae").

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.

    Returns:
        float: Specific substrate consumption rate.
    """
    return mu(S, P, mu_max, K_p, K_s) / Y_xs


def pi(S: ScalarLike, P: ScalarLike, pi_max=1.0, K_p=71.5, K_s=0.44) -> ScalarLike:
    """Specific productivity function.

    Taken from textbook "Fed Batch Cultures" by Shin and Lim (specifically example 13.E.3 "ethanol from glucose by s. cerevisiae").

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.

    Returns:
        float: Specific productivity rate.
    """
    return pi_max * S / ((1 + P / K_p) * (K_s + S))
