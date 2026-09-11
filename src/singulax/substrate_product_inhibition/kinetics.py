"""Specific rates -- growth (mu), substrate uptake (sigma) and product formation (pi)
-- for the substrate+product inhibition case. From Shin & Lim, "Fed-Batch Cultures",
example 13.E.3 (ethanol from glucose by S. cerevisiae). Each is a function of the
substrate S and product P concentrations.
"""
import jax
from jaxtyping import Array, Float, ScalarLike
import jax.numpy as jnp

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


def sigma(S: ScalarLike, P: ScalarLike, Y_xs=0.1) -> ScalarLike:
    """Substrate consumption function.

    Taken from textbook "Fed Batch Cultures" by Shin and Lim (specifically example 13.E.3 "ethanol from glucose by s. cerevisiae").

    Args:
        S (float): Substrate concentration.
        P (float): Product concentration.

    Returns:
        float: Specific substrate consumption rate.
    """
    return mu(S, P) / Y_xs


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
