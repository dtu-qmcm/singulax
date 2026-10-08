"""Guards on the two places where a silent default used to decide the physics.

Both were the same shape: a value that was correct at the time it was written,
substituted without a word when a caller left it out. One switched hybrid_spi.py
to the balanced model when DEFAULT_PARAMS moved; the other would corrupt every
Stage-2 residual if the feed concentration ever left 150 g/L. Neither failed
loudly, so neither was noticed. These tests keep the defaults out.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from singulax.substrate_product_inhibition.odes import (
    get_phi, params_from, vector_field, vf_states,
)
from singulax.substrate_product_inhibition.kinetics import REFERENCE, BALANCED

X = jnp.array([1.0, 50.0, 10.0, 5.0])                 # [XV, SV, PV, V]
Y = jnp.concatenate([X, jnp.array([1.0, 0.5, 1.0, -70.0])])   # [x, lambda]
S_F = 150.0


def test_params_from_rejects_a_short_args_tuple():
    """No silent default: a tuple with no KineticParams must raise, not guess."""
    with pytest.raises(ValueError, match="KineticParams must be at index"):
        params_from((S_F, 0.5), 2)


def test_rhs_rejects_a_short_args_tuple():
    """The guard has to bite at the call sites, not just in the helper."""
    for fn, args in [(vf_states, (S_F, 0.5)), (vector_field, (S_F, 0.5))]:
        with pytest.raises(ValueError, match="KineticParams must be at index"):
            fn(0.0, Y if fn is vector_field else X, args)


def test_params_from_returns_what_was_passed():
    assert params_from((S_F, 0.5, REFERENCE), 2) is REFERENCE
    assert params_from((S_F, 10.0, 0.0, 2.0, BALANCED), 4) is BALANCED


def test_the_two_models_give_different_dynamics():
    """If these ever agree, the parameter set is not reaching the rate laws."""
    ref = vf_states(0.0, X, (S_F, 0.5, REFERENCE))
    bal = vf_states(0.0, X, (S_F, 0.5, BALANCED))
    assert ref[0] == pytest.approx(bal[0])        # growth is identical...
    assert ref[2] == pytest.approx(bal[2])        # ...so is production
    assert ref[1] != pytest.approx(bal[1])        # only substrate demand differs


def test_get_phi_uses_the_S_F_it_is_given():
    """phi = S_F*lambda_SV + lambda_V. S_F used to be hardcoded to 150.0, so a
    run at any other feed concentration silently got the wrong switching
    function -- and phi = 0 is what the shooting residual pins at every switch."""
    lam_SV, lam_V = float(Y[5]), float(Y[7])
    assert float(get_phi(Y, 150.0)) == pytest.approx(150.0 * lam_SV + lam_V)
    assert float(get_phi(Y, 300.0)) == pytest.approx(300.0 * lam_SV + lam_V)
    # the two must differ, or S_F is being ignored again
    assert float(get_phi(Y, 300.0)) != pytest.approx(float(get_phi(Y, 150.0)))
