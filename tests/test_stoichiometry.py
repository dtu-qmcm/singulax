"""Regression guards for the carbon/redox-balanced coefficients.

The coefficients c_mu and c_pi are returned as a tuple, so swapping them is a
one-character edit that leaves the module importable and every number finite.
It shows up only as a wrong yield, several hours into a sampling run. The
corner tests below catch it immediately: under a swap the pure-ethanol corner
reads 1.018 (carbon created from nothing) instead of 0.667.
"""
import pytest

from singulax.substrate_product_inhibition import stoichiometry as st
from singulax.substrate_product_inhibition.kinetics import (
    REFERENCE, BALANCED, mu, pi, sigma,
)


def test_macro_reactions_close():
    """R1 and R2 balance on carbon and on available electrons."""
    # R1 catabolic:  CH2O -> (2/3) CH3O0.5 + (1/3) CO2
    y_eth = st.r1_yield()
    assert y_eth * st.ETHANOL.gamma == pytest.approx(st.GLUCOSE.gamma)
    assert y_eth * st.ETHANOL.C + (1 - y_eth) * st.CO2.C == pytest.approx(1.0)

    # R2 anabolic:  CH2O + a NH3 -> Yx CH1.8O0.5N0.2 + Yc CO2 + Yw H2O
    # (NH3 and CO2 both carry zero available electrons)
    y_bio = st.r2_yield()
    assert y_bio * st.BIOMASS.gamma == pytest.approx(st.GLUCOSE.gamma)
    assert y_bio <= 1.0          # the rest of the carbon leaves as CO2


def test_coefficient_values():
    """Pin the derived numbers so a formula change is noticed."""
    assert st.C_MU == pytest.approx(1.2802, abs=1e-4)
    assert st.C_PI == pytest.approx(1.9553, abs=1e-4)
    assert st.GAY_LUSSAC == pytest.approx(0.5114, abs=1e-4)


def test_corners_detect_a_swapped_pair():
    """c_mu multiplies mu and c_pi multiplies pi -- not the other way round.

    Pure ethanol must account for exactly r1_yield of the carbon (the other
    third leaves as CO2); pure growth for exactly r2_yield. Swapping the pair
    gives 1.018 and 0.624 instead.
    """
    pure_ethanol = st.carbon_accounted(0.0, 1.0, st.C_PI)
    pure_growth = st.carbon_accounted(1.0, 0.0, st.C_MU)
    assert pure_ethanol == pytest.approx(st.r1_yield())
    assert pure_growth == pytest.approx(st.r2_yield())


def test_balanced_never_creates_carbon():
    """No mix of the two reactions can account for more carbon than went in."""
    for mu_val in [0.0, 0.05, 0.2, 0.4]:
        for pi_val in [0.0, 0.3, 1.0, 2.0]:
            s = st.C_MU * mu_val + st.C_PI * pi_val
            if s == 0.0:
                continue
            assert st.carbon_accounted(mu_val, pi_val, s) <= 1.0
            assert pi_val / s <= st.GAY_LUSSAC + 1e-12


def test_reference_model_does_create_carbon():
    """The appendix claim: the Shin & Lim coefficients are not balanced.

    Charging substrate for growth only (c_pi = 0) means ethanol is free, so at
    high product -- where the fed-batch optimum actually operates -- the
    products hold more carbon than the glucose supplied.
    """
    S, P = 20.0, 80.0
    m, p = mu(S, P), pi(S, P)
    s_ref = sigma(S, P, REFERENCE.c_mu, REFERENCE.c_pi)
    s_bal = sigma(S, P, BALANCED.c_mu, BALANCED.c_pi)

    assert st.carbon_accounted(m, p, s_ref) > 1.0
    assert p / s_ref > st.GAY_LUSSAC
    # the balanced model, same rates, stays physical
    assert st.carbon_accounted(m, p, s_bal) <= 1.0
    assert p / s_bal <= st.GAY_LUSSAC


def test_kinetics_wires_the_coefficients_the_right_way_round():
    """The swap could also happen at the KineticParams boundary."""
    assert BALANCED.c_mu == pytest.approx(st.C_MU)
    assert BALANCED.c_pi == pytest.approx(st.C_PI)
    assert REFERENCE.c_mu == pytest.approx(st.REFERENCE_C_MU)
    assert REFERENCE.c_pi == pytest.approx(st.REFERENCE_C_PI)

    # sigma really is c_mu*mu + c_pi*pi at a representative state
    S, P = 20.0, 40.0
    expected = st.C_MU * mu(S, P) + st.C_PI * pi(S, P)
    got = sigma(S, P, BALANCED.c_mu, BALANCED.c_pi)
    assert got == pytest.approx(expected)
