from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, norm, qmc, uniform

CV = 0.10                                # relative sd assumed for every parameter


def _lognormal(mu_log, sigma_log):
    """Lognormal from the mean and sd of ln(X)."""
    return lognorm(s=sigma_log, scale=np.exp(mu_log))


def _lognormal_mean(mean, cv=CV):
    """Lognormal with this mean and coefficient of variation."""
    sigma_log = np.sqrt(np.log(1 + cv**2))
    return _lognormal(np.log(mean) - sigma_log**2 / 2, sigma_log)


# Priors -- means are the Shin & Lim example 13.E.3 values, sd = 10% of the mean
mu_max = _lognormal_mean(0.408)          # max specific growth rate [1/h]
k_p = _lognormal_mean(16.0)              # ethanol that halves growth [g/L]
k_s = _lognormal_mean(0.22)              # half-saturation glucose, growth [g/L]
Y_xs = _lognormal_mean(0.1)              # biomass yield [g cell/g glucose]; the paper's sigma = mu/Y_xs = 10*mu
pi_max = _lognormal_mean(1.0)            # max specific productivity [g EtOH/(g cell h)]
k_p_pi = _lognormal_mean(71.5)           # ethanol that halves productivity [g/L]
k_s_pi = _lognormal_mean(0.44)           # half-saturation glucose, production [g/L]

NAMES = ['mu_max', 'k_p', 'k_s', 'Y_xs', 'pi_max', 'k_p_pi', 'k_s_pi']
DISTS = [mu_max, k_p, k_s, Y_xs, pi_max, k_p_pi, k_s_pi]
M = len(DISTS)                           # number of input parameters


def sample_X(n, seed, method='lhs'):
    """Draw n scenarios -> array of shape (n, M) in real parameter units.

    method='lhs' gives a Latin hypercube design, 'mc' plain random sampling.
    """
    if method == 'lhs':
        u = qmc.LatinHypercube(d=M, seed=seed).random(n=n)
    elif method == 'mc':
        u = np.random.default_rng(seed).random((n, M))
    else:
        raise ValueError(f"unknown method {method!r}, expected 'lhs' or 'mc'")

    return np.stack([d.ppf(u[:, j]) for j, d in enumerate(DISTS)], axis=1)


N = 2**10 # Monte Carlo samples
SEED = 1234

X = sample_X(N, SEED)
Y = 



