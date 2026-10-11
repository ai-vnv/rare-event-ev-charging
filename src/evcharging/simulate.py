"""Outage simulation with Monte Carlo and control-variate estimators.

Station ``j`` has an outage on a simulated day when its electricity demand
``Z_j ~ N(mu_j, sigma_j^2)`` exceeds the threshold ``z_j`` (Table I of the paper),
so the availability indicator is ``I_j = 1{Z_j <= z_j}`` and ``p_j = E[I_j]``.

The control variate is the same indicator computed from a cheap standard-normal
proxy ``X_j`` of the standardised demand, ``C_j = 1{X_j <= kappa_j}``, whose mean
``Phi(kappa_j)`` is known.  ``proxy_corr`` is the correlation between ``X_j`` and the
standardised demand; it is the "partial knowledge" in Fig. 1 of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import multivariate_normal, norm

from .data import Case
from .model import Design

#: Proxy correlation used for the figures.  It is a modelling assumption: 0.999 is a
#: round number at which the control variate saves about a factor of ten in sample
#: size.  The saving depends strongly on it; see ``results/cv_sensitivity.csv``.
DEFAULT_PROXY_CORR = 0.999


@dataclass
class Draws:
    """``n`` simulated days for all stations."""

    available: np.ndarray       # [n, J] availability indicators I_jl
    control: np.ndarray         # [n, J] control indicators C_jl
    control_mean: np.ndarray    # [J]    known mean of the control, Phi(kappa_j)


@dataclass
class Estimate:
    mean: float
    se: float
    n: int

    def ci(self, level: float = 0.95) -> tuple:
        zq = norm.ppf(0.5 + level / 2)
        return self.mean - zq * self.se, self.mean + zq * self.se


class OutageSimulator:
    """Draws station availabilities together with their control variates."""

    def __init__(self, case: Case, proxy_corr: float = DEFAULT_PROXY_CORR):
        if not 0.0 <= proxy_corr <= 1.0:
            raise ValueError("proxy_corr must lie in [0, 1]")
        self.case = case
        self.proxy_corr = float(proxy_corr)
        self.kappa = np.array([case.kappa[j] for j in case.J])

    def sample(self, n: int, seed: Optional[int] = None) -> Draws:
        rng = np.random.default_rng(seed)
        J = len(self.case.J)
        x = rng.standard_normal((n, J))                      # proxy, observed cheaply
        noise = rng.standard_normal((n, J))
        rho = self.proxy_corr
        z_std = rho * x + np.sqrt(1.0 - rho**2) * noise      # standardised demand
        available = (z_std <= self.kappa).astype(float)
        control = (x <= self.kappa).astype(float)
        return Draws(available, control, norm.cdf(self.kappa))


# ----------------------------------------------------------------------------------
# Estimators of the station reliability p_j
# ----------------------------------------------------------------------------------

def mc_reliability(draws: Draws) -> List[Estimate]:
    """Plain Monte Carlo estimate of ``p_j`` for every station."""
    n = draws.available.shape[0]
    mean = draws.available.mean(axis=0)
    se = draws.available.std(axis=0, ddof=1) / np.sqrt(n)
    return [Estimate(float(m), float(s), n) for m, s in zip(mean, se)]


def cv_coefficients(draws: Draws) -> np.ndarray:
    """Variance-minimising coefficient ``Cov(I_j, C_j) / Var(C_j)`` for every station.

    This is ``-pi_j`` in equation (15) of the paper.
    """
    i_c = draws.available - draws.available.mean(axis=0)
    c_c = draws.control - draws.control.mean(axis=0)
    var_c = (c_c**2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(var_c > 0, (i_c * c_c).sum(axis=0) / var_c, 0.0)
    return beta


def cv_reliability(draws: Draws, beta: Optional[np.ndarray] = None) -> List[Estimate]:
    """Control-variate estimate of ``p_j`` for every station (equation (14))."""
    beta = cv_coefficients(draws) if beta is None else beta
    adjusted = draws.available - beta * (draws.control - draws.control_mean)
    n = adjusted.shape[0]
    return [Estimate(float(m), float(s), n)
            for m, s in zip(adjusted.mean(axis=0), adjusted.std(axis=0, ddof=1) / np.sqrt(n))]


# ----------------------------------------------------------------------------------
# Simulated objective of a fixed design
# ----------------------------------------------------------------------------------

def objective_samples(design: Design, draws: Draws, stations: Sequence[str],
                      beta: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-day objective values of a design, raw (MC) and control-variate adjusted (CV).

    Objective (3) is linear in the availability indicators, so the control variate
    of station ``j`` enters with weight ``(a_j + b_j) * beta_j``.
    """
    stations = list(stations)
    beta = cv_coefficients(draws) if beta is None else beta
    weight = np.array([design.a[j] + design.b[j] for j in stations]) * beta
    mc = design.objective_per_scenario(draws.available, stations)
    cv = mc - (draws.control - draws.control_mean) @ weight
    return {"mc": mc, "cv": cv}


def running_estimate(samples: np.ndarray, sizes: Sequence[int]) -> Dict[str, np.ndarray]:
    """Mean, standard error and 95% interval of the first ``n`` samples for each ``n``."""
    sizes = np.asarray(sizes, dtype=int)
    csum = np.cumsum(samples)
    csq = np.cumsum(samples**2)
    mean = csum[sizes - 1] / sizes
    var = np.maximum(csq[sizes - 1] / sizes - mean**2, 0.0) * sizes / np.maximum(sizes - 1, 1)
    se = np.sqrt(var / sizes)
    zq = norm.ppf(0.975)
    return {"n": sizes, "mean": mean, "se": se, "lo": mean - zq * se, "hi": mean + zq * se}


# ----------------------------------------------------------------------------------
# Theory helpers
# ----------------------------------------------------------------------------------

def indicator_corr2(kappa: float, proxy_corr: float) -> float:
    """Squared correlation between ``1{Z <= kappa}`` and ``1{X <= kappa}``.

    ``(Z, X)`` is standard bivariate normal with correlation ``proxy_corr``.  The
    control variate multiplies the Monte Carlo variance by ``1 - indicator_corr2``.
    """
    p = norm.cdf(kappa)
    if proxy_corr >= 1.0:
        return 1.0
    joint = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, proxy_corr], [proxy_corr, 1.0]]).cdf([kappa, kappa])
    return float((joint - p * p) ** 2 / (p * (1 - p)) ** 2)


def hoeffding_sample_size(delta: float, alpha: float) -> float:
    """Smallest ``n`` with ``P(|p_hat - p| > delta) <= alpha`` by Hoeffding's inequality."""
    return float(np.log(2.0 / alpha) / (2.0 * delta**2))
