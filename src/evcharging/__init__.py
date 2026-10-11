"""Robust EV charging-station location under rare-event disruptions."""
from .data import Case, load_case, scale_outage
from .model import Design, solve_design
from .simulate import (
    DEFAULT_PROXY_CORR,
    Draws,
    Estimate,
    OutageSimulator,
    cv_coefficients,
    cv_reliability,
    hoeffding_sample_size,
    indicator_corr2,
    mc_reliability,
    objective_samples,
    running_estimate,
)

__version__ = "1.0.0"

__all__ = [
    "Case", "load_case", "scale_outage",
    "Design", "solve_design",
    "DEFAULT_PROXY_CORR", "Draws", "Estimate", "OutageSimulator",
    "cv_coefficients", "cv_reliability", "mc_reliability",
    "objective_samples", "running_estimate",
    "indicator_corr2", "hoeffding_sample_size",
]
