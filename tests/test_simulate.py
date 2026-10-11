"""Checks on the outage simulator and the Monte Carlo / control-variate estimators."""
import numpy as np
import pytest

from evcharging import (
    OutageSimulator, cv_coefficients, cv_reliability, hoeffding_sample_size, indicator_corr2, load_case,
    mc_reliability, objective_samples, running_estimate, solve_design,
)


@pytest.fixture(scope="module")
def case():
    return load_case()


def test_estimators_agree_with_exact_reliability(case):
    draws = OutageSimulator(case).sample(20_000, seed=1)
    exact = np.array([case.reliability[j] for j in case.J])
    for estimates in (mc_reliability(draws), cv_reliability(draws)):
        mean = np.array([e.mean for e in estimates])
        se = np.array([e.se for e in estimates])
        assert np.all(np.abs(mean - exact) < 4.5 * se)


def test_control_variate_reduces_standard_error(case):
    draws = OutageSimulator(case, proxy_corr=0.999).sample(20_000, seed=2)
    mc = np.array([e.se for e in mc_reliability(draws)])
    cv = np.array([e.se for e in cv_reliability(draws)])
    ratio = mc / cv
    assert np.all(ratio > 2.0)
    theory = 1 / np.sqrt(1 - indicator_corr2(2.0, 0.999))
    assert ratio.mean() == pytest.approx(theory, rel=0.15)


def test_perfect_proxy_gives_exact_answer(case):
    draws = OutageSimulator(case, proxy_corr=1.0).sample(500, seed=3)
    exact = [case.reliability[j] for j in case.J]
    estimates = cv_reliability(draws)
    assert [e.mean for e in estimates] == pytest.approx(exact, abs=1e-12)
    assert max(e.se for e in estimates) < 1e-12


def test_indicator_correlation_limits():
    assert indicator_corr2(2.0, 1.0) == 1.0
    assert indicator_corr2(2.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    values = [indicator_corr2(2.0, r) for r in (0.5, 0.9, 0.99, 0.999)]
    assert values == sorted(values)


def test_simulated_objective_matches_expectation(case):
    design = solve_design(case, case.reliability)
    sim = OutageSimulator(case)
    beta = cv_coefficients(sim.sample(10_000, seed=4))
    samples = objective_samples(design, sim.sample(20_000, seed=5), case.J, beta)
    exact = design.expected(case.reliability)["objective"]
    for values in samples.values():
        se = values.std(ddof=1) / np.sqrt(len(values))
        assert abs(values.mean() - exact) < 4.5 * se
    assert samples["cv"].std() < samples["mc"].std()


def test_running_estimate():
    x = np.arange(1.0, 101.0)
    run = running_estimate(x, [10, 100])
    assert run["mean"] == pytest.approx([5.5, 50.5])
    assert run["se"][-1] == pytest.approx(x.std(ddof=1) / 10)


def test_hoeffding_sample_size():
    assert hoeffding_sample_size(1e-4, 0.05) == pytest.approx(1.844e8, rel=1e-3)


def test_proxy_corr_is_validated(case):
    with pytest.raises(ValueError):
        OutageSimulator(case, proxy_corr=1.5)
