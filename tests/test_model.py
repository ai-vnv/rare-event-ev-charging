"""Checks on the data, the optimisation model and the designs it returns."""
import pytest

from evcharging import load_case, scale_outage, solve_design

MILLION = 1e6


@pytest.fixture(scope="module")
def case():
    return load_case()


@pytest.fixture(scope="module")
def robust(case):
    return solve_design(case, case.reliability)


def test_case_dimensions(case):
    assert len(case.J) == 11 and len(case.I) == 31 and case.K == ["motorcycle", "car"]
    assert case.total_demand() == {"motorcycle": 515, "car": 116}
    assert all(0.976 < p < 0.980 for p in case.reliability.values())      # thresholds are about mu + 2 sigma


def test_robust_design_is_feasible(case, robust):
    d = robust
    for i in case.I:
        for k in case.K:
            assert sum(d.v[i, j, k] for j in case.J) == case.w[i, k]                      # (6) all demand served
        assert sum(d.y[i, j] for j in case.J) >= 1                                       # (10)
    for j in case.J:
        minutes = sum(case.t[k] * d.v[i, j, k] for i in case.I for k in case.K)
        assert minutes <= case.c * d.u[j] + 1e-6                                         # (7) capacity
        assert d.u[j] <= case.q * d.x[j]                                                 # (8)
    for (i, j), assigned in d.y.items():
        if assigned:
            assert d.x[j] == 1 and all(case.d[i, j, k] <= case.dmax for k in case.K)     # (5), (9)
    assert all(d.x[j] == 1 for j in case.existing)


def test_robust_optimum(case, robust):
    value = robust.expected(case.reliability)["objective"]
    assert value == pytest.approx(robust.model_objective, rel=1e-9)
    assert value / MILLION == pytest.approx(126.27, abs=0.01)
    assert len(robust.opened) == 3


def test_non_robust_range(case, robust):
    p = case.reliability
    ones = {j: 1.0 for j in case.J}
    worst = solve_design(case, ones, tie_break="worst", reference=p)
    best = solve_design(case, ones, tie_break="best", reference=p)
    assert worst.model_objective == pytest.approx(best.model_objective, abs=1.0)         # same non-robust optimum
    lo, hi = worst.expected(p)["objective"], best.expected(p)["objective"]
    assert lo / MILLION == pytest.approx(123.60, abs=0.01)
    assert hi / MILLION == pytest.approx(126.00, abs=0.01)
    assert lo <= hi <= robust.expected(p)["objective"]                                   # robust is optimal under p
    assert len(worst.opened) == 2
    assert worst.mean_travel_time(case) > best.mean_travel_time(case) > robust.mean_travel_time(case)
    assert robust.backup_share(case) > worst.backup_share(case)


def test_scale_outage_keeps_probabilities_valid(case):
    p = case.reliability
    for factor in (0.5, 2.0, 100.0):
        scaled = scale_outage(p, factor)
        assert all(0.0 <= v <= 1.0 for v in scaled.values())
    assert scale_outage(p, 2.0)["Tandes"] == pytest.approx(1 - 2 * (1 - p["Tandes"]))


def test_reliability_below_service_level_is_infeasible(case):
    # 0.95 * p_j is below the service level of 0.95, so constraint (13) cannot hold.
    with pytest.raises(RuntimeError):
        solve_design(case, {j: 0.95 * p for j, p in case.reliability.items()})


def test_tie_break_arguments(case):
    with pytest.raises(ValueError):
        solve_design(case, case.reliability, tie_break="worst")
    with pytest.raises(ValueError):
        solve_design(case, case.reliability, tie_break="random", reference=case.reliability)
