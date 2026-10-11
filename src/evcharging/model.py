"""The charging-station location model of the paper as a mixed-integer program.

Objective (2) and constraints (4)-(13), written with Pyomo and solved with HiGHS.
Passing ``p_j = 1`` for every station gives the Non-Robust Model; passing the
estimated reliabilities gives the Robust Model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from .data import Case


@dataclass
class Design:
    """A solved network design."""

    x: Dict[str, int]                         # station opened
    u: Dict[str, int]                         # connectors installed
    y: Dict[Tuple[str, str], int]             # demand node assigned to station (may include unused links)
    v: Dict[Tuple[str, str, str], int]        # vehicles of type k from i charged at j
    cost: float                               # daily operating + investment cost
    a: Dict[str, float]                       # a_j: revenue earned at j on a day without outage
    b: Dict[str, float]                       # b_j: penalty paid at j on a day with outage
    p_assumed: Dict[str, float]               # reliabilities the optimiser was given
    model_objective: float                    # optimal value of (2) under ``p_assumed``

    @property
    def opened(self) -> List[str]:
        return [j for j, val in self.x.items() if val]

    def expected(self, p: Dict[str, float]) -> Dict[str, float]:
        """Expected daily revenue, penalty, cost and objective under reliabilities ``p``."""
        revenue = sum(self.a[j] * p[j] for j in self.a)
        penalty = sum(self.b[j] * (1.0 - p[j]) for j in self.b)
        return {"revenue": revenue, "penalty": penalty, "cost": self.cost,
                "objective": revenue - penalty - self.cost}

    def objective_per_scenario(self, available: np.ndarray, stations: List[str]) -> np.ndarray:
        """Daily objective for each simulated day.

        ``available[l, j] = 1`` if station ``stations[j]`` has no outage on day ``l``.
        This is the summand of objective (3).
        """
        a = np.array([self.a[j] for j in stations])
        b = np.array([self.b[j] for j in stations])
        return available @ a - (1.0 - available) @ b - self.cost

    def flow(self, case: Case, i: str, j: str) -> int:
        """Vehicles per day from demand node ``i`` charged at station ``j``."""
        return sum(self.v[i, j, k] for k in case.K)

    def backup_share(self, case: Case) -> float:
        """Share of demand nodes with two or more opened stations within ``dmax``.

        This measures coverage redundancy from the opened stations alone.  It does
        not use ``y``, which is free whenever it is not needed by a constraint.
        """
        def in_reach(i: str, j: str) -> bool:
            return all(case.d[i, j, k] <= case.dmax for k in case.K)
        n = sum(1 for i in case.I if sum(in_reach(i, j) for j in self.opened) >= 2)
        return n / len(case.I)

    def mean_travel_time(self, case: Case) -> float:
        """Average travel time to the charging station, in minutes per vehicle."""
        minutes = sum(case.d[i, j, k] * n for (i, j, k), n in self.v.items())
        return minutes / sum(self.v.values())


def _objective(m: pyo.ConcreteModel, case: Case, p: Dict[str, float]):
    """Expected daily profit (2) of the decisions in ``m`` under reliabilities ``p``."""
    I, J, K = case.I, case.J, case.K
    revenue = sum(case.r * case.e[k] * p[j] * m.v[i, j, k] for i in I for j in J for k in K)
    penalty = sum(case.s * case.d[i, j, k] * (1.0 - p[j]) * m.v[i, j, k] for i in I for j in J for k in K)
    cost = sum(case.g * m.u[j] + case.h[j] * m.x[j] for j in J)
    return revenue - penalty - cost


def _build(case: Case, p: Dict[str, float], fix_open: List[str]) -> pyo.ConcreteModel:
    """Variables and constraints (4)-(13); the objective is attached by the caller."""
    I, J, K = case.I, case.J, case.K
    t, c, q = case.t, case.c, case.q
    big_m = {i: sum(case.w[i, k] for k in K) for i in I}          # tight M for (4)

    m = pyo.ConcreteModel()
    m.x = pyo.Var(J, domain=pyo.Binary)
    m.y = pyo.Var(I, J, domain=pyo.Binary)
    m.v = pyo.Var(I, J, K, domain=pyo.NonNegativeIntegers)
    m.u = pyo.Var(J, domain=pyo.NonNegativeIntegers)

    m.c4 = pyo.Constraint(I, J, rule=lambda m, i, j: sum(m.v[i, j, k] for k in K) <= big_m[i] * m.y[i, j])
    m.c5 = pyo.Constraint(I, J, K, rule=lambda m, i, j, k: case.d[i, j, k] * m.y[i, j] <= case.dmax)
    m.c6 = pyo.Constraint(I, K, rule=lambda m, i, k: sum(m.v[i, j, k] for j in J) == case.w[i, k])
    m.c7 = pyo.Constraint(J, rule=lambda m, j: sum(t[k] * m.v[i, j, k] for i in I for k in K) <= c * m.u[j])
    m.c8 = pyo.Constraint(J, rule=lambda m, j: m.u[j] <= q * m.x[j])
    m.c9 = pyo.Constraint(J, rule=lambda m, j: sum(m.y[i, j] for i in I) <= len(I) * m.x[j])
    m.c10 = pyo.Constraint(I, rule=lambda m, i: sum(m.y[i, j] for j in J) >= 1)
    m.c11 = pyo.Constraint(expr=sum(m.x[j] for j in J) <= case.N)
    m.c12 = pyo.Constraint(I, rule=lambda m, i: sum(p[j] * m.y[i, j] for j in J) >= case.pbar)
    m.c13 = pyo.Constraint(I, K, rule=lambda m, i, k:
                           sum(p[j] * m.v[i, j, k] for j in J) >= case.pbar * sum(m.v[i, j, k] for j in J))
    if fix_open:
        m.existing = pyo.Constraint(fix_open, rule=lambda m, j: m.x[j] == 1)
    return m


def _solve(m: pyo.ConcreteModel, time_limit: float, verbose: bool) -> None:
    solver = Highs()
    solver.config.time_limit = time_limit
    solver.config.mip_gap = 1e-9
    solver.config.stream_solver = verbose
    solver.config.load_solution = False
    result = solver.solve(m)
    if result.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(f"model not solved to optimality ({result.termination_condition.name})")
    result.solution_loader.load_vars()


def solve_design(case: Case, p: Dict[str, float], fix_open: Optional[Iterable[str]] = None,
                 tie_break: Optional[str] = None, reference: Optional[Dict[str, float]] = None,
                 time_limit: float = 300.0, verbose: bool = False) -> Design:
    """Solve the model for the reliabilities ``p`` and return the design.

    Parameters
    ----------
    p : reliabilities the planner assumes; ``1.0`` everywhere is the Non-Robust Model.
    fix_open : stations forced open (default: the existing stations in ``params.json``).
    tie_break : ``None``, ``"best"`` or ``"worst"``.  When several designs are optimal
        for ``p`` (always the case for the Non-Robust Model, whose objective ignores
        travel time), pick the one with the highest or lowest expected objective
        under the ``reference`` reliabilities.  ``None`` keeps whichever optimum the
        solver returns, which can differ between solver versions.
    reference : reliabilities used for the tie-break (usually the true ones).
    """
    if tie_break not in (None, "best", "worst"):
        raise ValueError("tie_break must be None, 'best' or 'worst'")
    if tie_break is not None and reference is None:
        raise ValueError("tie_break needs the reference reliabilities")
    I, J, K = case.I, case.J, case.K
    fix_open = case.existing if fix_open is None else list(fix_open)

    m = _build(case, p, fix_open)
    m.objective = pyo.Objective(expr=_objective(m, case, p), sense=pyo.maximize)
    _solve(m, time_limit, verbose)
    optimum = float(pyo.value(m.objective))

    if tie_break is not None:
        m.objective.deactivate()
        slack = 1e-8 * max(1.0, abs(optimum))                      # about 1 IDR for this case
        m.keep_optimal = pyo.Constraint(expr=_objective(m, case, p) >= optimum - slack)
        sense = pyo.maximize if tie_break == "best" else pyo.minimize
        m.second = pyo.Objective(expr=_objective(m, case, reference), sense=sense)
        _solve(m, time_limit, verbose)

    x = {j: int(round(pyo.value(m.x[j]))) for j in J}
    u = {j: int(round(pyo.value(m.u[j]))) for j in J}
    y = {(i, j): int(round(pyo.value(m.y[i, j]))) for i in I for j in J}
    v = {(i, j, k): int(round(pyo.value(m.v[i, j, k]))) for i in I for j in J for k in K}
    a = {j: case.r * sum(case.e[k] * v[i, j, k] for i in I for k in K) for j in J}
    b = {j: case.s * sum(case.d[i, j, k] * v[i, j, k] for i in I for k in K) for j in J}
    return Design(x=x, u=u, y=y, v=v, cost=sum(case.g * u[j] + case.h[j] * x[j] for j in J),
                  a=a, b=b, p_assumed=dict(p), model_objective=optimum)
