"""Re-solve the design for a range of maximum travel times.

The paper fixes the maximum travel time ``d_max`` of constraint (5) at 35 minutes.
This script sweeps it and records, for every value, the opened stations, the
connectors, the vehicle flows and the expected revenue, penalty, cost and objective
under the exact reliabilities, for both the Robust and the Non-Robust Model.  Values
at which no design exists are recorded as infeasible.  The output feeds
``results/interactive.html``.

    python scripts/sweep_travel_time.py --lo 10 --hi 75 --out results/travel_time_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evcharging.data import load_case
from evcharging.model import solve_design


def solve_or_none(case, p, **kwargs):
    try:
        return solve_design(case, p, **kwargs)
    except RuntimeError as err:               # infeasible or not solved
        if "infeasible" in str(err).lower():
            return None
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lo", type=int, default=10)
    ap.add_argument("--hi", type=int, default=75)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/travel_time_sweep.json"))
    args = ap.parse_args()

    case = load_case()
    p_true = case.reliability
    p_one = {j: 1.0 for j in case.J}
    paper_dmax = case.dmax

    models = {
        "robust": dict(p=p_true),
        "non_robust": dict(p=p_one, tie_break="worst", reference=p_true),
    }
    sweep = {name: {} for name in models}

    for dmax in range(args.lo, args.hi + 1, args.step):
        case.params["max_travel_time_min"] = dmax
        for name, kw in models.items():
            design = solve_or_none(case, **kw)
            if design is None:
                sweep[name][dmax] = {"feasible": False}
                print(f"{name:11s} dmax={dmax:3d}  infeasible")
                continue
            exp = design.expected(p_true)
            flows = [{"i": i, "j": j, "n": design.flow(case, i, j)}
                     for i in case.I for j in case.J if design.flow(case, i, j) > 0]
            sweep[name][dmax] = {
                "feasible": True,
                "opened": {j: design.u[j] for j in design.opened},
                "revenue": exp["revenue"], "penalty": exp["penalty"],
                "cost": exp["cost"], "objective": exp["objective"],
                "backup_share": design.backup_share(case),
                "mean_travel_min": design.mean_travel_time(case),
                "flows": flows,
            }
            print(f"{name:11s} dmax={dmax:3d}  {exp['objective']/1e6:8.2f} M IDR  "
                  f"{', '.join(f'{j} ({design.u[j]})' for j in design.opened)}")
    case.params["max_travel_time_min"] = paper_dmax

    out = {
        "paper_dmax": paper_dmax,
        "lo": args.lo, "hi": args.hi, "step": args.step,
        "stations": [{"name": r.station, "lat": r.lat, "lon": r.lon,
                      "existing": r.station in case.existing}
                     for r in case.stations.itertuples()],
        # reach: travel time to the nearest candidate (slower vehicle type); the
        # model is infeasible as soon as d_max falls below the largest of these.
        "demand": [{"name": r.subdistrict, "lat": r.lat, "lon": r.lon,
                    "population": int(r.population),
                    "vehicles": int(r.demand_motorcycle + r.demand_car),
                    "reach": min(max(case.d[r.subdistrict, j, k] for k in case.K) for j in case.J)}
                   for r in case.demand.itertuples()],
        "sweep": sweep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
