"""Reproduce every table and figure of the paper.

    python scripts/reproduce.py              # about one minute on a laptop
    python scripts/reproduce.py --quick      # fewer simulated days, for a smoke test

Outputs go to ``results/`` (CSV and JSON) and ``results/figures/`` (PDF and PNG).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evcharging import (  # noqa: E402
    DEFAULT_PROXY_CORR, OutageSimulator, cv_coefficients, cv_reliability, hoeffding_sample_size,
    indicator_corr2, load_case, mc_reliability, objective_samples, running_estimate, scale_outage,
    solve_design,
)
from evcharging import plots  # noqa: E402

SEED_ESTIMATION = 2026     # days used to estimate the reliabilities
SEED_EVALUATION = 2027     # fresh days used to evaluate the designs
OUTAGE_FACTOR = 2.0        # misspecification of the outage probability 1 - p_j


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10_000, help="simulated days (default 10,000)")
    parser.add_argument("--proxy-corr", type=float, default=DEFAULT_PROXY_CORR)
    parser.add_argument("--quick", action="store_true", help="use n = 2,000")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    n = 2_000 if args.quick else args.n
    out, figs = args.out, args.out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    case = load_case()
    J = case.J
    p_true = case.reliability
    sim = OutageSimulator(case, proxy_corr=args.proxy_corr)

    # 1. Station reliabilities: Monte Carlo against control variates ---------------------
    est = sim.sample(n, seed=SEED_ESTIMATION)
    beta = cv_coefficients(est)
    mc, cv = mc_reliability(est), cv_reliability(est, beta)
    reliability = pd.DataFrame({
        "station": J, "exact": [p_true[j] for j in J],
        "mc": [e.mean for e in mc], "mc_se": [e.se for e in mc],
        "cv": [e.mean for e in cv], "cv_se": [e.se for e in cv],
    })
    reliability["se_ratio"] = reliability.mc_se / reliability.cv_se
    reliability.to_csv(out / "reliability.csv", index=False, float_format="%.6f")
    p_hat = dict(zip(J, reliability.cv))

    # 2. Designs ------------------------------------------------------------------------
    ones = {j: 1.0 for j in J}
    designs = {
        "Non-Robust": solve_design(case, ones, tie_break="worst", reference=p_true),
        "Robust": solve_design(case, p_hat),
        "Misspecified #1": solve_design(case, scale_outage(p_hat, OUTAGE_FACTOR)),
        "Misspecified #2": solve_design(case, scale_outage(p_hat, 1.0 / OUTAGE_FACTOR)),
    }
    extra = {"Non-Robust (best tie-break)": solve_design(case, ones, tie_break="best", reference=p_true)}
    rows = []
    for name, d in {**designs, **extra}.items():
        ev = d.expected(p_true)
        rows.append({
            "design": name, "stations": len(d.opened),
            "opened (connectors)": "; ".join(f"{j} ({d.u[j]})" for j in d.opened),
            "cost": d.cost, "revenue": ev["revenue"], "penalty": ev["penalty"], "objective": ev["objective"],
            "backup_share": d.backup_share(case), "mean_travel_min": d.mean_travel_time(case),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out / "designs.csv", index=False, float_format="%.4f")

    # 3. Simulated objective on fresh days ------------------------------------------------
    days = sim.sample(n, seed=SEED_EVALUATION)
    sizes = np.unique(np.round(np.linspace(20, n, 500)).astype(int))
    curves, final = {}, []
    for name, d in designs.items():
        samples = objective_samples(d, days, J, beta)
        curves[name] = {k: running_estimate(v, sizes) for k, v in samples.items()}
        for k in ("mc", "cv"):
            run = curves[name][k]
            final.append({"design": name, "estimator": k, "n": n, "mean": run["mean"][-1], "se": run["se"][-1],
                          "ci_low": run["lo"][-1], "ci_high": run["hi"][-1]})
    pd.concat([pd.DataFrame({"design": name, "estimator": k, **curves[name][k]})
               for name in curves for k in ("mc", "cv")]).to_csv(out / "objective_vs_n.csv", index=False,
                                                                 float_format="%.2f")
    final = pd.DataFrame(final)
    final.to_csv(out / "objective_final.csv", index=False, float_format="%.2f")

    # paired difference Robust - Non-Robust on the same simulated days
    diff = objective_samples(designs["Robust"], days, J, beta)["cv"] - \
        objective_samples(designs["Non-Robust"], days, J, beta)["cv"]

    # 4. Variance reduction against the proxy correlation ---------------------------------
    kappa = float(np.mean(list(case.kappa.values())))
    theory = indicator_corr2(kappa, args.proxy_corr)
    sens = []
    for rho in (0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999):
        r2 = indicator_corr2(kappa, rho)
        sens.append({"proxy_corr": rho, "indicator_corr2": r2, "se_ratio": 1 / np.sqrt(1 - r2),
                     "sample_size_ratio": 1 / (1 - r2)})
    pd.DataFrame(sens).to_csv(out / "cv_sensitivity.csv", index=False, float_format="%.5f")

    # 5. Headline numbers ---------------------------------------------------------------
    t = table.set_index("design")
    rb, nr, nr_best = t.loc["Robust"], t.loc["Non-Robust"], t.loc["Non-Robust (best tie-break)"]
    f = final.set_index(["design", "estimator"])
    se_ratio = float(np.mean([f.loc[(d, "mc"), "se"] / f.loc[(d, "cv"), "se"] for d in designs]))
    summary = {
        "n": n, "proxy_corr": args.proxy_corr,
        "reliability_min": min(p_true.values()), "reliability_max": max(p_true.values()),
        "objective_million_idr": {d: round(t.loc[d, "objective"] / 1e6, 2) for d in t.index},
        "robust_vs_non_robust_pct": {
            "objective_worst_tie_break": 100 * (rb.objective / nr.objective - 1),
            "objective_best_tie_break": 100 * (rb.objective / nr_best.objective - 1),
            "cost": 100 * (rb.cost / nr.cost - 1),
            "revenue": 100 * (rb.revenue / nr.revenue - 1),
            "penalty_worst_tie_break": 100 * (rb.penalty / nr.penalty - 1),
            "penalty_best_tie_break": 100 * (rb.penalty / nr_best.penalty - 1),
        },
        "misspecified_loss_vs_robust_pct": {
            d: 100 * (1 - t.loc[d, "objective"] / rb.objective) for d in ("Misspecified #1", "Misspecified #2")},
        "backup_share": {d: float(t.loc[d, "backup_share"]) for d in t.index},
        "mean_travel_min": {d: float(t.loc[d, "mean_travel_min"]) for d in t.index},
        "paired_gain_million_idr": {"mean": diff.mean() / 1e6, "se": diff.std(ddof=1) / np.sqrt(n) / 1e6},
        "se_ratio_mc_over_cv": {"theory": 1 / np.sqrt(1 - theory), "this_sample": se_ratio},
        "sample_size_ratio_mc_over_cv": {"theory": 1 / (1 - theory), "this_sample": se_ratio**2},
        "reliability_se_ratio_mc_over_cv": float(reliability.se_ratio.mean()),
        "hoeffding_n_delta_1e-4_alpha_0.05": hoeffding_sample_size(1e-4, 0.05),
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # 6. Figures ------------------------------------------------------------------------
    plots.use_paper_style()
    exact = {name: t.loc[name, "objective"] for name in designs}
    plots.save(plots.figure_objective(curves, exact), figs / "fig3_objective")
    plots.save(plots.figure_cv_sensitivity(sens, args.proxy_corr), figs / "cv_sensitivity")

    # Maps are laid out to fill the figure, exactly like the published one, so they are
    # written without a tight bounding box and at twice their nominal size.
    maps = {"fig2_candidates": plots.figure_candidates(case),
            "fig4_designs": plots.figure_designs(case, {k: designs[k] for k in ("Non-Robust", "Robust")}),
            "designs_all": plots.figure_designs(case, designs),
            "map_robust": plots.figure_map(case, designs["Robust"]),
            "map_non_robust": plots.figure_map(case, designs["Non-Robust"])}
    for name, fig in maps.items():
        plots.save(fig, figs / name, tight=False, dpi=360)

    # 7. Console report -----------------------------------------------------------------
    show = table.copy()
    for col in ("cost", "revenue", "penalty", "objective"):
        show[col] = (show[col] / 1e6).round(2)
    show["backup_share"] = (100 * show["backup_share"]).round(0).astype(int).astype(str) + "%"
    show["mean_travel_min"] = show["mean_travel_min"].round(1)
    print("\nDesigns, evaluated under the exact reliabilities (million IDR per day)\n")
    print(show.to_string(index=False))
    g = summary["robust_vs_non_robust_pct"]
    print(f"\nRobust against Non-Robust: objective {g['objective_best_tie_break']:+.1f}% to "
          f"{g['objective_worst_tie_break']:+.1f}% (best to worst tie-break), cost {g['cost']:+.1f}%, "
          f"revenue {g['revenue']:+.2f}%, penalty {g['penalty_best_tie_break']:+.1f}% to "
          f"{g['penalty_worst_tie_break']:+.1f}%")
    print(f"Paired gain on {n:,} common days (CV): {summary['paired_gain_million_idr']['mean']:.2f} "
          f"\u00b1 {1.96 * summary['paired_gain_million_idr']['se']:.2f} million IDR per day")
    print(f"Standard error, MC over CV: {1 / np.sqrt(1 - theory):.1f}x in theory, {se_ratio:.1f}x in this sample "
          f"(sample size: {1 / (1 - theory):.0f}x and {se_ratio**2:.0f}x)")
    print(f"\nWrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}/")


if __name__ == "__main__":
    main()
