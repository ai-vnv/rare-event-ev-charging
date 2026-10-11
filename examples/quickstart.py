"""Smallest end-to-end example: estimate reliabilities, solve two designs, compare them.

    python examples/quickstart.py
"""
from evcharging import OutageSimulator, cv_reliability, load_case, solve_design

case = load_case()                                  # or load_case("path/to/your/data")

# 1. Estimate each station's reliability from 5,000 simulated days.
draws = OutageSimulator(case).sample(5_000, seed=0)
p_hat = {j: est.mean for j, est in zip(case.J, cv_reliability(draws))}

# 2. Solve the Robust Model (estimated reliabilities) and the Non-Robust Model (p_j = 1).
robust = solve_design(case, p_hat)
non_robust = solve_design(case, {j: 1.0 for j in case.J}, tie_break="worst", reference=p_hat)

# 3. Compare the two designs under the exact reliabilities.
for name, design in [("Robust", robust), ("Non-Robust", non_robust)]:
    value = design.expected(case.reliability)
    opened = ", ".join(f"{j} ({design.u[j]})" for j in design.opened)
    print(f"{name:11s} objective {value['objective'] / 1e6:7.2f} million IDR/day | {opened}")
