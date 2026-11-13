"""Fit starter/solution.py on train.csv once and save the per-task regression coefficients for plotting.

Run from Project_6/:  .venv/Scripts/python.exe plots/fit_coefficients.py
Writes plots/data/model_coefficients.json (task -> {covariate: coefficient in minutes}).
Takes as long as one fit() (about 3 minutes).
"""
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P6 = HERE.parent
spec = importlib.util.spec_from_file_location("solution", P6 / "starter" / "solution.py")
sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)


def covariate_names(hp):
    """Column names in the same order as solution._design (defined in solution.COVARIATES)."""
    return sol.covariate_names(hp)


if __name__ == "__main__":
    train = pd.read_csv(P6 / "data" / "train.csv", parse_dates=["return_ts"])
    branches = pd.read_csv(P6 / "data" / "branch_data.csv")
    t0 = time.time()
    model = sol.fit(train, branches)
    tab = sol.coefficient_table(model)
    out = {t: dict(zip(g.covariate, g.coefficient)) for t, g in tab.groupby("task", sort=False)}
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "model_coefficients.json").write_text(json.dumps(out, indent=2))
    print(f"fit {time.time() - t0:.0f}s; wrote {HERE / 'data' / 'model_coefficients.json'}")
