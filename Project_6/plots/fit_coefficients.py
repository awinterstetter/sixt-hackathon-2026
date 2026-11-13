"""Fit starter/solution.py on train.csv once and save the per-task regression coefficients for plotting.

Run from Project_6/:  .venv/Scripts/python.exe analysis/fit_coefficients.py
Writes analysis/data/model_coefficients.json (task -> {covariate: coefficient in minutes}).
Takes as long as one fit() (about 12 minutes).
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
    """Column names in the same order as solution._design()."""
    q = ["queue_length"] if hp["queue_in_cleaning"] else []
    if hp.get("covariates", "base") == "base":
        return {
            "quick_exterior_cleaning": ["intercept", "exterior_dirt", *q, "staff = 2", "staff = 3"],
            "normal_exterior_cleaning": ["intercept", "exterior_dirt", *q],
            "special_cleaning": ["intercept", "smoking_reported"],
            "interior_cleaning": ["intercept", "interior_dirt", "child_seat", *q],
            "damage_check": ["intercept"],
            "refuel": ["intercept", "staff = 2", "staff = 3"],
            "electric_charging": ["intercept", "battery_soc_pct"],   # enters as 100 - SOC
            "quick_damage_repair": ["intercept", "damage_reported"],
            "tyre_check": ["intercept"],
        }
    return {   # "corr" covariate set
        "quick_exterior_cleaning": ["intercept", "exterior_dirt", *q, "staff = 2", "staff = 3", "van"],
        "normal_exterior_cleaning": ["intercept", "exterior_dirt", *q, "van", "rain"],
        "special_cleaning": ["intercept", "smoking_reported", "premium"],
        "interior_cleaning": ["intercept", "interior_dirt", "child_seat", *q, "rental_days", "km_driven / 100",
                              "staff = 3"],
        "damage_check": ["intercept", "damage_reported"],
        "refuel": ["intercept", "staff = 2", "staff = 3", "suv", "van"],
        "electric_charging": ["intercept", "battery_soc_pct"],   # enters as 100 - SOC
        "quick_damage_repair": ["intercept", "damage_reported"],
        "tyre_check": ["intercept", "staff = 3"],
    }


if __name__ == "__main__":
    train = pd.read_csv(P6 / "data" / "train.csv", parse_dates=["return_ts"])
    branches = pd.read_csv(P6 / "data" / "branch_data.csv")
    t0 = time.time()
    model = sol.fit(train, branches)
    names = covariate_names(model["hp"])
    out = {}
    for t in sol.TASKS:
        coef = model["theta"][model["slices"][t]]
        assert len(coef) == len(names[t]), f"{t}: {len(coef)} coefficients but {len(names[t])} names"
        out[t] = {n: float(c) for n, c in zip(names[t], coef)}
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "model_coefficients.json").write_text(json.dumps(out, indent=2))
    print(f"fit {time.time() - t0:.0f}s; wrote {HERE / 'data' / 'model_coefficients.json'}")
