"""Covariate screen per task on the validation week's TRUE task durations.

For each task: least squares on the rows where the task was performed, scored by leave-one-day-out MAE.
  base     = covariates currently used in solution.py (_design)
  add  X   = base + one unused candidate      (negative delta = candidate helps)
  drop X   = base - one used covariate         (positive delta = covariate was helping)
Run from Project_6/:  .venv/Scripts/python.exe analysis/feature_screen.py
Writes analysis/figures/feature_screen.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TASKS = ["quick_exterior_cleaning", "normal_exterior_cleaning", "special_cleaning", "interior_cleaning",
         "damage_check", "refuel", "electric_charging", "quick_damage_repair", "tyre_check"]
STAFF = {"AIRPORT": 3, "STATION": 2, "CENTER": 1}
CURRENT = {  # as in solution._design (queue_in_cleaning = True)
    "quick_exterior_cleaning": ["exterior_dirt", "queue_length", "staff"],
    "normal_exterior_cleaning": ["exterior_dirt", "queue_length"],
    "special_cleaning": ["smoking_reported"],
    "interior_cleaning": ["interior_dirt", "child_seat", "queue_length"],
    "damage_check": [],
    "refuel": ["staff"],
    "electric_charging": ["battery_missing"],
    "quick_damage_repair": ["damage_reported"],
    "tyre_check": [],
}
CANDIDATES = ["interior_dirt", "exterior_dirt", "child_seat", "pet", "smoking_reported", "damage_reported",
              "fuel_missing", "battery_missing", "km_driven", "rental_days", "vehicle_class", "powertrain",
              "weather", "queue_length", "staff", "branch", "return_hour", "weekday"]


def blocks(va):
    """Each candidate as a block of columns (categoricals as dummies)."""
    staff = va.branch.str.split("_").str[1].map(STAFF)
    num = {
        "interior_dirt": va.interior_dirt, "exterior_dirt": va.exterior_dirt, "child_seat": va.child_seat,
        "pet": va.pet, "smoking_reported": va.smoking_reported, "damage_reported": va.damage_reported,
        "fuel_missing": 100 - va.fuel_level_pct.fillna(100), "battery_missing": 100 - va.battery_soc_pct.fillna(50),
        "km_driven": va.km_driven / 100, "rental_days": va.rental_days, "queue_length": va.queue_length,
        "return_hour": va.return_hour,
    }
    out = {k: np.asarray(v, float)[:, None] for k, v in num.items()}
    cat = {"vehicle_class": va.vehicle_class, "powertrain": va.powertrain, "weather": va.weather,
           "staff": staff.astype(str), "branch": va.branch, "weekday": va.weekday.astype(str)}
    for k, v in cat.items():
        out[k] = pd.get_dummies(v, drop_first=True).values.astype(float)
    return out


def lodo_mae(X, y, day):
    pred = np.empty(len(y))
    for d in np.unique(day):
        te = day == d
        b = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
        pred[te] = X[te] @ b
    return np.abs(np.clip(pred, 0.5, None) - y).mean()


if __name__ == "__main__":
    va = pd.read_csv(DATA / "validation.csv", parse_dates=["return_ts"])
    lab = pd.read_csv(DATA / "validation_labels.csv")
    B = blocks(va)
    day = va.return_ts.dt.dayofyear.values
    rows = []
    for t in TASKS:
        m = (va[t] == 1).values
        y, dd = lab.loc[m, t].values, day[m]
        design = lambda names: np.column_stack([np.ones(m.sum())] + [B[n][m] for n in names])
        base = lodo_mae(design(CURRENT[t]), y, dd)
        rows.append({"task": t, "n": int(m.sum()), "change": "base (current)", "covariate": "", "mae": base, "delta": 0.0})
        rows.append({"task": t, "n": int(m.sum()), "change": "intercept only", "covariate": "",
                     "mae": lodo_mae(design([]), y, dd), "delta": lodo_mae(design([]), y, dd) - base})
        for c in CURRENT[t]:
            v = lodo_mae(design([x for x in CURRENT[t] if x != c]), y, dd)
            rows.append({"task": t, "n": int(m.sum()), "change": "drop", "covariate": c, "mae": v, "delta": v - base})
        for c in CANDIDATES:
            if c in CURRENT[t] or (c == "branch" and "staff" in CURRENT[t]):
                continue
            v = lodo_mae(design(CURRENT[t] + [c]), y, dd)
            rows.append({"task": t, "n": int(m.sum()), "change": "add", "covariate": c, "mae": v, "delta": v - base})
    res = pd.DataFrame(rows)
    res.to_csv(HERE / "figures" / "feature_screen.csv", index=False)
    pd.set_option("display.width", 200)
    for t in TASKS:
        r = res[res.task == t]
        base = r[r.change == "base (current)"].mae.iloc[0]
        helps = r[(r.change == "add") & (r.delta < -0.05)].sort_values("delta")
        needed = r[(r.change == "drop")].sort_values("delta", ascending=False)
        print(f"\n{t} (n={r.n.iloc[0]}): current MAE {base:.2f}, intercept only {r[r.change == 'intercept only'].mae.iloc[0]:.2f}")
        print("   drop -> delta: " + ", ".join(f"{c} {d:+.2f}" for c, d in zip(needed.covariate, needed.delta)))
        print("   add (helps >0.05): " + (", ".join(f"{c} {d:+.2f}" for c, d in zip(helps.covariate, helps.delta)) or "none"))
