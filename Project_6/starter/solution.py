"""Structural turnaround model: task durations inside the real process (order of work, staff, opening hours).

    total = clock_open(return_ts, W + work)
    work  = max(charging, chains split across staff_on_duty)
    chains: interior -> special | damage check -> repair | exterior | refuel | tyre check

* Task duration = small linear formula of its drivers (dirt, child seat, battery level, ...).
* One person works the chains one after another; 2-3 people split them and the car is ready when the busiest finishes.
  Charging runs on its own.
* W = waiting before work starts: median by staff x work backlog (predicted work of cars returned at the same branch in
  the previous 60 min, per staff member).
* clock_open counts only minutes while the branch is open (branch_data.csv), so work pauses overnight and on closed days.

Fitting (train.csv only): quantile regression on 1-staff rows far from closing gives starting values; Powell then
minimises the absolute total error through the full process on a seeded sample; the waiting table is re-estimated last.
"""
import functools
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TASKS = ["quick_exterior_cleaning", "normal_exterior_cleaning", "special_cleaning", "interior_cleaning",
         "damage_check", "refuel", "electric_charging", "quick_damage_repair", "tyre_check"]
SEED = 42
TI = {t: i for i, t in enumerate(TASKS)}
CHAINS = [["interior_cleaning", "special_cleaning"], ["damage_check", "quick_damage_repair"],
          ["quick_exterior_cleaning", "normal_exterior_cleaning"], ["refuel"], ["tyre_check"]]
EPOCH = pd.Timestamp("2025-01-01")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BACKLOG_WINDOW = 60   # minutes
N_BINS = 8            # backlog bins per staff level in the waiting table
N_SUB = 8000          # cars used in the Powell search
MIN_DUR = 0.5         # shortest duration a performed task can get (minutes)


# ---------------------------------------------------------------- opening-hours clock
def _build_clock(branches, ts):
    """Per branch: open intervals (minutes since EPOCH) and cumulative open minutes. Closed days get no interval."""
    hours = {(r.branch, DAYS.index(r.day)): (r.open, r.close) for r in branches.itertuples() if r.open != "closed"}
    days = pd.date_range(ts.min().normalize() - pd.Timedelta(days=2), ts.max().normalize() + pd.Timedelta(days=30))
    mins = lambda day, hhmm: (day + pd.Timedelta(hhmm + ":00") - EPOCH) / pd.Timedelta(minutes=1)
    clock = {}
    for b in branches.branch.unique():
        iv = [(mins(d, hours[b, d.weekday()][0]), mins(d, hours[b, d.weekday()][1]))
              for d in days if (b, d.weekday()) in hours]
        s, e = np.array(iv).T
        cum = np.concatenate([[0.0], np.cumsum(e - s)[:-1]])
        clock[b] = (s, e, cum, cum + (e - s))
    return clock


def _to_open(clock, b, t):
    """Wall-clock minutes -> cumulative open minutes."""
    s, e, c, _ = clock[b]
    i = np.searchsorted(e, t, side="right")
    return c[i] + np.clip(t - s[i], 0, None)


def _from_open(clock, b, o):
    """Cumulative open minutes -> wall-clock minutes."""
    s, _, c, ce = clock[b]
    i = np.searchsorted(ce, o, side="left")
    return s[i] + (o - c[i])


# ---------------------------------------------------------------- data
class _Data:
    """Numpy view of a turnaround table; rows grouped by branch and sorted by return time."""

    def __init__(self, df, branches, staff_map, clock=None):
        ts = pd.to_datetime(df.return_ts)
        self.n = len(df)
        self.clock = clock or _build_clock(branches, ts)
        self.tmin = ((ts - EPOCH) / pd.Timedelta(minutes=1)).values.astype(float)
        self.staff = df.branch.map(staff_map).values.astype(int)
        self.flags = df[TASKS].values.astype(float)
        self.y = df.total_turnaround_min.values.astype(float) if "total_turnaround_min" in df else None
        self.X = _design(df, self.staff)
        branch = df.branch.values
        self.groups = [(b, i[np.argsort(self.tmin[i], kind="stable")])
                       for b in pd.unique(branch) for i in [np.where(branch == b)[0]]]
        self.o0 = self._per_branch(lambda b, i: _to_open(self.clock, b, self.tmin[i]))
        self.to_close = self._per_branch(lambda b, i: self._to_close(b, self.tmin[i]))

    def _to_close(self, b, t):
        s, e, _, _ = self.clock[b]
        j = np.searchsorted(e, t, side="right")
        return np.where(t >= s[j], e[j] - t, -1.0)      # -1 = returned while closed

    def _per_branch(self, f):
        out = np.empty(self.n)
        for b, i in self.groups:
            out[i] = f(b, i)
        return out

    def finish_total(self, add_open):
        """Elapsed minutes from return until ready, given the open minutes needed (wait + work)."""
        return self._per_branch(lambda b, i: _from_open(self.clock, b, self.o0[i] + add_open[i]) - self.tmin[i])

    def open_used(self, total):
        """Open minutes between return and the observed ready time."""
        return self._per_branch(lambda b, i: _to_open(self.clock, b, self.tmin[i] + total[i]) - self.o0[i])

    def backlog(self, work):
        """Predicted work of cars returned at the same branch in the previous BACKLOG_WINDOW minutes, per staff."""
        def f(b, i):
            t, cs = self.tmin[i], np.concatenate([[0.0], np.cumsum(work[i])])
            return (cs[np.searchsorted(t, t, "left")] - cs[np.searchsorted(t, t - BACKLOG_WINDOW, "left")]) / self.staff[i]
        return self._per_branch(f)


def _design(df, staff):
    """Driver columns per task (first column = intercept)."""
    one, soc = np.ones(len(df)), df.battery_soc_pct.fillna(50).values
    cols = {
        "quick_exterior_cleaning": [one, df.exterior_dirt, df.queue_length, staff == 2, staff == 3],
        "normal_exterior_cleaning": [one, df.exterior_dirt, df.queue_length],
        "special_cleaning": [one, df.smoking_reported],
        "interior_cleaning": [one, df.interior_dirt, df.child_seat, df.queue_length],
        "damage_check": [one],
        "refuel": [one, staff == 2, staff == 3],
        "electric_charging": [one, 100 - soc],
        "quick_damage_repair": [one, df.damage_reported],
        "tyre_check": [one],
    }
    return {t: np.column_stack([np.asarray(c, dtype=float) for c in v]) for t, v in cols.items()}


# ---------------------------------------------------------------- process model
@functools.lru_cache(maxsize=None)
def _assignments(k):
    """Distinct ways to give the chains to k people, as one-hot (ways x chains x k)."""
    canonical = lambda a: all(a[i] <= max(a[:i], default=-1) + 1 for i in range(len(a)))
    ways = [a for a in itertools.product(range(k), repeat=len(CHAINS)) if canonical(a)]
    return (np.array(ways)[:, :, None] == np.arange(k)).astype(float)


def _durations(data, theta, sl):
    return np.column_stack([np.clip(data.X[t] @ theta[sl[t]], MIN_DUR, None) * data.flags[:, TI[t]] for t in TASKS])


def _work(dur, staff):
    """Time until the busiest person (or the charger) is done."""
    ch = np.column_stack([sum(dur[:, TI[t]] for t in g) for g in CHAINS])
    mk = np.empty(len(dur))
    for k in np.unique(staff):
        m = staff == k
        mk[m] = np.einsum("nc,ack->nak", ch[m], _assignments(int(k))).max(2).min(1)
    return np.maximum(mk, dur[:, TI["electric_charging"]])


# ---------------------------------------------------------------- waiting table
def _wait_table(staff, backlog, wait):
    """Median wait by staff x backlog bin (bin edges = quantiles of the training backlog per staff level)."""
    table = {}
    for k in np.unique(staff):
        b, w = backlog[staff == k], wait[staff == k]
        edges = np.unique(np.quantile(b, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        bins = np.searchsorted(edges, b, side="right")
        table[int(k)] = (edges, np.array([np.median(w[bins == j]) if np.any(bins == j) else np.median(w)
                                          for j in range(len(edges) + 1)]))
    return table


def _wait_predict(staff, backlog, table):
    out = np.zeros(len(staff))
    for k, (edges, med) in table.items():
        out[staff == k] = med[np.searchsorted(edges, backlog[staff == k], side="right")]
    return out


# ---------------------------------------------------------------- fitting
def _quantile_irls(X, y, tau, iters=80, eps=1e-2):
    b = np.linalg.lstsq(X, y, rcond=1e-6)[0]
    for _ in range(iters):
        r = y - X @ b
        Xw = X * (np.where(r >= 0, tau, 1 - tau) / np.maximum(np.abs(r), eps))[:, None]
        b = np.linalg.lstsq(X.T @ Xw + 1e-6 * np.eye(X.shape[1]), Xw.T @ y, rcond=None)[0]
    return b


def fit(train, branches):
    staff_map = branches.groupby("branch").staff_on_duty.first()
    data = _Data(train, branches, staff_map)
    sizes = [data.X[t].shape[1] for t in TASKS]
    sl = {t: slice(a - k, a) for t, k, a in zip(TASKS, sizes, itertools.accumulate(sizes))}
    n_par = sum(sizes)

    # 1. starting values: quantile regression where total = sum of tasks (1 staff, > 4 h before closing)
    clean = (data.staff == 1) & (data.to_close > 240)
    Z = np.column_stack([data.X[t] * data.flags[:, [TI[t]]] for t in TASKS])
    theta = _quantile_irls(Z[clean], data.y[clean], 0.2)

    # 2. structural fit: Powell on a seeded sample; waiting = staff level + slope * backlog (backlog from step 1)
    backlog = data.backlog(_work(_durations(data, theta, sl), data.staff))
    sub = np.sort(np.random.RandomState(SEED).choice(data.n, min(N_SUB, data.n), replace=False))
    ds, bl = _Data(train.iloc[sub], branches, staff_map, clock=data.clock), backlog[sub]

    def loss(p):
        th, w = p[:n_par], p[n_par:]
        wait = np.clip(w[ds.staff - 1] + w[3 + ds.staff - 1] * bl, 0, None)
        return np.abs(ds.finish_total(wait + _work(_durations(ds, th, sl), ds.staff)) - ds.y).mean()

    p = minimize(loss, np.concatenate([theta, [10.0, 0.0, 0.0, 0.1, 0.1, 0.1]]), method="Powell",
                 options={"maxfev": 12000, "xtol": 1e-3, "ftol": 1e-5}).x
    theta = p[:n_par]

    # 3. waiting table from the final durations
    work = _work(_durations(data, theta, sl), data.staff)
    wait = np.clip(data.open_used(data.y) - work, 0, None)
    return {"theta": theta, "slices": sl, "staff": staff_map.to_dict(),
            "wait_table": _wait_table(data.staff, data.backlog(work), wait)}


def predict(model, df, branches):
    data = _Data(df, branches, pd.Series(model["staff"]))
    dur = _durations(data, model["theta"], model["slices"])
    work = _work(dur, data.staff)
    total = data.finish_total(_wait_predict(data.staff, data.backlog(work), model["wait_table"]) + work)
    return pd.DataFrame({"turnaround_id": df.turnaround_id.values, "total_turnaround_min": total,
                         **{f"dur_{t}": dur[:, TI[t]] for t in TASKS}})
