"""Structural turnaround model: task durations inside the real process (order of work, staff, opening hours).

    total = clock_open(return_ts, W + work)
    work  = max(charging, chains split across staff_on_duty)
    chains: interior -> special | damage check -> repair | exterior | refuel | tyre check

* Task duration = small linear formula of its drivers (dirt, child seat, battery level, ...).
* One person works the chains one after another; 2-3 people split them and the car is ready when the busiest finishes.
  Charging runs on its own.
* W = waiting before work starts: median regression per staff level on backlog, overnight drop-offs, time since
  opening, own work, daily load, hour and weekday.
* clock_open counts only minutes while the branch is open (branch_data.csv), so work pauses overnight and on closed days.

Fitting (train.csv only): (1) starting values from median totals with vs without each task; (2) Powell minimises the
absolute total error through the full process on a seeded sample; (3) EM on the critical path reconstructs each car's
task durations and refits every task formula on them (these are the final coefficients); (4) waiting model last.
All settings are hyperparameters in HP; fit(train, branches, hp={...}) overrides them.
"""
import functools
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

TASKS = ["quick_exterior_cleaning", "normal_exterior_cleaning", "special_cleaning", "interior_cleaning",
         "damage_check", "refuel", "electric_charging", "quick_damage_repair", "tyre_check"]
TI = {t: i for i, t in enumerate(TASKS)}
CHAINS = [["interior_cleaning", "special_cleaning"], ["damage_check", "quick_damage_repair"],
          ["quick_exterior_cleaning", "normal_exterior_cleaning"], ["refuel"], ["tyre_check"]]
EPOCH = pd.Timestamp("2025-01-01")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

HP = {
    # waiting model
    "backlog_window": 60,      # minutes of earlier returns that count towards the backlog
    "wait_model": "linear",    # "linear": median regression per staff level; "table": staff x backlog-bin lookup
    "n_bins": 8,               # backlog bins per staff level (table model)
    "wait_quantile": 0.5,      # which quantile of the training waits to predict (0.5 = median)
    "morning_cap": 240,        # minutes since opening after which the time-of-day feature stops growing
    "wait_select": True,       # waiting drivers per staff level chosen by the Bonferroni rule (WAIT_SELECTED)
    "daily_features": True,    # add daily load, work so far today, hour and weekday to the linear waiting model
    # starting values
    "start": "median_diff",    # "median_diff": median total with vs without each task; "quantile": quantile regression
    "tau": 0.2,                # quantile for the starting-value regression
    "clean_to_close": 240,     # starting values use 1-staff cars returned at least this many minutes before closing
    # structural fit
    "n_sub": 8000,             # cars in the Powell sample
    "maxfev": 12000,           # Powell function-evaluation budget (first structural fit)
    "optimizer": "powell",     # "powell": scipy Powell (default); "coordinate": one parameter at a time (tested: worse)
    # EM on the critical path
    "em_iters": 30,            # EM rounds (0 = no EM)
    "xtol": 1e-3,
    "ftol": 1e-5,
    "w_init": (10.0, 0.0, 0.0, 0.1, 0.1, 0.1),   # waiting level per staff level (1, 2, 3), then backlog slope
    # drivers and plausibility
    "queue_in_cleaning": True, # queue_length as a driver of the three cleaning tasks
    "covariates": "corr",      # "corr": drivers selected by correlation with task duration; "base": original set
    "min_dur": 0.5,            # shortest duration a performed task can get (minutes)
    "seed": 42,
}


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

    def __init__(self, df, branches, staff_map, hp, clock=None):
        ts = pd.to_datetime(df.return_ts)
        self.n, self.hp = len(df), hp
        self.clock = clock or _build_clock(branches, ts)
        self.tmin = ((ts - EPOCH) / pd.Timedelta(minutes=1)).values.astype(float)
        self.staff = df.branch.map(staff_map).values.astype(int)
        self.flags = df[TASKS].values.astype(float)
        self.y = df.total_turnaround_min.values.astype(float) if "total_turnaround_min" in df else None
        self.X = _design(df, self.staff, hp)
        branch = df.branch.values
        self.groups = [(b, i[np.argsort(self.tmin[i], kind="stable")])
                       for b in pd.unique(branch) for i in [np.where(branch == b)[0]]]
        self.o0 = self._per_branch(lambda b, i: _to_open(self.clock, b, self.tmin[i]))
        self.to_close = self._per_branch(lambda b, i: self._to_close(b, self.tmin[i]))
        # opening interval of each car (cars returned while closed belong to the next opening) and open minutes since it began
        self.interval = self._per_branch(lambda b, i: np.searchsorted(self.clock[b][1], self.tmin[i], side="right"))
        self.since_open = self._per_branch(lambda b, i: self.o0[i] - self.clock[b][2][self.interval[i].astype(int)])
        self.closed_return = self.to_close < 0
        self.weekday = ts.dt.weekday.values
        self.weekend = (self.weekday >= 5).astype(float)
        self.hour = ts.dt.hour.values.astype(float)

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
        """Predicted work of cars returned at the same branch in the previous backlog_window minutes, per staff."""
        win = self.hp["backlog_window"]

        def f(b, i):
            t, cs = self.tmin[i], np.concatenate([[0.0], np.cumsum(work[i])])
            return (cs[np.searchsorted(t, t, "left")] - cs[np.searchsorted(t, t - win, "left")]) / self.staff[i]
        return self._per_branch(f)

    def overnight(self, work):
        """Predicted work of the drop-offs (cars returned while closed) ahead of this car at the same opening, per staff.

        Drop-offs pile up until the next opening and are then worked through in drop-off order, so a car dropped off
        while closed only waits for the drop-offs before it; a car returned while open has the whole pile ahead of it.
        """
        def f(b, i):
            closed = self.closed_return[i]
            w = pd.Series(work[i] * closed)
            g = w.groupby(self.interval[i])                    # rows of i are in return order
            before = g.cumsum().values - w.values              # drop-offs earlier the same night
            return np.where(closed, before, g.transform("sum").values) / self.staff[i]
        return self._per_branch(f)

    def daily_load(self, work):
        """Predicted work of all cars belonging to the same opening day at the branch, per staff."""
        return self._per_branch(lambda b, i: pd.Series(work[i]).groupby(self.interval[i]).transform("sum").values
                                / self.staff[i])

    def work_so_far(self, work):
        """Predicted work of the earlier cars of the same opening day at the branch, per staff."""
        return self._per_branch(lambda b, i: (pd.Series(work[i]).groupby(self.interval[i]).cumsum().values - work[i])
                                / self.staff[i])


# covariates per task (first = intercept). "corr" (default): a covariate is used if |r| >= 0.1 and p < 0.05 / 199
# (Bonferroni over all 199 task x variable tests) against the task's true duration on validation (tasks performed only).
# Branch type = staff level here (airport 3, station 2, center 1), so it enters as "staff = 2" / "staff = 3".
COVARIATES = {
    "base": {
        "quick_exterior_cleaning": ["intercept", "exterior_dirt", "queue_length", "staff = 2", "staff = 3"],
        "normal_exterior_cleaning": ["intercept", "exterior_dirt", "queue_length"],
        "special_cleaning": ["intercept", "smoking_reported"],
        "interior_cleaning": ["intercept", "interior_dirt", "child_seat", "queue_length"],
        "damage_check": ["intercept"],
        "refuel": ["intercept", "staff = 2", "staff = 3"],
        "electric_charging": ["intercept", "battery_soc_pct"],
        "quick_damage_repair": ["intercept", "damage_reported"],
        "tyre_check": ["intercept"],
    },
    "corr": {
        "quick_exterior_cleaning": ["intercept", "exterior_dirt", "queue_length", "staff = 2", "staff = 3", "van"],
        "normal_exterior_cleaning": ["intercept", "exterior_dirt", "queue_length", "van", "rain"],
        "special_cleaning": ["intercept", "smoking_reported", "premium"],
        "interior_cleaning": ["intercept", "interior_dirt", "child_seat", "queue_length", "rental_days",
                              "km_driven / 100", "staff = 3"],
        "damage_check": ["intercept", "damage_reported"],
        "refuel": ["intercept", "staff = 2", "staff = 3", "suv", "van"],
        "electric_charging": ["intercept", "battery_soc_pct"],
        "quick_damage_repair": ["intercept", "damage_reported"],
        "tyre_check": ["intercept", "staff = 3"],
    },
}


def covariate_names(hp):
    """Covariate names per task, in the column order of _design (queue_length only if queue_in_cleaning)."""
    return {t: [c for c in v if c != "queue_length" or hp["queue_in_cleaning"]]
            for t, v in COVARIATES[hp["covariates"]].items()}


def _design(df, staff, hp):
    """Covariate columns per task, built from covariate_names(hp)."""
    vc = df.vehicle_class.values
    col = {
        "intercept": np.ones(len(df)), "exterior_dirt": df.exterior_dirt, "interior_dirt": df.interior_dirt,
        "queue_length": df.queue_length, "child_seat": df.child_seat, "smoking_reported": df.smoking_reported,
        "damage_reported": df.damage_reported, "rental_days": df.rental_days, "km_driven / 100": df.km_driven / 100,
        "battery_soc_pct": 100 - df.battery_soc_pct.fillna(50).values,      # enters as % battery missing
        "staff = 2": staff == 2, "staff = 3": staff == 3,
        "van": vc == "van", "suv": vc == "suv", "premium": vc == "premium", "rain": (df.weather == "rain").values,
    }
    return {t: np.column_stack([np.asarray(col[c], dtype=float) for c in names])
            for t, names in covariate_names(hp).items()}


# ---------------------------------------------------------------- process model
@functools.lru_cache(maxsize=None)
def _assignments(k):
    """Distinct ways to give the chains to k people, as one-hot (ways x chains x k)."""
    canonical = lambda a: all(a[i] <= max(a[:i], default=-1) + 1 for i in range(len(a)))
    ways = [a for a in itertools.product(range(k), repeat=len(CHAINS)) if canonical(a)]
    return (np.array(ways)[:, :, None] == np.arange(k)).astype(float)


def _durations(data, theta, sl):
    return np.column_stack([np.clip(data.X[t] @ theta[sl[t]], data.hp["min_dur"], None) * data.flags[:, TI[t]]
                            for t in TASKS])


def _work(dur, staff, critical=False):
    """Time until the busiest person (or the charger) is done; optionally also which tasks are on the critical path."""
    ch = np.column_stack([sum(dur[:, TI[t]] for t in g) for g in CHAINS])
    mk = np.empty(len(dur))
    crit_ch = np.zeros((len(dur), len(CHAINS)), bool)
    for k in np.unique(staff):
        m = np.where(staff == k)[0]
        oh = _assignments(int(k))
        loads = np.einsum("nc,ack->nak", ch[m], oh)
        best = loads.max(2).argmin(1)
        mk[m] = loads.max(2)[np.arange(len(m)), best]
        if critical:   # chains done by the busiest person in the best split
            crit_ch[m] = oh[best][np.arange(len(m)), :, loads[np.arange(len(m)), best].argmax(1)] > 0
    charge = dur[:, TI["electric_charging"]]
    work = np.maximum(mk, charge)
    if not critical:
        return work
    crit = np.zeros_like(dur, bool)
    for c, g in enumerate(CHAINS):
        for t in g:
            crit[:, TI[t]] = crit_ch[:, c] & (charge <= mk)
    crit[:, TI["electric_charging"]] = charge > mk
    return work, crit & (dur > 0)


# ---------------------------------------------------------------- waiting table
def _wait_table(staff, backlog, wait, hp):
    """Quantile of the waits by staff x backlog bin (bin edges = quantiles of the training backlog per staff)."""
    q, table = hp["wait_quantile"], {}
    for k in np.unique(staff):
        b, w = backlog[staff == k], wait[staff == k]
        edges = np.unique(np.quantile(b, np.linspace(0, 1, hp["n_bins"] + 1)[1:-1]))
        bins = np.searchsorted(edges, b, side="right")
        table[int(k)] = (edges, np.array([np.quantile(w[bins == j] if np.any(bins == j) else w, q)
                                          for j in range(len(edges) + 1)]))
    return table


def _wait_predict(staff, backlog, table):
    out = np.zeros(len(staff))
    for k, (edges, val) in table.items():
        out[staff == k] = val[np.searchsorted(edges, backlog[staff == k], side="right")]
    return out


# ---------------------------------------------------------------- fitting
def _quantile_irls(X, y, tau, iters=80, eps=1e-2):
    b = np.linalg.lstsq(X, y, rcond=1e-6)[0]
    for _ in range(iters):
        r = y - X @ b
        Xw = X * (np.where(r >= 0, tau, 1 - tau) / np.maximum(np.abs(r), eps))[:, None]
        b = np.linalg.lstsq(X.T @ Xw + 1e-6 * np.eye(X.shape[1]), Xw.T @ y, rcond=None)[0]
    return b


def _fit_durations(train, branches, staff_map, hp):
    """Steps 1-3: starting values, structural fit, EM. Returns the prepared data, parameters and slices."""
    data = _Data(train, branches, staff_map, hp)
    sizes = [data.X[t].shape[1] for t in TASKS]
    sl = {t: slice(a - k, a) for t, k, a in zip(TASKS, sizes, itertools.accumulate(sizes))}

    # 1. starting values: quantile regression where total = sum of tasks (1 staff, far from closing)
    clean = (data.staff == 1) & (data.to_close > hp["clean_to_close"])
    if hp["start"] == "median_diff":
        theta = _median_diff_start(data, clean, sl)
    else:
        Z = np.column_stack([data.X[t] * data.flags[:, [TI[t]]] for t in TASKS])
        theta = _quantile_irls(Z[clean], data.y[clean], hp["tau"])

    # 2. structural fit through the full process
    sub = np.sort(np.random.RandomState(hp["seed"]).choice(data.n, min(hp["n_sub"], data.n), replace=False))
    ds = _Data(train.iloc[sub], branches, staff_map, hp, clock=data.clock)
    theta, w = _powell(data, ds, sub, theta, np.array(hp["w_init"], float), sl, hp, hp["maxfev"])
    if hp["em_iters"] == 0:
        return data, theta, sl

    # 3. EM on the critical path: reconstruct each car's task durations, refit each task on its own drivers
    theta = _struct_em(data, theta, sl, hp)
    return data, theta, sl


def _median_diff_start(data, clean, sl):
    """Intuitive starting values on 1-staff cars far from closing: what does each task add to the median total?
    Interior and quick exterior are on every 'basic' car, so their median total is split half and half.
    All driver effects start at 0."""
    F, y = data.flags[clean].astype(bool), data.y[clean]
    other = [t for t in TASKS if t not in ("interior_cleaning", "quick_exterior_cleaning", "normal_exterior_cleaning")]
    no_other = ~F[:, [TI[t] for t in other]].any(1)
    basic = np.median(y[no_other & F[:, TI["quick_exterior_cleaning"]]])
    basic_normal = np.median(y[no_other & F[:, TI["normal_exterior_cleaning"]]])
    start = {"interior_cleaning": basic / 2, "quick_exterior_cleaning": basic / 2,
             "normal_exterior_cleaning": basic / 2 + (basic_normal - basic)}
    for t in other:
        start[t] = np.median(y[F[:, TI[t]]]) - np.median(y[~F[:, TI[t]]])
    theta = np.zeros(sum(s.stop - s.start for s in sl.values()))
    for t in TASKS:
        theta[sl[t].start] = max(start[t], 0.5)                    # intercept; slopes stay 0
    return theta


def _powell(data, ds, sub, theta, w, sl, hp, maxfev):
    """Minimise mean |total - predicted total| on the sample ds; waiting = staff level + slope * recent backlog."""
    n_par = len(theta)
    bl = data.backlog(_work(_durations(data, theta, sl), data.staff))[sub]   # backlog from the starting durations

    def unpack(p):
        return p[:n_par], p[n_par:]

    def loss(p):
        th, ww = unpack(p)
        wait = np.clip(ww[ds.staff - 1] + ww[3 + ds.staff - 1] * bl, 0, None)
        return np.abs(ds.finish_total(wait + _work(_durations(ds, th, sl), ds.staff)) - ds.y).mean()

    x0 = np.concatenate([theta, w])
    if hp["optimizer"] == "coordinate":
        p = _coordinate_descent(loss, x0, maxfev, hp["ftol"])
    else:
        p = minimize(loss, x0, method="Powell", options={"maxfev": maxfev, "xtol": hp["xtol"], "ftol": hp["ftol"]}).x
    return unpack(p)


def _coordinate_descent(f, x0, maxfev, ftol, step=1.0):
    """Change one parameter at a time: a 1-D search along each axis, sweep after sweep, until a full sweep no longer
    lowers the error (relative improvement below ftol) or the evaluation budget is used up."""
    x = np.array(x0, float)
    fx, n_eval = f(x), 1
    while n_eval < maxfev:
        f_start = fx
        for j in range(len(x)):
            def along(s, j=j):
                y = x.copy()
                y[j] += s
                return f(y)
            r = minimize_scalar(along, bracket=(-step, step), method="brent", options={"xtol": 1e-3, "maxiter": 50})
            n_eval += r.nfev
            if r.fun < fx:
                x[j] += r.x
                fx = r.fun
            if n_eval >= maxfev:
                break
        if f_start - fx <= ftol * max(abs(f_start), 1e-9):
            break
    return x


def _struct_em(data, theta, sl, hp):
    """EM: per car, share the time the model can't explain among the tasks on the critical path and the waiting time
    (in proportion to how much each varies), then refit each task's formula on its reconstructed durations."""
    th = theta.copy()
    used = data.open_used(data.y)
    F = data.flags.astype(bool)
    sig2, s2w = np.full(len(TASKS), 9.0), 100.0
    wait_coef = None
    for _ in range(hp["em_iters"]):
        mu = _durations(data, th, sl)
        work, crit = _work(mu, data.staff, critical=True)
        X = _wait_features(data, mu, work, hp)
        if wait_coef is None:                                     # first round: waiting = what work doesn't explain
            wait_coef = _fit_wait_linear(data, X, np.clip(used - work, 0, None), hp)
        w0 = _predict_wait_linear(data, X, wait_coef)
        gap = used - w0 - work                                    # time the current model can't explain
        vt = (crit * sig2).sum(1)
        v = vt + s2w
        d_hat = mu + np.where(crit, sig2 / v[:, None] * gap[:, None], 0.0)
        w_hat = w0 + s2w / v * gap
        neg = w_hat < 0                                           # waiting can't be negative
        if neg.any():
            g0 = (used - work)[neg]
            d_hat[neg] = mu[neg] + np.where(crit[neg], sig2 / np.maximum(vt[neg], 1e-9)[:, None] * g0[:, None], 0.0)
            w_hat[neg] = 0.0
        d_hat = np.clip(d_hat, hp["min_dur"], None) * F
        for t in TASKS:                                           # M-step: one regression per task
            m = F[:, TI[t]]
            th[sl[t]] = np.linalg.lstsq(data.X[t][m], d_hat[m, TI[t]], rcond=1e-6)[0]
            c = crit[m, TI[t]]
            res = d_hat[m, TI[t]][c] - data.X[t][m][c] @ th[sl[t]] if c.sum() > 10 else np.array([3.0])
            sig2[TI[t]] = max(float(np.mean(res ** 2)), 0.25)
        wait_coef = _fit_wait_linear(data, X, w_hat, hp, iters=30)
        s2w = max(float(np.var(w_hat - w0)), 1.0)
    return th


def _wait_features(data, dur, work, hp):
    """Waiting drivers: recent backlog, overnight drop-offs still left at this time, time since opening (capped),
    repair needed, the car's own work, weekend."""
    left_overnight = np.clip(data.overnight(work) - data.since_open, 0, None)
    since = np.minimum(data.since_open, hp["morning_cap"]) / 60
    repair = (dur[:, TI["quick_damage_repair"]] > 0).astype(float)
    cols = [np.ones(data.n), data.backlog(work), left_overnight, since, repair, work, data.weekend]
    if hp["daily_features"]:
        weekday = (data.weekday[:, None] == np.arange(1, 7)).astype(float)    # Tue..Sun vs Monday
        cols += [data.daily_load(work), data.work_so_far(work), data.hour, *weekday.T]
    return np.column_stack(cols)


def _fit_wait(data, theta, sl, hp):
    """Step 4: waiting model from the final durations (per staff level, so drivers can act differently)."""
    dur = _durations(data, theta, sl)
    work = _work(dur, data.staff)
    wait = np.clip(data.open_used(data.y) - work, 0, None)
    if hp["wait_model"] == "table":
        return "table", _wait_table(data.staff, data.backlog(work), wait, hp)
    return "linear", _fit_wait_linear(data, _wait_features(data, dur, work, hp), wait, hp)


# waiting drivers per staff level: |r| >= 0.1 and p < 0.05 / 48 (Bonferroni over 16 features x 3 staff levels)
# against the true waiting time on validation; column 0 (intercept) is always used
WAIT_NAMES = ["intercept", "backlog", "overnight_left", "since_open", "repair", "own_work", "weekend", "daily_load",
              "work_so_far", "hour", "tue", "wed", "thu", "fri", "sat", "sun"]
WAIT_SELECTED = {
    1: ["backlog", "overnight_left", "since_open", "weekend", "hour", "sun"],
    2: ["backlog", "repair", "own_work", "daily_load", "work_so_far"],
    3: ["backlog", "since_open", "repair", "own_work", "weekend", "daily_load", "work_so_far", "hour", "fri", "sun"],
}


def _wait_cols(k, X, hp):
    if not hp["wait_select"] or X.shape[1] != len(WAIT_NAMES):
        return np.arange(X.shape[1])
    return np.array([0] + [WAIT_NAMES.index(n) for n in WAIT_SELECTED[k]])


def _fit_wait_linear(data, X, wait, hp, iters=80):
    """One quantile (median) regression of the waiting time per staff level, on that level's selected drivers."""
    out = {}
    for k in np.unique(data.staff):
        c, m = _wait_cols(int(k), X, hp), data.staff == k
        out[int(k)] = (c, _quantile_irls(X[m][:, c], wait[m], hp["wait_quantile"], iters=iters))
    return out


def _predict_wait_linear(data, X, coef):
    out = np.zeros(data.n)
    for k, (c, b) in coef.items():
        m = data.staff == k
        out[m] = np.clip(X[m][:, c] @ b, 0, None)
    return out


def _wait_apply(data, dur, work, wait_model, hp):
    kind, m = wait_model
    if kind == "table":
        return _wait_predict(data.staff, data.backlog(work), m)
    return _predict_wait_linear(data, _wait_features(data, dur, work, hp), m)


def fit(train, branches, hp=None):
    hp = {**HP, **(hp or {})}
    staff_map = branches.groupby("branch").staff_on_duty.first()
    data, theta, sl = _fit_durations(train, branches, staff_map, hp)
    return {"hp": hp, "theta": theta, "slices": sl, "staff": staff_map.to_dict(),
            "wait_model": _fit_wait(data, theta, sl, hp)}


def coefficient_table(model):
    """Fitted task coefficients as a table: task, covariate, coefficient (minutes per unit; intercept = base minutes)."""
    names = covariate_names(model["hp"])
    return pd.DataFrame([(t, c, float(b)) for t in TASKS for c, b in zip(names[t], model["theta"][model["slices"][t]])],
                        columns=["task", "covariate", "coefficient"])


def predict(model, df, branches):
    data = _Data(df, branches, pd.Series(model["staff"]), model["hp"])
    dur = _durations(data, model["theta"], model["slices"])
    work = _work(dur, data.staff)
    total = data.finish_total(_wait_apply(data, dur, work, model["wait_model"], model["hp"]) + work)
    return pd.DataFrame({"turnaround_id": df.turnaround_id.values, "total_turnaround_min": total,
                         **{f"dur_{t}": dur[:, TI[t]] for t in TASKS}})
