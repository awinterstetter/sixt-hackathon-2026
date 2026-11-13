"""Descriptive plots: which tasks drive total turnaround time?

Run from Project_6/:  .venv/Scripts/python.exe analysis/task_totals.py
Writes PNGs into analysis/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

TASKS = ["quick_exterior_cleaning", "normal_exterior_cleaning", "special_cleaning", "interior_cleaning",
         "damage_check", "refuel", "electric_charging", "quick_damage_repair", "tyre_check"]
SHORT = {"quick_exterior_cleaning": "quick ext.", "normal_exterior_cleaning": "normal ext.",
         "special_cleaning": "special", "interior_cleaning": "interior", "damage_check": "damage chk",
         "refuel": "refuel", "electric_charging": "charging", "quick_damage_repair": "repair",
         "tyre_check": "tyre chk"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SAME_DAY_MIN = 120  # returned at least this many minutes before closing -> treated as "no overnight wait"

# Colour scheme, the same in every plot (palette = reference instance from the dataviz skill):
#   branch type -> blue / aqua / violet (used only for branches)
#   data source -> orange = train, yellow = validation week (used only for data source)
TYPE_COLOUR = {"AIRPORT": "#2a78d6", "STATION": "#1baf7a", "CENTER": "#6250d6"}
TYPE_LABEL = {"AIRPORT": "airport (3 staff)", "STATION": "station (2 staff)", "CENTER": "centre (1 staff)"}
SIXT_ORANGE = "#ff5f00"   # Sixt orange; drawn at 75% opacity in box plots
ORANGE = "#fe863f"        # how SIXT_ORANGE at 75% looks on the page: used for every solid orange
TRAIN_C, VAL_C, GREY = SIXT_ORANGE, "#eda100", "#898781"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
# approximate locations, inferred from the branch names (the data is synthetic)
BRANCH_LONLAT = {"DUS_AIRPORT": (6.77, 51.29), "FRA_AIRPORT": (8.56, 50.04), "MUC_AIRPORT": (11.79, 48.35),
                 "CGN_STATION": (6.96, 50.94), "HAM_STATION": (10.01, 53.55), "LEJ_STATION": (12.38, 51.35),
                 "BER_CENTER": (13.40, 52.52), "NUE_CENTER": (11.08, 49.45), "STR_CENTER": (9.18, 48.78)}


def branch_type(b):
    return b.split("_")[1]


def branch_order(branches):
    """Airports, then stations, then centres; alphabetical within type."""
    return sorted(branches, key=lambda b: (list(TYPE_COLOUR).index(branch_type(b)), b))


def branch_colour(branches):
    """One shade of its type colour per branch (darker -> lighter within each type)."""
    from matplotlib.colors import to_rgb
    out = {}
    for typ, base in TYPE_COLOUR.items():
        bs = [b for b in branch_order(branches) if branch_type(b) == typ]
        for b, f in zip(bs, [-0.35, 0.0, 0.4]):
            c = np.array(to_rgb(base))
            out[b] = tuple(c * (1 + f) if f < 0 else c + (1 - c) * f)
    return out
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def to_min(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def add_open_flag(df, branches):
    """True if the branch was open at return_ts (from branch_data opening hours)."""
    b = branches.copy()
    b["weekday"] = b["day"].map({d: i for i, d in enumerate(DAYS)})
    closed = b["open"].eq("closed")
    b["open_min"] = np.where(closed, -1, b["open"].where(~closed, "0:0").map(to_min))
    b["close_min"] = np.where(closed, -1, b["close"].where(~closed, "0:0").map(to_min))
    df = df.merge(b[["branch", "weekday", "open_min", "close_min", "staff_on_duty"]],
                  on=["branch", "weekday"], how="left")
    t = df["return_ts"].dt.hour * 60 + df["return_ts"].dt.minute
    df["branch_open"] = (t >= df["open_min"]) & (t < df["close_min"])
    df["min_to_close"] = np.where(df["branch_open"], df["close_min"] - t, np.nan)
    # returns late in the day are often carried over to the next morning; keep only clear same-day cases
    df["same_day"] = df["branch_open"] & (df["min_to_close"] >= SAME_DAY_MIN)
    df["return_status"] = np.select([~df["branch_open"], ~df["same_day"]],
                                    ["closed at return", f"open, <{SAME_DAY_MIN // 60}h before closing"],
                                    f"open, >={SAME_DAY_MIN // 60}h before closing")
    return df


def box(ax, groups, labels, colors):
    bp = ax.boxplot(groups, tick_labels=labels, showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=GREY),
                    capprops=dict(color=GREY))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor(c)


def fig1_with_without(tr):
    """Total with vs without each task: all returns vs returns while branch open."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (title, d) in zip(axes, [("All returns (incl. overnight waiting)", tr),
                                      (f"Only returns >= {SAME_DAY_MIN // 60}h before closing", tr[tr.same_day])]):
        groups, labels, colors = [], [], []
        for t in TASKS:
            for flag, c in [(0, GREY), (1, TRAIN_C)]:
                g = d.loc[d[t] == flag, "total_turnaround_min"]
                groups.append(g.values if len(g) else [np.nan])
                labels.append(""); colors.append(c)
        box(ax, groups, labels, colors)
        ax.set_xticks([2 * i + 1.5 for i in range(len(TASKS))], [SHORT[t] for t in TASKS], rotation=35, ha="right")
        ax.set_yscale("log"); ax.set_title(title)
        ax.set_ylabel("total turnaround (min, log scale)")
    axes[0].plot([], [], color=GREY, lw=8, alpha=.75, label="task not performed")
    axes[0].plot([], [], color=TRAIN_C, lw=8, alpha=.75, label="task performed")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Train: total turnaround with vs without each task (boxes = IQR, whiskers = 1.5 IQR, no outliers)",
                 x=0.01, ha="left", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "01_total_with_without_task.png", dpi=150); plt.close(fig)


def fig2_uplift(tr):
    """Median uplift in total when task performed, same-day returns only."""
    d = tr[tr.same_day]
    up = {t: d.loc[d[t] == 1, "total_turnaround_min"].median() - d.loc[d[t] == 0, "total_turnaround_min"].median()
          for t in TASKS if d[t].nunique() == 2}
    s = pd.Series(up).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([SHORT[t] for t in s.index], s.values, color=ORANGE, height=0.6)
    for i, v in enumerate(s.values):
        ax.text(v + (0.6 if v >= 0 else -0.6), i, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                color=INK2, fontsize=9)
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("median total with task − median total without (min)")
    ax.set_title("Extra total time when a task is done (median, same-day returns)")
    ax.text(0, -0.2, f"Same-day = returned >= {SAME_DAY_MIN // 60}h before closing. Raw difference, not causal:\n"
            "tasks co-occur (exactly one exterior). Interior omitted (always performed).",
            transform=ax.transAxes, fontsize=8, color=GREY, va="top")
    fig.tight_layout(); fig.savefig(OUT / "02_median_uplift_same_day.png", dpi=150); plt.close(fig)


def fig3_combos(tr, top=15):
    """Total by most frequent task combinations, same-day returns."""
    d = tr[tr.same_day].copy()
    d["combo"] = d[TASKS].apply(lambda r: " + ".join(SHORT[t] for t in TASKS if r[t] == 1), axis=1)
    counts = d["combo"].value_counts().head(top)
    order = d[d.combo.isin(counts.index)].groupby("combo").total_turnaround_min.median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 7))
    groups = [d.loc[d.combo == c, "total_turnaround_min"].values for c in order]
    bp = ax.boxplot(groups, orientation="horizontal", showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=GREY),
                    capprops=dict(color=GREY))
    for p in bp["boxes"]:
        p.set_facecolor(TRAIN_C); p.set_alpha(0.75); p.set_edgecolor(TRAIN_C)
    ax.set_yticks(range(1, len(order) + 1), [f"{c}  (n={counts[c]:,})" for c in order], fontsize=8.5)
    ax.set_xlabel("total turnaround (min)")
    fig.suptitle(f"Total by the {top} most common task combinations (same-day returns)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "03_total_by_combination.png", dpi=150); plt.close(fig)


# Work chains in the order the tasks are done; charging runs in parallel to everything, so it goes last.
CHAIN_LAYOUT = [("damage check → repair", ["damage_check", "quick_damage_repair"]),
                ("exterior (quick or normal)", ["quick_exterior_cleaning", "normal_exterior_cleaning"]),
                ("interior → special", ["interior_cleaning", "special_cleaning"]),
                ("refuel", ["refuel"]), ("tyre check", ["tyre_check"]),
                ("charging (parallel)", ["electric_charging"])]


def chain_positions(within=0.72, between=1.35):
    """x positions: tasks of one chain close together, a wider gap between chains."""
    order, pos, spans, x = [], [], [], 1.0
    for name, ts in CHAIN_LAYOUT:
        start = x
        for t in ts:
            order.append(t); pos.append(x); x += within
        spans.append((name, start, x - within))
        x += between - within
    return order, pos, spans


def draw_chain_brackets(ax, spans, y=-0.13):
    """Grey line plus chain name under each group of tick labels."""
    for name, a, z in spans:
        ax.annotate("", xy=(a - 0.3, y), xytext=(z + 0.3, y), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=1.2))
        ax.text((a + z) / 2, y - 0.02, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color=INK2)


def fig4_true_durations(lab, va):
    """Validation week: true per-task durations where performed, grouped by work chain."""
    order, pos, spans = chain_positions()
    fig, ax = plt.subplots(figsize=(12, 9))
    groups = [lab.loc[va[t] == 1, t].values for t in order]
    bp = ax.boxplot(groups, positions=pos, showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=GREY),
                    capprops=dict(color=GREY))
    for p in bp["boxes"]:
        p.set_facecolor(SIXT_ORANGE); p.set_alpha(0.75); p.set_edgecolor(SIXT_ORANGE)
    ax.set_xticks(pos, [f"{SHORT[t]}\n(n={len(g)})" for t, g in zip(order, groups)])
    draw_chain_brackets(ax, spans, y=-0.085)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(False)
    ax.set_ylabel("true task duration (min)")
    ax.set_title("Validation week: true duration of each task by work chain, where performed")
    fig.tight_layout(); fig.savefig(OUT / "04_true_task_durations_validation.png", dpi=150); plt.close(fig)


def fig4a_durations_by_branch(lab, va):
    """Validation week: true per-task durations by branch (one panel per task), grouped and coloured by type."""
    branches = branch_order(va.branch.unique())
    colours = [TYPE_COLOUR[branch_type(b)] for b in branches]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for ax, t in zip(axes.flat, TASKS):
        m = va[t] == 1
        groups = [lab.loc[m & (va.branch == b), t].values for b in branches]
        groups = [g if len(g) else [np.nan] for g in groups]
        box(ax, groups, [""] * len(branches), colours)
        for x in (3.5, 6.5):
            ax.axvline(x, color="#c3c2b7", lw=0.8, ls=":")
        for i, g in enumerate(groups, start=1):
            n = int(np.sum(~np.isnan(g)))
            ax.text(i, 0.02, f"{n}", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color=GREY)
        ax.set_title(f"{SHORT[t]}  (n={int(m.sum())})", fontsize=10)
        ax.set_xticks(range(1, len(branches) + 1), branches, rotation=45, ha="right", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("true duration (min)")
    fig.suptitle("Validation week: true task duration by branch (grey numbers = rows per branch; y-axes differ)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75) for c in TYPE_COLOUR.values()]
    fig.legend(handles, list(TYPE_LABEL.values()), frameon=False,
               loc="upper right", ncol=3, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(OUT / "04a_true_task_durations_by_branch.png", dpi=150); plt.close(fig)


def fig4b_branch_medians(lab, va):
    """Validation week: overall per-task boxes with each branch's median overlaid as a marker.

    Tasks are grouped by work chain (tasks within a chain are done one after another); charging runs in
    parallel to everything and sits on the far right. Boxes within a chain are drawn closer together.
    """
    chains = [("damage check → repair", ["damage_check", "quick_damage_repair"]),
              ("exterior (quick or normal)", ["quick_exterior_cleaning", "normal_exterior_cleaning"]),
              ("interior → special", ["interior_cleaning", "special_cleaning"]),
              ("refuel", ["refuel"]), ("tyre check", ["tyre_check"]),
              ("charging (parallel)", ["electric_charging"])]
    within, between = 0.72, 1.35
    order, pos, spans, x = [], [], [], 1.0
    for name, ts in chains:
        start = x
        for t in ts:
            order.append(t); pos.append(x); x += within
        spans.append((name, start, x - within))
        x += between - within
    # colour = shade of the branch-type colour (one per branch); shape = branch type
    type_marker = {"AIRPORT": "o", "STATION": "s", "CENTER": "^"}
    branches = branch_order(va.branch.unique())
    shade = branch_colour(branches)
    colours = [shade[b] for b in branches]
    markers = [type_marker[branch_type(b)] for b in branches]
    offsets = np.linspace(-0.25, 0.25, len(branches))

    fig, ax = plt.subplots(figsize=(13, 7))
    groups = [lab.loc[va[t] == 1, t].values for t in order]
    bp = ax.boxplot(groups, positions=pos, showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=GREY),
                    capprops=dict(color=GREY))
    for p in bp["boxes"]:
        p.set_facecolor("#e1e0d9"); p.set_alpha(0.75); p.set_edgecolor("#e1e0d9")
    ax.set_xticks(pos, [f"{SHORT[t]}\n(n={len(g)})" for t, g in zip(order, groups)])
    for b, dx, c, mk in zip(branches, offsets, colours, markers):
        med = [lab.loc[(va[t] == 1) & (va.branch == b), t].median() for t in order]
        ax.scatter(np.array(pos) + dx, med, color=c, marker=mk, s=34, zorder=3,
                   edgecolors=INK2, linewidths=0.5, label=b)
    # chain brackets below the tick labels
    for name, a, z in spans:
        ax.annotate("", xy=(a - 0.3, -0.13), xytext=(z + 0.3, -0.13), xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color="#c3c2b7", lw=1.2))
        ax.text((a + z) / 2, -0.15, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.5, color=INK2)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    ax.set_ylabel("true task duration (min)")
    ax.set_title("Validation week: true duration of each task by work chain, with branch medians")
    ax.legend(title="branch median", frameon=False, fontsize=8.5, title_fontsize=9, loc="upper right", ncol=3)
    ax.text(0, -0.23, "Boxes = all branches (IQR, whiskers 1.5 IQR). Markers = median per branch "
            "(circle = airport, square = station, triangle = centre); "
            "repair and tyre check have only 1-18 rows per branch.", transform=ax.transAxes, fontsize=8,
            color=GREY, va="top")
    fig.tight_layout(); fig.savefig(OUT / "04b_true_task_durations_branch_medians.png", dpi=150); plt.close(fig)


def fig5_task_counts(tr, va):
    """How often each task is performed: share of turnarounds, train vs validation week (counts in labels)."""
    order = sorted(TASKS, key=lambda t: tr[t].mean())
    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for dy, d, c, name in [(h / 2, tr, ORANGE, f"train (n={len(tr):,})"),
                           (-h / 2, va, VAL_C, f"validation week (n={len(va):,})")]:
        share = [100 * d[t].mean() for t in order]
        ax.barh(y + dy, share, height=h - 0.04, color=c, label=name)
        for yi, s, t in zip(y + dy, share, order):
            ax.text(s + 1, yi, f"{s:.1f}%  ({int(d[t].sum()):,})", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [SHORT[t] for t in order])
    ax.set_xlim(0, 118); ax.set_xlabel("share of turnarounds with the task (%)")
    ax.set_title("How often each task is performed")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "05_task_counts.png", dpi=150); plt.close(fig)


def fig6_branch_map(tr, va):
    """Map of the branches: colour = branch type (with staff), point area = turnarounds (train + validation)."""
    import json
    geo = json.loads((Path(__file__).resolve().parent / "data" / "germany_region.geojson").read_text())
    fig, ax = plt.subplots(figsize=(8, 9.5))
    for f in geo["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        de = f["properties"]["iso"] == "DEU"
        for poly in polys:
            x, y = np.array(poly[0]).T
            ax.fill(x, y, facecolor="#f0efec" if de else "#f7f7f5", edgecolor="#c3c2b7" if de else GRID,
                    linewidth=1.0 if de else 0.6, zorder=1)
    n = pd.concat([tr.branch, va.branch]).value_counts()
    scale = 2200 / n.max()
    left = {"DUS_AIRPORT": 8, "CGN_STATION": -10}   # west-side labels (DUS and CGN are close together)
    for b in branch_order(n.index):
        lon, lat = BRANCH_LONLAT[b]
        ax.scatter(lon, lat, s=n[b] * scale, color=TYPE_COLOUR[branch_type(b)], alpha=0.8,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
        r = np.sqrt(n[b] * scale) / 2 + 5          # circle radius in points + gap
        dx, dy, ha = (-r, left[b], "right") if b in left else (r, 0, "left")
        ax.annotate(f"{b}\n{n[b]:,}", (lon, lat), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center", fontsize=8.5, color=INK, zorder=4)
    type_handles = [plt.scatter([], [], s=120, color=TYPE_COLOUR[t], alpha=0.8, edgecolors=SURFACE)
                    for t in TYPE_COLOUR]
    leg = ax.legend(type_handles, list(TYPE_LABEL.values()), title="branch type (staff on duty)",
                    frameon=False, loc="upper left", fontsize=9, title_fontsize=9.5, labelspacing=1.0)
    ax.add_artist(leg)
    sizes = [2000, 5000, 10000]
    size_handles = [plt.scatter([], [], s=v * scale, color="none", edgecolors=GREY, linewidths=1.2) for v in sizes]
    ax.legend(size_handles, [f"{v:,}" for v in sizes], title="turnarounds (train + validation)", frameon=False,
              loc="lower left", fontsize=9, title_fontsize=9.5, labelspacing=2.2, borderpad=1.2,
              handletextpad=2.2)
    ax.set_xlim(4.6, 15.6); ax.set_ylim(47.1, 55.2)
    ax.set_aspect(1 / np.cos(np.radians(51.2)))
    ax.axis("off")
    ax.set_title("Branches: type, staff and number of turnarounds")
    ax.text(0, -0.02, "Locations approximate, inferred from branch names (the data is synthetic). "
            "Borders: Natural Earth.", transform=ax.transAxes, fontsize=7.5, color=GREY, va="top")
    fig.tight_layout(); fig.savefig(OUT / "06_branch_map.png", dpi=150); plt.close(fig)


# Covariates of each per-task regression in starter/solution.py (_design, with HP["queue_in_cleaning"] = True).
# Keep in sync with solution.py by hand.
MODEL_COVARIATES = {   # HP["covariates"] = "corr" (default), queue_in_cleaning = True
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
}
CANDIDATE_COVARIATES = ["intercept", "interior_dirt", "exterior_dirt", "child_seat", "pet", "smoking_reported",
                        "damage_reported", "fuel_level_pct", "battery_soc_pct", "km_driven / 100", "rental_days",
                        "van", "suv", "premium", "powertrain", "rain", "queue_length", "staff = 2", "staff = 3",
                        "return_hour", "weekday"]


def fig7_model_covariates(tr):
    """Which covariates enter which per-task regression (orange = used), with the fitted coefficient in each box.

    Coefficients come from analysis/data/model_coefficients.json (written by analysis/fit_coefficients.py);
    without that file the boxes are drawn without values.
    """
    import json
    from matplotlib.colors import ListedColormap
    used_c = ORANGE
    coef_file = Path(__file__).resolve().parent / "data" / "model_coefficients.json"
    coefs = json.loads(coef_file.read_text()) if coef_file.exists() else {}
    rows, cols = TASKS, CANDIDATE_COVARIATES
    M = np.array([[c in MODEL_COVARIATES[t] for c in cols] for t in rows], float)
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.imshow(M, cmap=ListedColormap(["#f0efec", used_c]), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), [f"{SHORT[t]} ({int(tr[t].sum()):,})" for t in rows])
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(False); ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0); ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, t in enumerate(rows):
        ax.text(len(cols) - 0.3, i, f"{len(MODEL_COVARIATES[t]) - 1}", va="center", ha="left", fontsize=9, color=INK2)
        for j, c in enumerate(cols):
            if M[i, j] and c in coefs.get(t, {}):
                v = coefs[t][c]
                ax.text(j, i, f"{v:+.2f}" if c != "intercept" else f"{v:.1f}", ha="center", va="center",
                        fontsize=8, color=INK, fontweight="bold" if c == "intercept" else "normal")
    ax.text(len(cols) - 0.3, -0.9, "covariates", va="center", ha="left", fontsize=8, color=GREY)
    ax.set_title("Covariates in each per-task regression, with fitted effects in minutes (starter/solution.py)"
                 if coefs else "Covariates in each per-task regression (starter/solution.py)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=used_c), plt.Rectangle((0, 0), 1, 1, color="#f0efec")],
              ["used", "not used"], frameon=False, loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize=9)
    ax.text(0, -0.42, "Values: intercept = base minutes; others = minutes added per unit (dirt score, flag = 1, "
            "queued car, % battery missing). battery_soc_pct enters as 100 − SOC (missing → 50).\n"
            "staff = 2 / 3 are dummies against 1 staff (= station / airport vs centre). Count excludes the intercept. "
            "Each duration is clipped at 0.5 min.\n(n) after the task = turnarounds in train.csv where the task was "
            "performed (the rows its regression is fitted on).", transform=ax.transAxes, fontsize=8, color=GREY, va="top")
    fig.tight_layout(); fig.savefig(OUT / "07_per_task_model_covariates.png", dpi=150); plt.close(fig)


def fig8_covariate_correlation(lab, va):
    """Validation week: correlation of each covariate with each task's TRUE duration (rows where the task was done).

    Numeric / yes-no covariates: Pearson r (signed). Categorical covariates: correlation ratio eta (0..1, unsigned).
    Same rows and columns as plot 7 (intercept dropped).
    """
    from matplotlib.colors import LinearSegmentedColormap
    staff = va.branch.str.split("_").str[1].map({"AIRPORT": 3, "STATION": 2, "CENTER": 1})
    numeric = {"interior_dirt": va.interior_dirt, "exterior_dirt": va.exterior_dirt, "child_seat": va.child_seat,
               "pet": va.pet, "smoking_reported": va.smoking_reported, "damage_reported": va.damage_reported,
               "fuel_level_pct": va.fuel_level_pct, "battery_soc_pct": va.battery_soc_pct,
               "km_driven / 100": va.km_driven / 100, "km_driven": va.km_driven,
               "rental_days": va.rental_days, "queue_length": va.queue_length, "staff = 2": staff == 2,
               "staff = 3": staff == 3, "return_hour": va.return_hour,
               "van": va.vehicle_class == "van", "suv": va.vehicle_class == "suv",
               "premium": va.vehicle_class == "premium", "rain": va.weather == "rain"}
    categorical = {"vehicle_class": va.vehicle_class, "powertrain": va.powertrain, "weather": va.weather,
                   "weekday": va.weekday}
    cols = sorted((c for c in CANDIDATE_COVARIATES if c != "intercept"), key=str.lower)   # alphabetical

    def eta(y, g):
        y, g = pd.Series(y), pd.Series(g).values
        ss_tot = ((y - y.mean()) ** 2).sum()
        ss_b = sum(len(v) * (v.mean() - y.mean()) ** 2 for _, v in y.groupby(g))
        return np.sqrt(ss_b / ss_tot) if ss_tot > 0 and pd.Series(g).nunique() > 1 else np.nan

    R = np.full((len(TASKS), len(cols)), np.nan)
    for i, t in enumerate(TASKS):
        m = (va[t] == 1).values
        y = lab.loc[m, t].values
        for j, c in enumerate(cols):
            if c in numeric:
                x = pd.Series(np.asarray(numeric[c], float)[m])
                ok = x.notna().values
                if ok.sum() > 10 and x[ok].nunique() > 1:
                    R[i, j] = np.corrcoef(x[ok], y[ok])[0, 1]
            else:
                R[i, j] = eta(y, np.asarray(categorical[c])[m])

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", ORANGE])
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.imshow(np.nan_to_num(R), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i, t in enumerate(TASKS):
        for j, c in enumerate(cols):
            v = R[i, j]
            used = c in MODEL_COVARIATES[t]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=GREY)
                continue
            txt = (f"η {v:.2f}" if c in categorical else f"{v:+.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if abs(v) > 0.6 else INK, fontweight="bold" if used else "normal")
            if used:   # outline = covariate used in solution.py for this task
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, edgecolor=INK, lw=1.2))
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(TASKS)), [f"{SHORT[t]} ({int(va[t].sum()):,})" for t in TASKS])
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(TASKS)), minor=True)
    ax.grid(False); ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02); cb.outline.set_visible(False)
    cb.set_label("correlation with true duration", color=INK2)
    ax.set_title("Validation week: correlation of each covariate with each task's true duration")
    ax.text(0, -0.42, "Pearson r for numeric / yes-no covariates; η (correlation ratio, 0-1, no sign) for categorical "
            "ones. Rows: only cars where the task was performed, (n) = rows in the validation week.\n"
            "Outlined bold cells = covariate used for that task in solution.py. '–' = not defined (covariate constant "
            "or missing for that task, e.g. fuel level for electric cars).", transform=ax.transAxes, fontsize=8,
            color=GREY, va="top")
    fig.tight_layout(); fig.savefig(OUT / "08_covariate_task_correlation.png", dpi=150); plt.close(fig)
    pd.DataFrame(R, index=TASKS, columns=cols).round(3).to_csv(OUT / "08_covariate_task_correlation.csv")


# typical change used to turn a coefficient into "minutes added", per covariate (binary covariates: 0 -> 1)
EFFECT_STEP = {"exterior_dirt": (4, "dirt 1 → 5"), "interior_dirt": (4, "dirt 1 → 5"),
               "queue_length": (6, "queue 0 → 6 cars"), "battery_soc_pct": (60, "battery 80% → 20%"),
               "rental_days": (13, "rental 1 → 14 days"), "km_driven / 100": (10, "+1,000 km")}


def fig9_effects_check(lab, va):
    """Does it make sense? Per task and covariate: effect learned from train totals (model) vs the effect a
    regression on the validation week's TRUE task durations finds, in minutes for a typical change."""
    import importlib.util, json
    here = Path(__file__).resolve().parent
    coef_file = here / "data" / "model_coefficients.json"
    if not coef_file.exists():
        print("fig9 skipped: run analysis/fit_coefficients.py first"); return
    fitted = json.loads(coef_file.read_text())
    spec = importlib.util.spec_from_file_location("solution", ROOT / "starter" / "solution.py")
    sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
    branches = pd.read_csv(DATA / "branch_data.csv")
    staff = va.branch.map(branches.groupby("branch").staff_on_duty.first()).values
    X = sol._design(va, staff, dict(sol.HP))

    rows = []   # (task, label, fitted effect, true effect)
    for t in TASKS:
        names = list(fitted[t])
        if X[t].shape[1] != len(names):
            print(f"fig9 skipped: {t} has {X[t].shape[1]} columns in solution.py but {len(names)} saved coefficients "
                  "- rerun analysis/fit_coefficients.py"); return
        m = (va[t] == 1).values
        true_coef = np.linalg.lstsq(X[t][m], lab.loc[m, t].values, rcond=None)[0]
        for j, n in enumerate(names):
            if n == "intercept":
                continue
            step, desc = EFFECT_STEP.get(n, (1, f"{n}: no → yes"))
            if n.startswith("staff"):
                desc = {"staff = 2": "station vs centre", "staff = 3": "airport vs centre"}[n]
            rows.append((t, desc, fitted[t][n] * step, true_coef[j] * step))

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.8))
    y = np.arange(len(rows))[::-1]
    for yi, (t, desc, f, tr_) in zip(y, rows):
        ax.plot([f, tr_], [yi, yi], color="#c3c2b7", lw=1.5, zorder=1)
        ax.scatter(tr_, yi, color=INK2, s=40, zorder=3)
        ax.scatter(f, yi, color=ORANGE, s=48, zorder=4, edgecolors=SURFACE, linewidths=0.8)
        bad_sign = np.sign(f) != np.sign(tr_) and max(abs(f), abs(tr_)) > 1
        off = abs(f - tr_) > max(2.0, 0.5 * abs(tr_))
        if bad_sign or off:
            ax.annotate("sign differs" if bad_sign else "size off", (max(f, tr_), yi), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8, color=ORANGE, fontweight="bold")
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_yticks(y, [f"{SHORT[t]}: {desc}" for t, desc, _, _ in rows], fontsize=8.5)
    prev = None
    for yi, (t, *_ ) in zip(y, rows):          # thin separators between tasks
        if prev is not None and t != prev:
            ax.axhline(yi + 0.5, color=GRID, lw=0.8)
        prev = t
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("minutes added to the task by the change")
    ax.scatter([], [], color=ORANGE, s=48, label="model (learned from train totals)")
    ax.scatter([], [], color=INK2, s=40, label="truth (regression on validation task times)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Does it make sense? Learned effect vs true effect, per task and covariate")
    ax.text(0, -0.9 / (0.42 * len(rows) + 1.8) * 1.8, "Flags: sign differs = opposite direction (and > 1 min); "
            "size off = differs by more than max(2 min, 50%). Same covariates as solution.py on both sides.",
            transform=ax.transAxes, fontsize=8, color=GREY, va="top")
    fig.tight_layout(); fig.savefig(OUT / "09_effects_model_vs_truth.png", dpi=150); plt.close(fig)


def _load_solution():
    import importlib.util
    spec = importlib.util.spec_from_file_location("solution", ROOT / "starter" / "solution.py")
    sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
    return sol


def _validation_parts(lab, va, sol, total, dur):
    """Split each car's total (wall-clock minutes) into work, waiting while open, and time the branch is closed."""
    branches = pd.read_csv(DATA / "branch_data.csv")
    d = sol._Data(va, branches, branches.groupby("branch").staff_on_duty.first(), dict(sol.HP))
    work = sol._work(dur, d.staff)
    used = d.open_used(total)                       # open minutes between return and ready
    wait = np.clip(used - work, 0, None)
    closed = np.clip(total - used, 0, None)
    situation = np.select([d.closed_return, d.to_close < SAME_DAY_MIN],
                          ["returned while closed", f"returned < {SAME_DAY_MIN // 60}h before closing"],
                          f"returned ≥ {SAME_DAY_MIN // 60}h before closing")
    return d, work, wait, closed, situation


def fig10_total_breakdown(lab, va):
    """What makes the total long? Mean total split into work, waiting while open and closed time, true vs model."""
    sol = _load_solution()
    pred_file = ROOT / "starter" / "pred_merged.csv"
    sources = [("true", lab.total_turnaround_min.values, lab[TASKS].values.astype(float))]
    if pred_file.exists():
        pr = pd.read_csv(pred_file)
        sources.append(("model", pr.total_turnaround_min.values, pr[[f"dur_{t}" for t in TASKS]].values))
    groups = [f"returned ≥ {SAME_DAY_MIN // 60}h before closing", f"returned < {SAME_DAY_MIN // 60}h before closing",
              "returned while closed"]
    parts = {}
    for name, total, dur in sources:
        _, work, wait, closed, sit = _validation_parts(lab, va, sol, total, dur)
        for g in groups:
            m = sit == g
            parts[(g, name)] = (work[m].mean(), wait[m].mean(), closed[m].mean(), m.sum())
    comps = [("work on the car", ORANGE), ("waiting while open", INK2), ("branch closed", "#c3c2b7")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    for ax, g in zip(axes, groups):
        names = [n for n, *_ in sources]
        for xi, n in enumerate(names):
            vals = parts[(g, n)][:3]
            bottom = 0
            for (lbl, c), v in zip(comps, vals):
                ax.bar(xi, v, bottom=bottom, color=c, width=0.6, edgecolor=SURFACE, linewidth=1.5)
                bottom += v
            ax.text(xi, bottom, f"{bottom:,.0f} min", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(len(names)), names)
        ax.set_title(f"{g}\n(n = {parts[(g, 'true')][3]:,})", fontsize=10)
        ax.set_ylim(0, max(sum(parts[(g, n)][:3]) for n in names) * 1.15)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("average minutes from return to ready")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in comps]
    fig.legend(handles, [l for l, _ in comps], frameon=False, loc="upper right", ncol=3, fontsize=9)
    fig.suptitle("What makes the total long: waiting and closed hours, not the work (validation week)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "true = validation labels; model = starter/pred_merged.csv. Work = time of the busiest worker "
             "(or charging); waiting = open minutes not explained by work;\n"
             "closed = minutes the branch was shut. Note the different y-axis per panel.", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94)); fig.savefig(OUT / "10_total_breakdown.png", dpi=150); plt.close(fig)


def summary_table(tr):
    rows = []
    for t in TASKS:
        for name, d in [("all", tr), ("same_day", tr[tr.same_day])]:
            w, wo = d.loc[d[t] == 1, "total_turnaround_min"], d.loc[d[t] == 0, "total_turnaround_min"]
            rows.append({"task": t, "subset": name, "share_performed": round(d[t].mean(), 3),
                         "median_with": round(w.median(), 1), "median_without": round(wo.median(), 1) if len(wo) else np.nan,
                         "mean_with": round(w.mean(), 1), "mean_without": round(wo.mean(), 1) if len(wo) else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "summary_with_without.csv", index=False)
    return t


if __name__ == "__main__":
    branches = pd.read_csv(DATA / "branch_data.csv")
    tr = add_open_flag(pd.read_csv(DATA / "train.csv", parse_dates=["return_ts"]), branches)
    va = pd.read_csv(DATA / "validation.csv")
    lab = pd.read_csv(DATA / "validation_labels.csv")
    assert (va.turnaround_id.values == lab.turnaround_id.values).all()

    print(f"train rows {len(tr):,}")
    print(tr.groupby("return_status").total_turnaround_min.describe(percentiles=[.5, .9, .99]).round(1))
    fig1_with_without(tr)
    fig2_uplift(tr)
    fig3_combos(tr)
    fig4_true_durations(lab, va)
    fig4a_durations_by_branch(lab, va)
    fig4b_branch_medians(lab, va)
    fig5_task_counts(tr, va)
    fig6_branch_map(tr, va)
    fig7_model_covariates(tr)
    fig8_covariate_correlation(lab, va)
    fig9_effects_check(lab, va)
    fig10_total_breakdown(lab, va)
    print(summary_table(tr).to_string(index=False))
    print(f"figures written to {OUT}")
