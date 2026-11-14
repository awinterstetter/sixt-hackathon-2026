# Submission: Project 6 – Task Time Detective, lmu-stads

**Team:** lmu-stads · **Slides:** [view online](https://awinterstetter.github.io/sixt-hackathon-2026/Project_6/slides/slides.html) · [PDF](slides.pdf)

## Approach
Train only has **total** times, so we rebuild each car's path from return to ready and fit the task formulas so the
rebuilt totals match the real ones (car $i$: return $t_i$, $k_i$ staff, $f_{it}=1$ if task $t$ done):

$$d_{it} = f_{it}\max(0.5,\, x_{it}^\top \beta_t), \quad
\mathrm{work}_i(\beta) = \max\Big(d_{i,\text{charge}},\, \min_{a \in \mathcal{A}(k_i)} \max_{p} \sum_{c:\,a(c)=p} \sum_{t \in c} d_{it}\Big), \quad
\hat{y}_i = \mathrm{clock}_{b_i}\big(t_i,\, \mathrm{wait}_i(\gamma) + \mathrm{work}_i(\beta)\big)$$

Task chains (interior→special, damage check→repair, exterior, refuel, tyre check) are split over the staff; the car is
ready when the busiest person is done. Charging runs in parallel. Waiting $\mathrm{wait}_i(\gamma) = \max(0, z_i^\top\gamma_{k_i})$ is a
median regression per staff level (center 1, station 2, airport 3). Only opening hours count.

**Fitting** (train.csv only, numpy/pandas/scipy, deterministic, ~3 min):
1. Starting values: median total with the task minus without it.
2. Powell minimises the absolute total error on 8,000 seeded cars.
3. EM, 30 rounds: each car's unexplained time $g_i$ is shared over its critical-path tasks and waiting by variance,
   $\hat d_{it} = \mu_{it} + g_i\,\sigma_t^2 / (\sum_{s \in \mathcal{C}_i} \sigma_s^2 + \sigma_{\mathrm{wait}}^2)$; each $\beta_t$ is refit by least squares.
4. Waiting model fit on the remaining open minutes.

## Final coefficients β (minutes)
Learned on train after EM. A driver is included if $|r| \ge 0.1$ with the true value and Bonferroni-significant; this
selection used validation labels, so validation scores are slightly optimistic. Dirt is scored 1–5, queue = cars
waiting at return, flags are 0/1, station/airport vs. city center.

| Task | Intercept | Drivers |
|---|--:|---|
| Quick exterior | 9.51 | exterior dirt +0.18, queue +0.16, station −0.30, airport +3.52, van −2.49 |
| Normal exterior | 13.42 | exterior dirt +0.92, queue +0.13, van −8.80, rain −0.19 |
| Special cleaning | 26.93 | smoking +3.59, premium +2.63 |
| Interior | 10.14 | interior dirt +1.29, child seat +5.98, queue +0.15, rental day +0.12, per 100 km +0.05, airport +0.61 |
| Damage check | 2.55 | damage reported +3.49 |
| Refuel | 6.29 | station +2.18, airport +5.91, SUV +0.07, van +6.35 |
| Charging | 2.89 | per % missing battery +0.34 |
| Repair | 49.48 | damage reported +20.47 |
| Tyre check | 31.31 | airport −12.26 |

**Waiting** (median regression per staff level; backlog, daily load and work in minutes of predicted work per staff,
time since opening in hours, capped at 4):

| Branch | Intercept | Drivers |
|---|--:|---|
| City center (1 staff) | 3.95 | backlog +0.15, overnight drop-offs left +0.02, time since opening −1.56/h, weekend +2.78, hour +0.39, Sunday +96.2 |
| Station (2 staff) | −0.59 | backlog +0.14, repair +13.54, own work +0.10, daily load +0.001, work so far −0.001 |
| Airport (3 staff) | 3.42 | backlog +0.16, time since opening −0.09/h, repair +19.30, own work +0.08, weekend +0.04, daily load −0.005, work so far +0.007, hour −0.26, Friday +0.61, Sunday +1.83 |

## Results (validation week, 2,791 cars)
**Total MAE 13.56 min** (baseline 98.6) · **per-task WAPE 0.140** (baseline 0.549) · avg task MAE 3.87 · no negative durations.

| Task | Cars | True mean | Pred. mean | MAE |
|---|--:|--:|--:|--:|
| Quick exterior | 1,575 | 13.3 | 12.2 | 1.76 |
| Normal exterior | 1,216 | 19.2 | 15.5 | 4.00 |
| Special cleaning | 277 | 30.0 | 27.6 | 2.41 |
| Interior | 2,791 | 16.4 | 15.5 | 1.73 |
| Damage check | 1,394 | 2.4 | 2.9 | 0.57 |
| Refuel | 410 | 13.5 | 10.8 | 2.98 |
| Charging | 526 | 24.8 | 22.3 | 2.77 |
| Repair | 71 | 68.6 | 62.3 | 12.68 |
| Tyre check | 29 | 26.1 | 24.1 | 5.90 |

- **Plausible:** child seat +6.0 min (interior), smoking +3.6 (special), damage reported +20.5 (repair), 0.34 min per %
  missing battery, at airports refuel +5.9 and tyre check −12.3.
- **Not plausible:** van −8.8 (normal exterior); damage check +3.5 when damage reported (true +0.3).
- **Limit:** even the true task times give a total MAE of ~13 through our process rule, so the remaining error is
  waiting. Repair and tyre check are rare and least reliable.

## How to run
Needs Python 3.11+. From a terminal, starting in the repository root:
```bash
cd Project_6
python3 -m venv .venv                 # one-time: create a virtual environment
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt       # numpy, pandas, scipy (pinned)

cd starter
python run.py --repeat                # fit on train.csv, predict validation.csv -> predictions.csv,
                                      # and check that two runs give identical values (~3 min per run)
python evaluate.py predictions.csv ../data/validation_labels.csv \
    --task-truth ../data/validation_labels.csv --flags ../data/validation.csv \
    --train-truth ../data/validation_labels.csv   # prints total MAE, per-task MAE and WAPE
```
Shortcut: `bash setup.sh` (in `Project_6/`) runs all of the above in one go.
To predict another file: `python run.py --test <file.csv> --out <predictions.csv>`.

## Next steps
Better waiting model (waits near closing and arrival bursts at city centers drive most of the remaining error);
per-task times per branch type for staff planning and "car ready at" estimates.
