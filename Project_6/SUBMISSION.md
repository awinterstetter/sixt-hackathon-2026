# Submission: Project 6 – Task Time Detective, lmu-stads

**Team:** lmu-stads · **Demo:** ⏳ `submission/slides.pdf`

## Approach
Train only has the **total** time per car. We rebuild the path from return to ready and fit each task's formula so the
rebuilt totals match the real ones. For car $i$ (return time $t_i$, $k_i$ staff, $f_{it}=1$ if task $t$ was done):

$$d_{it} = f_{it}\cdot\max\left(0.5,\; x_{it}^\top \beta_t\right), \qquad
\text{work}_i = \max\Big(d_{i,\text{charge}},\; \min_{a \in \mathcal{A}(k_i)} \max_{p} \sum_{c:\,a(c)=p} \sum_{t \in c} d_{it}\Big)$$

$$W_i = \max\left(0,\; z_i^\top \gamma_{k_i}\right), \qquad \hat{y}_i = \mathrm{clock}_{b_i}\left(t_i,\; W_i + \text{work}_i\right)$$

- **Work:** task chains (interior→special, damage check→repair, exterior, refuel, tyre check) are split over the staff
  on duty; the car is ready when the busiest person is done. Charging runs on its own.
- **Waiting** $W_i$: median regression per staff level (= branch type: center 1, station 2, airport 3).
- **Clock:** only opening hours count; work pauses at closing and on closed days.

**Fitting (train.csv only; numpy, pandas, scipy; about 4 min, deterministic):**
1. Starting values: median total with the task minus without it.
2. Powell minimises the absolute total error on 8,000 seeded cars.
3. EM, 30 rounds: the unexplained time $g_i$ is shared among the critical-path tasks $\mathcal{C}_i$ and waiting by
   variance, $\hat d_{it} = \mu_{it} + g_i\,\sigma_t^2 / (\sum_{s \in \mathcal{C}_i} \sigma_s^2 + \sigma_W^2)$, then each
   $\beta_t$ is refit by least squares on $\hat d_{it}$.
4. Waiting: median regression of the remaining open minutes on the waiting drivers.

## Drivers
A driver is used if $|r| \ge 0.1$ with the true value on validation and Bonferroni-significant (tasks p < 0.05/199,
waiting p < 0.05/48). Selection used validation labels, so validation scores are slightly optimistic; all coefficients
are learned on train.

| Task | Drivers (r) |
|---|---|
| Quick exterior | exterior dirt +0.37, airport +0.30, queue +0.21, van +0.15, station −0.11 |
| Normal exterior | exterior dirt +0.62, rain +0.22, queue +0.14, van +0.12 |
| Special cleaning | smoking +0.83, premium +0.29 |
| Interior | child seat +0.65, interior dirt +0.58, rental days +0.19, km +0.16, airport +0.11 |
| Damage check / Repair | damage reported +0.36 / +0.57 |
| Refuel | airport +0.80, station −0.35, SUV +0.24, van +0.19 |
| Charging | battery −0.96 |
| Tyre check | airport −0.84 |
| Waiting, center | Sunday +0.78, overnight drop-offs +0.68, weekend +0.43, time since opening −0.42, hour +0.26, backlog +0.16 |
| Waiting, station / airport | backlog +0.56 / +0.58, repair +0.31, daily load ~+0.28, work so far ~+0.27, own work ~+0.18; airport also time of day, Fri, Sun, weekend |

## Results (validation week, 2,791 cars)
**Total MAE 13.57** (baseline 98.6) · **per-task WAPE 0.140** (baseline 0.549) · avg task MAE 3.87 · no negative durations.

| Task | Cars | True mean (min) | Predicted mean | MAE |
|---|---|---|---|---|
| Quick exterior | 1,575 | 13.3 | 12.2 | 1.77 |
| Normal exterior | 1,216 | 19.2 | 15.5 | 4.00 |
| Special cleaning | 277 | 30.0 | 27.6 | 2.41 |
| Interior | 2,791 | 16.4 | 15.5 | 1.73 |
| Damage check | 1,394 | 2.4 | 2.9 | 0.57 |
| Refuel | 410 | 13.5 | 10.8 | 2.98 |
| Charging | 526 | 24.8 | 22.3 | 2.78 |
| Repair | 71 | 68.6 | 62.3 | 12.73 |
| Tyre check | 29 | 26.1 | 24.1 | 5.90 |

- **Plausible (min):** child seat +6.0 (interior), smoking +3.6 (special), damage reported +20.5 (repair),
  0.34 per % missing battery, refuel +5.9 and tyre check −12.3 at airports.
- **Not plausible:** van −8.8 (normal exterior); damage check +3.5 when damage reported (true +0.3).
- **Limits:** even the true task durations give a total MAE of about 13 through our process rule, so the remaining
  error is waiting. Repair and tyre check are rare and least reliable.

## How to run
Python 3.11+, no environment variables.
```
pip install -r requirements.txt
cd starter
python run.py --repeat     # fit on train, predict validation, check two runs are identical
python evaluate.py predictions.csv ../data/validation_labels.csv --task-truth ../data/validation_labels.csv \
    --flags ../data/validation.csv --train-truth ../data/validation_labels.csv
```

## Next steps
- Better waiting model: waits near closing and arrival bursts at city centers drive most of the remaining error.
- For SIXT: per-task times per branch type for staff planning and "car ready at" estimates.
