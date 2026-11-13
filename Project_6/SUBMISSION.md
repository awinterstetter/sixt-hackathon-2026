# Submission: Project 6 – Task Time Detective, lmu-stads

## Authors
**Team name:** lmu-stads

**Project:** Project 6: Task Time Detective – how long does each turnaround task take?

## Problem and approach
Train only has the **total** time per car, so each task's duration is a latent quantity. We model the process that turns task
durations into the total, and fit the task formulas through it:

```
total = clock_open(return_ts, W + work)
work  = max(charging, best split of the chains over staff_on_duty)
chains: interior -> special | damage check -> repair | exterior | refuel | tyre check
```

- **Task duration** = linear formula of its drivers, at least 0.5 min, 0 if the task was not done.
- **Work:** 1 person does the chains one after another; 2–3 people split them, the car is ready when the busiest is done.
  Charging runs on its own. Order of work and staff come from the data (`branch_data.csv`).
- **Waiting W:** median regression per staff level (= branch type: airport 3, station 2, center 1) on backlog, overnight
  drop-offs, time since opening, own work, daily load, hour, weekday.
- **clock_open:** only open minutes count; work pauses at closing and on closed days.

## How the model is fitted (train.csv only)
| Stage | What | Why |
|---|---|---|
| 1. Starting values | per task: median total with the task minus without it | simple, intuitive start |
| 2. Structural fit | Powell (SciPy) minimises mean absolute total error through the full process, 8,000 seeded cars | fits all formulas jointly via the total |
| 3. EM, 30 rounds | **E:** per car, split the unexplained time among the critical-path tasks and waiting by their variance. **M:** least squares per task on these durations | gives every task its own target per car, so time cannot shift freely between tasks |
| 4. Structural fit again | Powell, starting from the EM values, all parameters free | re-aligns with the total |
| 5. Waiting model | fitted on the final durations | |

At prediction time: formulas → work → waiting → clock. The E-step is only used in training.

## Covariate selection
A driver enters a task's formula if its correlation with the **true task duration on validation** (performed tasks only)
is |r| ≥ 0.1 **and** significant after **Bonferroni correction**: p < 0.05 / 199, since we ran 199 task × variable tests.
Without the correction about 10 drivers would be expected to pass by chance. Branch type enters as the staff dummies
(airport = 3 staff, station = 2, center = 1), since they are the same split.

| Task | Drivers (r on validation) | Passed p < 0.05 but dropped by Bonferroni |
|---|---|---|
| Quick exterior | exterior dirt +0.37, airport +0.30, queue +0.21, van +0.15, station −0.11 | – |
| Normal exterior | exterior dirt +0.62, rain +0.22, queue +0.14, van +0.12 | – |
| Special cleaning | smoking +0.83, premium +0.29 | pet +0.17, child seat +0.13 |
| Interior | child seat +0.65, interior dirt +0.58, rental days +0.19, km +0.16, airport +0.11 | – |
| Damage check | damage reported +0.36 | – |
| Refuel | airport +0.80, station −0.35, SUV +0.24, van +0.19 | rain +0.14, weekend +0.13 |
| Charging | battery −0.96 | weekend +0.10 |
| Repair | damage reported +0.57 | – |
| Tyre check | airport −0.84 | return hour −0.42 |

Fuel level is not a driver of refuel time (r = +0.08): refuel takes a fixed time per branch type.

**Caveat:** selection used validation labels, so validation scores are slightly optimistic. Coefficients are learned on
train only. `HP["covariates"] = "base"` gives the earlier, smaller set (strong drivers only, chosen informally).

## Models and why
All models run locally (numpy, pandas, scipy), open, no API cost. Scored on validation with `evaluate.py`.

| Model | Total MAE | Per-task WAPE | Avg task MAE |
| --- | --- | --- | --- |
| Structural fit + EM, Bonferroni-selected covariates | ⏳ | ⏳ | ⏳ |

## Results
Final model, validation week (2,791 cars), fitted on train only:

| Task | Cars | True mean (min) | Predicted mean | MAE |
|---|---|---|---|---|
| Quick exterior | 1,575 | 13.3 | 8.3 | 4.99 |
| Normal exterior | 1,216 | 19.2 | 21.9 | 3.02 |
| Special cleaning | 277 | 30.0 | 31.0 | 1.30 |
| Interior | 2,791 | 16.4 | 16.9 | 1.66 |
| Damage check | 1,394 | 2.4 | 2.2 | 0.65 |
| Refuel | 410 | 13.5 | 13.2 | 1.90 |
| Charging | 526 | 24.8 | 24.7 | 1.62 |
| Repair | 71 | 68.6 | 86.0 | 20.20 |
| Tyre check | 29 | 26.1 | 29.4 | 6.50 |

Total MAE 13.36 (baseline 98.6), per-task WAPE 0.164 (baseline 0.549), no negative durations, two runs identical.

**Plausible effects** (learned coefficients): child seat +7.8 min interior, smoking +7.2 min special cleaning, +2.3 min per
dirt level normal exterior, charging 0.36 min per % missing battery, damage reported +24.5 min repair, tyre check 15 min
shorter at airports, refuel 9 min longer at airports.

**Not plausible:** the structural fit only sees the total, so it can shift time between tasks on the same car. Quick
exterior comes out too short (van −5.6, airport −4.5 min, where vans and airports are slower in reality) and repair too
long (86 vs 68.6 min).

**Limits:** total MAE is far from the "strong" bar of 5. Even the **true** task durations put through our process rule give
a total MAE of about 13, so the remaining error is waiting time (queues near closing, city centers), not task durations.
Repair and tyre check are rare, so their estimates are the least reliable.

## How to run
Python 3.11+ (tested with 3.13), no environment variables.
```
pip install -r requirements.txt          # numpy 2.3.4, pandas 2.3.3, scipy 1.16.1
cd starter
python run.py --repeat                   # fits on train, predicts validation, checks two runs are identical (~3.5 min each)
python evaluate.py predictions.csv ../data/validation_labels.csv --task-truth ../data/validation_labels.csv \
    --flags ../data/validation.csv --train-truth ../data/validation_labels.csv
```
Deterministic: fixed seed (`HP["seed"] = 42`), two runs give identical predictions.

## Demo
⏳ `submission/slides.pdf`

## Next steps
- Better waiting model: waits near closing and arrival bursts at city centers drive most of the remaining error.
- Multi-start for the structural fit; faster work computation.
- For SIXT: per-task times per branch type support staff planning and "car ready at" estimates for customers.
