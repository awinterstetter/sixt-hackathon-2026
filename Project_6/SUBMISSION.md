# Submission: Project 6 – Task Time Detective, lmu-stads

## Authors
**Team name:** lmu-stads

**Project:** Project 6: Task Time Detective – how long does each turnaround task take?

## Problem and approach
The training data only tells us how long a car took **in total**, not how long each task took. So we rebuild the
path from "car returned" to "car ready" step by step, and tune each task's formula until the rebuilt totals match the real ones.

**What happens to a returned car, in our model:**

1. **Each task gets a duration.** Every task has its own simple formula: a base time plus extra time per driver
   (e.g. more minutes for a dirtier car). A task takes at least 0.5 min, and 0 if it was not done on this car.
2. **The staff work through the tasks.** Some tasks must follow each other, so they form fixed sequences:
   interior → special cleaning, damage check → repair. Exterior, refuel and tyre check stand alone.
   - With **1 person**, everything is done one after another, so the times add up.
   - With **2–3 people**, the sequences are shared out as evenly as possible; the car is ready when the busiest person finishes.
   - **Charging** runs by itself in the background, so it only matters if it takes longer than the hands-on work.
   - Staff numbers and work order come from `branch_data.csv`.
3. **The car first waits in the queue.** Before anyone starts, the car usually waits (W). We predict this wait
   separately for each branch type (airport = 3 staff, station = 2, center = 1), based on: cars already waiting,
   cars dropped off overnight, time since opening, the car's own amount of work, how busy the day is, hour and weekday.
4. **Only opening hours count.** Waiting + work is laid onto the branch's opening hours: if the branch closes,
   the clock stops and continues when it reopens (skipping closed days).

In formulas, for car $i$ returned at time $t_i$ at a branch with $k_i$ staff:

**Task duration** ($f_{it} = 1$ if task $t$ was done, $x_{it}$ = its drivers):

$$d_{it} = f_{it}\cdot\max\left(0.5,\; x_{it}^\top \beta_t\right)$$

**Work** (chains $c$: interior→special, damage check→repair, exterior, refuel, tyre check; $\mathcal{A}(k_i)$ = all ways
to give the chains to $k_i$ people):

$$\text{work}_i = \max\Big(d_{i,\text{charge}},\; \min_{a \in \mathcal{A}(k_i)}\; \max_{p=1,\dots,k_i} \sum_{c:\,a(c)=p}\; \sum_{t \in c} d_{it}\Big)$$

**Waiting** (median regression per staff level, $z_i$ = backlog, overnight drop-offs left, time since opening, repair,
own work, weekend, daily load, work so far today, hour, weekday):

$$W_i = \max\left(0,\; z_i^\top \gamma_{k_i}\right)$$

**Total** ($\mathrm{clock}_b$ adds minutes only while branch $b$ is open):

$$\hat{y}_i = \mathrm{clock}_{b_i}\left(t_i,\; W_i + \text{work}_i\right)$$

## How the model is fitted (train.csv only)
| Stage | What | Why |
|---|---|---|
| 1. Starting values | per task: median total with the task minus without it | simple, intuitive start |
| 2. Structural fit | Powell (SciPy) minimises mean absolute total error through the full process, 8,000 seeded cars | fits all formulas jointly via the total |
| 3. EM, 30 rounds | **E:** per car, split the unexplained time among the critical-path tasks and waiting by their variance. **M:** least squares per task on these durations | gives every task its own target per car, so time cannot shift freely between tasks. **The last M-step gives the final task coefficients.** |
| 4. Waiting model | median regression on the EM durations | its target (open minutes − work) and its features (backlog, daily load) depend on the final durations |

**Stage 1, starting values** (1-staff cars far from closing; interior and quick exterior split the median of basic cars):

$$\beta_{t,0}^{(0)} = \operatorname{median}(y_i \mid f_{it}=1) - \operatorname{median}(y_i \mid f_{it}=0), \qquad \text{all other effects } 0$$

**Stage 2, structural fit** on a seeded sample $S$ of 8,000 cars, with a placeholder wait
$W_i^0 = \max(0,\ \omega_{k_i} + \omega'_{k_i}\cdot \text{backlog}_i)$:

$$\min_{\beta,\,\omega}\; \frac{1}{|S|} \sum_{i \in S} \Big|\, y_i - \mathrm{clock}_{b_i}\big(t_i,\; W_i^0 + \text{work}_i(\beta)\big) \Big|$$

**Stage 3, EM.** Let $u_i$ = open minutes the car actually used, $\mu_{it}$ = current formula durations, $\mathcal{C}_i$ =
tasks on the critical path, $\hat W_i$ = current waiting prediction. The time the model cannot explain is

$$g_i = u_i - \hat W_i - \text{work}_i$$

*E-step:* share $g_i$ among the critical-path tasks and waiting in proportion to their variances:

$$\hat d_{it} = \mu_{it} + \frac{\sigma_t^2}{\sum_{s \in \mathcal{C}_i} \sigma_s^2 + \sigma_W^2}\, g_i \quad (t \in \mathcal{C}_i), \qquad
\hat W_i \leftarrow \hat W_i + \frac{\sigma_W^2}{\sum_{s \in \mathcal{C}_i} \sigma_s^2 + \sigma_W^2}\, g_i$$

If that makes $\hat W_i < 0$, the wait is set to 0 and all of the unexplained time goes to the tasks. Tasks off the
critical path keep $\hat d_{it} = \mu_{it}$.

*M-step:* refit each task on its reconstructed durations and update its variance from the critical-path residuals:

$$\beta_t = \arg\min_\beta \sum_{i:\,f_{it}=1} \left(\hat d_{it} - x_{it}^\top \beta\right)^2, \qquad
\sigma_t^2 = \operatorname{mean}_{i:\, t \in \mathcal{C}_i}\left(\hat d_{it} - x_{it}^\top \beta_t\right)^2$$

**Stage 4, waiting model** ($\rho_{0.5}$ = absolute loss, i.e. median regression):

$$\gamma_k = \arg\min_\gamma \sum_{i:\,k_i = k} \rho_{0.5}\left(w_i - z_i^\top \gamma\right), \qquad w_i = \max\left(0,\; u_i - \text{work}_i\right)$$

At prediction time: formulas → work → waiting → clock. The E-step is only used in training.

## Covariate selection
A driver enters a task's formula if its correlation with the **true task duration on validation** (performed tasks only)
is |r| ≥ 0.1 **and** significant after **Bonferroni correction**: p < 0.05 / 199 = 2.5e-4, since we ran 199 task × variable tests.
Without the correction about 10 drivers would be expected to pass by chance. For task $t$ and candidate $j$:

$$j \in \text{drivers}(t) \iff |r_{tj}| \ge 0.1 \;\text{ and }\; p_{tj} < \frac{0.05}{199}$$
 Branch type enters as the staff dummies
(airport = 3 staff, station = 2, center = 1), since they are the same split.

Values are r (p).

| Task | Selected drivers | Dropped by Bonferroni (p < 0.05 but not < 2.5e-4) |
|---|---|---|
| Quick exterior | exterior dirt +0.37 (2e-53), airport +0.30 (3e-33), queue +0.21 (1e-16), van +0.15 (7e-9), station −0.11 (2e-5) | – |
| Normal exterior | exterior dirt +0.62 (3e-132), rain +0.22 (1e-14), queue +0.14 (1e-6), van +0.12 (4e-5) | – |
| Special cleaning | smoking +0.83 (6e-72), premium +0.29 (8e-7) | pet +0.17 (0.004), child seat +0.13 (0.028) |
| Interior | child seat +0.65 (<1e-300), interior dirt +0.58 (1e-254), rental days +0.19 (8e-25), km +0.16 (6e-18), airport +0.11 (3e-9) | – |
| Damage check | damage reported +0.36 (2e-44) | – |
| Refuel | airport +0.80 (1e-90), station −0.35 (3e-13), SUV +0.24 (1e-6), van +0.19 (2e-4) | rain +0.14 (0.005), weekend +0.13 (0.008) |
| Charging | battery −0.96 (1e-301) | weekend +0.10 (0.021) |
| Repair | damage reported +0.57 (2e-7) | – |
| Tyre check | airport −0.84 (1e-8) | return hour −0.42 (0.024) |

Fuel level is not a driver of refuel time (r = +0.08): refuel takes a fixed time per branch type.

**Caveat:** selection used validation labels, so validation scores are slightly optimistic. Coefficients are learned on
train only. `HP["covariates"] = "base"` gives the earlier, smaller set (strong drivers only, chosen informally).

## Models and why
All models run locally (numpy, pandas, scipy), open, no API cost. Scored on validation with `evaluate.py`.

| Model | Total MAE | Per-task WAPE | Avg task MAE |
| --- | --- | --- | --- |
| Structural fit + EM, Bonferroni-selected covariates | 13.34 | 0.140 | 3.86 |

## Results
Final model, validation week (2,791 cars), fitted on train only:

| Task | Cars | True mean (min) | Predicted mean | MAE |
|---|---|---|---|---|
| Quick exterior | 1,575 | 13.3 | 12.2 | 1.76 |
| Normal exterior | 1,216 | 19.2 | 15.5 | 4.00 |
| Special cleaning | 277 | 30.0 | 27.6 | 2.44 |
| Interior | 2,791 | 16.4 | 15.5 | 1.73 |
| Damage check | 1,394 | 2.4 | 2.9 | 0.57 |
| Refuel | 410 | 13.5 | 10.8 | 2.98 |
| Charging | 526 | 24.8 | 22.3 | 2.77 |
| Repair | 71 | 68.6 | 63.4 | 12.55 |
| Tyre check | 29 | 26.1 | 24.1 | 5.92 |

Total MAE 13.34 (baseline 98.6), per-task WAPE 0.140 (baseline 0.549, strong bar < 0.2), average task MAE 3.86, no negative
durations.

**Plausible effects** (learned coefficients, minutes): child seat +6.0 and +1.3 per dirt level (interior), smoking +3.6 and
premium +2.6 (special cleaning), damage reported +20.3 (repair; true difference +23), 0.34 per % missing battery (charging),
refuel +6.0 at airports and +2.2 at stations, tyre check −12.1 at airports.

**Not plausible:** van −8.8 (normal exterior) and −2.5 (quick exterior), although vans take longer in reality; damage
reported +3.5 for damage check (true +0.3). Durations are slightly underestimated on average (predicted total of task means
below the truth), the difference goes into waiting.

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
Deterministic: fixed seed (`HP["seed"] = 42`); `run.py --repeat` checks that two runs are identical.

## Demo
⏳ `submission/slides.pdf`

## Next steps
- Better waiting model: waits near closing and arrival bursts at city centers drive most of the remaining error.
- Multi-start for the structural fit; faster work computation.
- For SIXT: per-task times per branch type support staff planning and "car ready at" estimates for customers.
