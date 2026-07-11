# worldcup-2026-results-ml

An honestly-evaluated ML system that predicts (a) individual match outcomes and (b) tournament-level probabilities for the 2026 FIFA World Cup, built on **~49,500 international results (1872–present)** from the [martj42/international_results](https://github.com/martj42/international_results) dataset.

The emphasis is a systematized model measured against strong baselines with proper scoring rules — not a naive winner-picker. The standout deliverable is a **Monte Carlo tournament simulator** driven by a match model that beat every baseline on held-out data.

---

## TL;DR

Three match models, evaluated on the same held-out window (2024-01-01 through pre-WC-2026) with **Ranked Probability Score** (lower = better):

| Model | Notebook | Eval RPS |
|---|---|---|
| Naive base rates | — | 0.227 |
| Elo (K=30, D=150, home_bonus=50) | `04_elo` | 0.1695 |
| Feature ensemble (avg of 6 tuned learners) | `05_ensemble` | 0.1674 |
| Dixon-Coles (ξ=0.2, α=1e-05) | `03_dixon_coles` | 0.1646 |
| **Blend (50/50 ensemble avg + DC)** | `05_ensemble` | **0.1640** |

The blend is what drives the Monte Carlo simulator.

---

## Pre-tournament forecast (Path 1, 50k iterations)

Top 10 champion probabilities from the sim before the tournament started:

| Team | P(champion) |
|---|---|
| Argentina | 12.0% |
| Spain | 12.0% |
| Brazil | 9.2% |
| France | 7.7% |
| England | 7.2% |
| Portugal | 6.7% |
| Germany | 5.8% |
| Netherlands | 4.9% |
| Colombia | 4.6% |
| Belgium | 4.1% |

Top 6 concentrate ~55% of the equity — matches bookmaker consensus.

## Live forecast (Path 2, as of 2026-07-11)

Refreshed with real tournament results conditioned in: France in SF (beat Morocco), Spain in SF (beat Belgium). Only six possible champions remain.

| Team | P(champion) |
|---|---|
| Spain | 31.2% |
| France | 23.5% |
| Argentina | 22.5% |
| England | 15.3% |
| Norway | 4.0% |
| Switzerland | 3.6% |

Spain leads because they've already booked the SF, so they skip a QF coinflip that Argentina/England still face.

---

## Methodology

### Two-model spine on a scarce-data domain
Only ~2.5k held-out eval matches. The design compresses features hard, regularizes hard, and prefers flat stacking:

- **Dixon-Coles goal model** — `PoissonRegressor` with attack/defense team dummies + `is_home` (gated by `neutral`); time-decay weighting `exp(-ξ·age)`; L2 penalty. Tuned jointly on eval RPS.
- **6-learner Poisson-loss ensemble** — XGBoost, LightGBM, CatBoost, HistGradientBoosting, RandomForest, plus a `PoissonRegressor` GLM (kept specifically for pairwise decorrelation with the boosters — pairwise λ correlations across the 6 land at 0.87–0.99, GLM the biggest outlier). Features are derived, leakage-safe: as-of Elo, rolling form (`shift(1).rolling(5).mean()`), rest days, tournament tier, running H2H. Each learner tuned independently via Optuna TPE (20 trials, seed=42) with a joint recency-decay `ξ`. Booster n_estimators from early stopping on an inner-val slice.
- **Stacking meta was built and rejected.** A `PoissonRegressor` meta trained on OOF λ's from 3-fold expanding-window CV lost cleanly to the plain-average baseline (0.1761 vs 0.1674). Classic small-data outcome — the tuned base learners' eval predictions carry an optimistic bias that plain averaging preserves; the meta was trained on unbiased OOF λ's with a different noise structure than the eval-time λ's it's scored on. Same logic ruled out fitting a 2-feature ensemble+DC meta.
- **50/50 blend** with DC — the winning match model.

### Leakage-safe evaluation (walk-forward, three-tier split)
- **Train** — everything before 2024-01-01. Base learners fit here.
- **Eval** — 2024-01-01 → pre-WC-2026 (WC excluded). Tuning + selection bench.
- **wc2026** — the actual World Cup matches. **Sole test set**, held aside end-to-end.

Booster tuning additionally carves an **inner_val** (last 6 months of `train_df`, from `2023-07-01`) for early stopping — Optuna scores trials on `eval_df` while early stopping watches `inner_val`. `eval_df` is never mixed into training.

### Monte Carlo simulator
Feeds off the blend's λ's — the simulator adds no predictive skill of its own; it's only as good as the match model.

- **State snapshot** — walk all pre-WC matches chronologically, freeze end-of-history Elo + rolling form + last-match date per team, and running H2H counts + goal-diff sums per pair.
- **Fixture-level λ predictor** — given `(home, away, neutral, tier, date)`, build the same long-format 2-row feature frame the models were trained on, average the 6 tuned boosters, blend 50/50 with DC.
- **λ cache** — precompute λ for all `C(48,2) = 1128` possible team pairings once. Simulator inner loop is pure dict lookups — 50k iterations run in seconds.
- **One iteration**: 12 groups × 6 matches → standings by FIFA tiebreak (pts → GD → GF) → top 2 per group + 8 best 3rd-placed → 32 R32 qualifiers → 5-round bracket. Knockouts on draws resolved by 50/50 shootout.
- **Path 1** (pre-tournament) — simulate the whole thing from the draw. 50k iterations → champion probabilities.
- **Path 2** (live) — condition on played matches, only simulate what's left. Currently (2026-07-11) only 5 matches remain, so the distribution collapses onto the 6 alive teams.

### Design punts (v2)
- Host advantage (USA/CAN/MEX play group games at home) — currently modeled as neutral for simplicity; <5% of group fixtures affected.
- Real FIFA R32 bracket lookup — currently random-paired among the 32 qualifiers.
- Extra time before penalties — 90+ET collapsed to "one match; coinflip if drawn."
- Bayesian shrinkage layer for in-tournament λ updates (see `CLAUDE.md` §4.5) — deferred.

---

## Repo structure

```
worldcup-2026-results-ml/
├── CLAUDE.md              ← full operating manual & design notes
├── README.md              ← this file
├── environment.yml        ← conda env: worldcup-results (Python 3.11)
├── data/                  ← GITIGNORED (raw data licensing + bloat)
├── models/                ← GITIGNORED (large pickles)
├── src/
│   └── evaluation.py      ← ranked_probability_score, rps_from_lambdas
└── notebooks/
    ├── 01_data_cleaning.ipynb   ← load, clean, save parquet
    ├── 02_eda.ipynb             ← Poisson justified, decay & neutral decisions
    ├── 03_dixon_coles.ipynb     ← Model 1: RPS 0.1646, deploys dc_deployed.joblib
    ├── 04_elo.ipynb             ← Model 2: RPS 0.1695 baseline
    ├── 05_ensemble.ipynb        ← Model 3: 6-learner ensemble + DC blend, RPS 0.1640
    └── 06_monte_carlo.ipynb     ← Simulator: Path 1 pre-tournament + Path 2 live
```

## Running it

```bash
conda env create -f environment.yml
conda activate worldcup-results
# download martj42/international_results into data/raw/
# then walk the notebooks in order 01 → 06
```

Data isn't redistributed — pull it from [martj42/international_results](https://github.com/martj42/international_results) yourself.

---

## Constraints & honest caveats

- **Data ceiling is binding.** ~49k matches but only ~2.5k in eval. Regularization > flexibility. This is why simple averaging beat a stacking meta.
- **Calibration over raw accuracy.** The simulator compounds miscalibration across 7 rounds — RPS was tuned for calibrated probabilities, not top-1 correctness.
- **Provider-consistent features only.** No mixing xG definitions across sources — this project is pure results, no event data.
- **Licensing.** Raw data stays local; the repo publishes code + results only.
