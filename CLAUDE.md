# CLAUDE.md — World Cup 2026 Results-Based Prediction System

## Project goal

Predict 2026 World Cup match outcomes and tournament probabilities using a **pure results
dataset** (~49k international matches, martj42/international_results). Portfolio project —
emphasis on honest evaluation against strong baselines, not a naive winner-picker.

The standout deliverable is a Monte Carlo tournament simulator that refreshes as real 2026
results come in.

Repo: `JohnKarounis/worldcup-2026-results-ml`. Owner is an ML practitioner with deep football
domain knowledge and a discretionary-trading background (treat it like a backtest: no
lookahead, time-ordered everything).

---

## Dataset

Source: `martj42/international_results` (GitHub / Kaggle). Three CSVs in `data/raw/` (gitignored):

- **results.csv** — ~49,482 clean matches (1872–present). Cols: `date, home_team, away_team,
  home_score, away_score, tournament, city, country, neutral`.
- **shootouts.csv** — ~679 rows. Knockout resolution.
- **goalscorers.csv** — ~47,837 goals. Minute-level, optional.

Join key: `(date, home_team, away_team)`.

---

## Model results (eval = 2024→pre-WC-2026, metric = RPS, lower = better)

| Model | Notebook | RPS | Notes |
|-------|----------|-----|-------|
| Naive base rates | — | 0.227 | Floor |
| Elo | `04_elo` | 0.1695 | Walk-forward, K=30, D=150, home_bonus=50 |
| Feature ensemble (plain avg of 6 tuned learners) | `05_ensemble` | 0.1674 | XGB/LGB/CB/HGB/RF/GLM, Optuna-tuned |
| Dixon-Coles | `03_dixon_coles` | 0.1646 | Poisson attack/defense, ξ=0.2, α=1e-05, ρ=0 |
| **Blend (50/50 ensemble avg + DC)** | `05_ensemble` | **0.1640** | Winner. Match model for the simulator. |

**Decision:** the 50/50 blend of the plain-averaged ensemble and Dixon-Coles is the match model
fed to the Monte Carlo simulator. DC alone lost by 0.0006 but is still the strongest single
model; the ensemble adds a small amount of complementary signal (form/rest/H2H-based patterns
vs pure attack/defense structure).

### Why plain average, not a fitted meta

A stacking meta (`PoissonRegressor` on OOF predictions from time-ordered 3-fold expanding-window
CV) was built and tuned via Optuna. **It lost cleanly to the plain average** (0.1761 vs 0.1674)
even with regularization pushed to zero. Root cause: the base learners were tuned to minimize
eval RPS, so their eval-time predictions carry an optimistic bias that plain averaging
preserves; the meta was trained on unbiased OOF λ's with a different noise structure than the
eval-time λ's it's scored on. Classic small-data stacking outcome. Same applies to the final
DC+ensemble blend — a 2-feature fitted meta wasn't attempted; 50/50 wins.

---

## Architecture

### Poisson goal model (spine)
- Two λ's per match: `λ = exp(attack_team + defense_opp + venue)`.
- λ → Poisson PMF → scoreline matrix → W/D/L probabilities.
- Scorelines needed (not just W/D/L) because the 48-team format resolves on goal difference.

### Feature-based ensemble (`05_ensemble`, complete)

**Features (all leakage-safe / as-of):**
- **As-of Elo** — K=30, base=1500. Recorded pre-update.
- **Rolling form** — `shift(1).rolling(5).mean()` of goals scored/conceded per team.
- **Rest days** — `groupby('team')['date'].diff().dt.days`.
- **Tournament tier** — ordinal (Friendly=0, Qualifier=1, Continental=2, WC=3).
- **Head-to-head** — matches played + goal difference from home team's perspective, running
  dicts keyed by `tuple(sorted([home, away]))` with sign-flip logic.

**Long-format reshape** — each match becomes 2 rows (home-attacker / away-attacker) so one
Poisson model learns attack and defense symmetrically. Doubles training rows (~38k → ~78k).

**Base learners (6, all Poisson objective):**
XGBoost, LightGBM, CatBoost, HistGradientBoosting, RandomForest, PoissonRegressor (GLM).
The 4 boosters + RF eat features raw; the GLM needs a `ColumnTransformer` pipeline
(median-impute, StandardScaler on numerics, OneHotEncoder on `tournament_tier`, passthrough
on `neutral`/`is_home`). The GLM is included specifically for **decorrelation** — pairwise λ
correlations across the 6 landed at 0.87–0.99 (GLM the biggest outlier), enough for stacking
to be well-posed.

**Tuning — Optuna TPE per learner** (20 trials each, seed=42, `direction='minimize'` on eval
RPS). Recency decay `ξ` (search range 0.01→0.1) tuned *jointly* with every learner's tree/reg
knobs, applied as `sample_weight = exp(-ξ * age_years)`. Booster n_estimators handled via
early stopping on an inner-val slice (last 6 months of `long_train`, from `2023-07-01`);
`best_iteration` saved as `trial.user_attrs['best_iteration']` and reused frozen at refit.

**Rule enforced:** `study.best_params.copy()` is the *only* source of truth for refit configs
— hand-rounding cost XGB 0.0002 RPS the first time (log-scaled `reg_alpha=0.00891` → 0.008 is
~10% off) and it's fully reproducible seed-scatter noise otherwise.

**Individual tuned RPS on eval:**
CB 0.1679, LGB 0.1679, XGB 0.1678, HGB 0.1681, RF 0.1680, GLM 0.1689 → plain average 0.1674.

**Stacking layer (built and rejected):** 3-fold expanding-window OOF (2008-2013, 2013-2018,
2018-2023) with pre-2008 kept as always-available training history. Meta = `PoissonRegressor`
with L2, trained on OOF λ's, scored on eval. Never beat the plain average.

### Blend layer (final match model)
Simple 50/50 blend of the ensemble's plain-average λ with Dixon-Coles' train-only λ on eval.
DC's eval predictions are cached at `data/interim/dc_eval_lambda.npy` from notebook 03. This
is the λ source for the Monte Carlo simulator.

### Deployed models (train + eval refit)
Once eval had done its job, each tuned learner was **refit on `pre_wc_df` (train + eval
combined)** to use every available match for actual 2026 predictions. Saved as
`models/{name}_deployed.joblib` (gitignored). The corresponding Optuna study objects are also
persisted (`models/study_{name}.joblib`) so a fresh session can skip the ~1.5h of retuning by
loading them and replaying `best_params` / `best_trial.user_attrs`.

Also saved: `models/{name}_tuned.joblib` (train-only fits) for reproducibility of the RPS
numbers reported above.

### Monte Carlo simulator (`06_monte_carlo`, complete)

**State snapshot (`build_team_state`, `build_h2h_state`)** — walk `pre_wc_df`
chronologically, freeze end-of-history Elo / rolling form / last-match date per team,
and running H2H counts + goal-diff-sums per pair. Feeds the fixture-level λ predictor.

**Fixture λ predictor (`predict_blend_lambdas`)** — given (home, away, neutral, tier,
date), builds the same long-format 2-row feature frame the models were trained on,
averages the 6 tuned boosters, blends 50/50 with DC → `(λ_home, λ_away)`. Stateless
— caller supplies the frozen state dicts. Score-time symmetry: same call gives same
λ regardless of team order (frozenset-keyed cache in the simulator).

**λ cache** — precompute λ for all `C(48,2) = 1128` possible pairings once
(neutral, tier=3, mid-July date). Simulator then does dict lookups only, not model
calls. Reduces 50k-iteration cost from hours to seconds.

**Group draw derivation** — reconstruct the 12 groups from `wc2026_df`'s first 72
matches (each team's group = itself + its 3 group-stage opponents). Anchor-based
relabel to official FIFA letters so knockout bracket wiring is deterministic across
kernel restarts (set-iteration order is hash-randomized).

**Tournament sim (`simulate_tournament`)**: 12 groups × 6 matches → standings sorted
by pts / GD / GF (FIFA tiebreak) → top 2 per group direct + 8 best 3rd-placed → 32 R32
qualifiers → 5-round bracket (R32/R16/QF/SF/F). Each knockout match: sample two Poisson
draws; draws resolved by a 50/50 shootout.

**Path 1 — pre-tournament sim (50k iterations, seed=42):**

| Team | P(champion) |
|------|-------------|
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

Top 6 sum to ~54.9%. Argentina/Spain tie matches their Elo tie (2043 vs 2039).
Nothing absurd — no minnow above 3.5%, no strong nation missing.

**Path 2 — live conditioning (as of 2026-07-11, 500k iterations):**

Tournament state: France beat Morocco, Spain beat Belgium (both in SF). Norway–England
and Argentina–Switzerland still to play; SF1 = France vs Spain; SF2 = QF3 winner vs
QF4 winner. Only 6 possible champions:

| Team | P(champion) |
|------|-------------|
| Spain | 31.2% |
| France | 23.5% |
| Argentina | 22.5% |
| England | 15.3% |
| Norway | 4.0% |
| Switzerland | 3.6% |

Spain leads because they've already booked the SF (skips a QF coinflip) and Elo-tie
Argentina still has to survive Switzerland.

**Punts noted for v2:**
- Host advantage in group games (USA/CAN/MEX at home, not neutral)
- Real FIFA R32 bracket lookup (currently random-paired among 32 qualifiers)
- Extra time before penalties (collapsed 90+ET → shootout coinflip)
- `team_state` frozen through the tournament (no rolling Elo/form updates as sim progresses)

### Final result — Spain won the 2026 World Cup

Model called **all three knockout matchups** it was consulted on (Path 2 + single-match reports):

| Match | Model pick | W% | Actual |
|---|---|---|---|
| SF: France vs Spain | Spain | 54.6% | Spain won ✓ |
| SF: Argentina vs England | Argentina | 55.3% | Argentina won ✓ |
| Final: Spain vs Argentina | Spain | 50.5% | Spain won ✓ |

Final was effectively a coinflip in the model (Spain 50.5% / Argentina 49.5%) — very close reflects the Elo tie (2039 vs 2043) with Spain slightly favored by DC/form components in the blend. All three predictions correct, headlined by Spain lifting the trophy.

---

## Evaluation methodology

- **RPS** (ranked probability score) — ordinal-aware, the football standard. Also log loss, Brier.
- **Time-ordered / walk-forward only.** No random K-fold (leakage).
- **Three-tier split**, boundaries mutually exclusive:
  - `train_df`: matches before 2024-01-01.
  - `eval_df`: 2024-01-01 → pre-WC-2026 (WC excluded). Tuning + selection bench.
  - `wc2026_df`: the actual World Cup matches. **Sole test set**, held aside.
- **Booster tuning** additionally carves an **inner_val** (last 6 months of `train_df`, from
  `2023-07-01`) for early stopping — Optuna scores trials on `eval_df` while early stopping
  watches `inner_val`. `eval_df` is never mixed into training.
- **RandomForest** has no early stopping — `n_estimators` tuned directly by Optuna.
- **HistGradientBoosting** can't accept an external `eval_set` — early stopping disabled,
  `max_iter` tuned directly.
- **PoissonRegressor (GLM)** is fit inside a `Pipeline` with a `ColumnTransformer` — sample
  weights forwarded via `glm__sample_weight` kwarg.
- Always compare against baselines (naive, Elo, DC, plain average).

---

## Repo structure

```
worldcup-2026-results-ml/
├── CLAUDE.md              ← this file
├── README.md
├── .gitignore
├── environment.yml        ← conda env: worldcup-results, python 3.11
├── data/                  ← GITIGNORED (licensing + bloat)
│   ├── raw/               ← results.csv, shootouts.csv, goalscorers.csv
│   └── interim/           ← results_clean.parquet, dc_eval_lambda.npy,
│                            model_df_featured.parquet
├── models/                ← GITIGNORED (large)
│   ├── {name}_tuned.joblib     ← 6 tuned learners fit on train only
│   ├── {name}_deployed.joblib  ← 6 tuned learners refit on train + eval
│   └── study_{name}.joblib     ← Optuna study objects for reload-skip-retune
├── src/
│   └── evaluation.py      ← ranked_probability_score, rps_from_lambdas
└── notebooks/
    ├── 01_data_cleaning.ipynb   ← load, clean, save parquet
    ├── 02_eda.ipynb             ← Poisson justified, friendly vs competitive, neutral handling
    ├── 03_dixon_coles.ipynb     ← Model 1: Poisson attack/defense, RPS 0.1646, saves dc_eval_lambda.npy
    ├── 04_elo.ipynb             ← Model 2: Elo → λ → scoreline, RPS 0.1695
    ├── 05_ensemble.ipynb        ← Model 3: 6-learner ensemble + blend with DC, RPS 0.1640
    └── 06_monte_carlo.ipynb     ← Simulator: state snapshot + λ predictor + Path 1 + Path 2
```

## Conventions

- **Data is gitignored.** Publish code + results only. Anyone re-runs on their own copy.
- **Models are gitignored** (`models/`) — pickles can be tens of MB (RF the worst).
- **No lookahead.** All features computed as-of match date (pre-update Elo, shifted rolling form).
- **Secrets never committed.** No API keys in this project (dataset is public).
- **Environment:** Windows + Anaconda + VS Code, conda env `worldcup-results` (Python 3.11).
- Explicit variable names, no cryptic abbreviations.
- Keep it simple — add machinery only with a clear reason.

---

## Key functions (cross-notebook)

- `reshape_to_long(df)` — match → 2 rows (attack/defend perspective). DC model + its eval loop.
- `reshape_to_long_training(df)` — same idea but carrying engineered strength/form features
  (used by the ensemble). Sign-flips `h2h_goal_diff` on the away-attacker row.
- `fit_poisson_model(df, alpha, sample_weight)` — ColumnTransformer + PoissonRegressor pipeline (DC).
- `predict_match(model, home, away, neutral, max_goals)` — two λ's → scoreline matrix → W/D/L.
- `add_elo_features(matches, k, base_rating)` — as-of Elo (`05_ensemble`).
- `add_h2h_features(matches)` — running H2H count + goal diff (`05_ensemble`).
- `build_team_state / build_h2h_state(matches)` — end-of-history snapshots feeding the
  simulator's fixture-level λ predictor (`06_monte_carlo`).
- `predict_blend_lambdas(...)` — stateless fixture λ producer: builds the 2-row long
  format, averages 6 boosters + blends 50/50 with DC (`06_monte_carlo`).
- `simulate_group / simulate_group_phase / simulate_knockout_match / simulate_bracket /
  simulate_tournament(...)` — the Monte Carlo stack (`06_monte_carlo`).
- `fit_cb / fit_lgb / fit_xgb / fit_hgb / fit_rf / fit_glm(X, y, w)` — one per base learner,
  each reads the tuned config from its `study_*` object (no rounding).
- `ranked_probability_score(predicted_probs, actual_outcome)` — cumsum-based RPS (`src/evaluation.py`).
- `rps_from_lambdas(lambda_vec, df, max_goals=10)` — full λ → scoreline → W/D/L → mean RPS pipeline;
  expects `lambda_vec` shape `(2n,)` (home half then away half in df order). Single source of
  truth for the RPS-from-λ boilerplate that was previously duplicated across cells (`src/evaluation.py`).

---

## Build order

1. ~~Data cleaning~~ ✓
2. ~~EDA~~ ✓
3. ~~Dixon-Coles model~~ ✓ (RPS 0.1646)
4. ~~Elo model~~ ✓ (RPS 0.1695)
5. ~~Feature ensemble + DC blend~~ ✓ (RPS 0.1640 — winner)
6. ~~Monte Carlo simulator~~ ✓ (Path 1 pre-tournament + Path 2 live-conditioning)
7. **2026 live forecast** ← current (Path 2 done; Bayesian shrinkage update layer still pending)

---

## Constraints

- **Data ceiling is binding** — ~49k matches but only ~2.5k in eval. Regularize hard, flat stack.
- **Calibration over raw accuracy** — simulator compounds miscalibration across 7 rounds.
- **Licensing** — raw data stays local; publish code + results only.
