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
| **Dixon-Coles** | `03_dixon_coles` | **0.1646** | Winner. Poisson attack/defense, ξ=0.2, α=1e-05, ρ=0 |
| Elo | `04_elo` | 0.1695 | Walk-forward, K=30, D=150, home_bonus=50 |
| Feature ensemble | `05_ensemble` | TBD | In progress |

**Decision:** Dixon-Coles is the match model. Elo is an honest baseline that lost on evidence.

---

## Architecture

### Poisson goal model (spine)
- Two λ's per match: `λ = exp(attack_team + defense_opp + venue)`.
- λ → Poisson PMF → scoreline matrix → W/D/L probabilities.
- Scorelines needed (not just W/D/L) because the 48-team format resolves on goal difference.

### Feature-based ensemble (05_ensemble, in progress)
- **As-of Elo** — pre-update rating logged per match, no leakage.
- **Rolling form** — `shift(1).rolling(5).mean()` of goals scored/conceded per team.
- Planned: rest days, H2H, tournament tier.
- Base learners: XGBoost, LightGBM, CatBoost, HistGradientBoosting (all `objective='poisson'`).
- Stack with penalized Poisson meta (same as main CLAUDE.md §4.3).

### Monte Carlo simulator (planned)
- Sample scorelines from λ's → group tables → advancement (top 2 + 8 best 3rd) → knockouts.
- ~50k iterations → tournament probabilities.

---

## Evaluation methodology

- **RPS** (ranked probability score) — ordinal-aware, the football standard. Also log loss, Brier.
- **Time-ordered / walk-forward only.** No random K-fold (leakage).
- Train: pre-2024. Eval: 2024→pre-WC-2026 (WC excluded). WC 2026: held aside.
- Always compare against baselines (naive, DC, Elo, bookmaker odds if available).

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
│   └── interim/           ← results_clean.parquet
└── notebooks/
    ├── 01_data_cleaning.ipynb   ← load, clean, save parquet
    ├── 02_eda.ipynb             ← Poisson justified, friendly vs competitive, neutral handling
    ├── 03_dixon_coles.ipynb     ← Model 1: Poisson attack/defense, RPS 0.1646
    ├── 04_elo.ipynb             ← Model 2: Elo → λ → scoreline, RPS 0.1695
    └── 05_ensemble.ipynb        ← Model 3: feature-based boosters (in progress)
```

## Conventions

- **Data is gitignored.** Publish code + results only. Anyone re-runs on their own copy.
- **No lookahead.** All features computed as-of match date (pre-update Elo, shifted rolling form).
- **Secrets never committed.** No API keys in this project (dataset is public).
- **Environment:** Windows + Anaconda + VS Code, conda env `worldcup-results` (Python 3.11).
- Explicit variable names, no cryptic abbreviations.
- Keep it simple — add machinery only with a clear reason.

---

## Key functions (cross-notebook)

- `reshape_to_long(df)` — match → 2 rows (attack/defend perspective). Used in DC model + eval.
- `fit_poisson_model(df, alpha, sample_weight)` — ColumnTransformer + PoissonRegressor pipeline.
- `predict_match(model, home, away, neutral, max_goals)` — two λ's → scoreline matrix → W/D/L.
- `ranked_probability_score(predicted_probs, actual_outcome)` — cumsum-based RPS.
- `evaluate_model(model, eval_df, rho, max_goals)` — batched predict, per-match RPS.
- `add_elo_features(matches, k, base_rating)` — as-of Elo (05_ensemble).
- `reshape_to_long_format(df)` — per-team table for rolling form (05_ensemble).

---

## Build order

1. ~~Data cleaning~~ ✓
2. ~~EDA~~ ✓
3. ~~Dixon-Coles model~~ ✓ (RPS 0.1646)
4. ~~Elo model~~ ✓ (RPS 0.1695)
5. **Feature ensemble** ← current (features in progress)
6. Monte Carlo simulator
7. 2026 live forecast

---

## Constraints

- **Data ceiling is binding** — ~49k matches but only ~2.5k in eval. Regularize hard, flat stack.
- **Calibration over raw accuracy** — simulator compounds miscalibration across 7 rounds.
- **Licensing** — raw data stays local; publish code + results only.
