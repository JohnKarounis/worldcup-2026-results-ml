# I built an ML system that predicted every knockout matchup at the 2026 World Cup

Model called Spain over France in the semi-final. Argentina over England in the other semi. And Spain over Argentina in the Final — 50.5% to 49.5%, effectively a coinflip that the model got right.

Three knockout matchups, three correct predictions, headlined by Spain lifting the trophy.

Here's how it works — and, more interestingly, what didn't.

---

## The problem: prediction is easy, honest prediction is hard

You can throw XGBoost at football results, get 55% accuracy on W/D/L, and call it a day. That number is almost meaningless. The bookmaker's implied probabilities beat you on accuracy anyway, and — worse — a model that's confidently wrong is more dangerous than a model that's uncertain and right.

The football-standard metric is **Ranked Probability Score** (RPS). It's ordinal-aware — a "predicted draw, home actually won" prediction is penalised less than "predicted away win, home actually won", because outcomes are ordered by goal difference. Lower is better. Naive base rates (just predicting the historical W/D/L split every match) sit at 0.227. Bookmakers land in the 0.16 range.

The whole project is designed around one question: **can a model I build actually beat honest baselines on RPS, on truly held-out data?**

## The dataset

`martj42/international_results` on GitHub. **~49,500 international matches** back to 1872. Just fixtures + scores + tournament + venue — no xG, no possession, no lineups. Pure results.

I chose results-only deliberately. Mixing xG or event data across providers is a reproducibility nightmare (an Opta "interception" doesn't equal a StatsBomb one), and the goal was a clean, defensible pipeline — not to squeeze every last basis point out of the model.

Data split, walk-forward, three tiers:
- **Train**: everything before 2024-01-01
- **Eval**: 2024-01-01 → pre-WC-2026 (WC excluded) — this is the tuning + selection bench, ~2,500 matches
- **wc2026**: the actual World Cup matches, **held aside end-to-end** as the sole test

## Three models, one metric

I built three match models. Same target, same split, same RPS metric — head to head.

| Model | Eval RPS |
|---|---|
| Naive base rates | 0.227 |
| Elo (K=30, D=150, home_bonus=50) | 0.1695 |
| 6-learner Poisson ensemble (plain avg) | 0.1674 |
| **Dixon-Coles** (ξ=0.2, α=1e-05) | **0.1646** |
| **50/50 blend (DC + ensemble)** | **0.1640** ⭐ |

The interesting stories are in what these numbers mean.

### Dixon-Coles won as a single model

DC is a `PoissonRegressor` with attack/defense dummies for each team + a home indicator, exponentially time-decayed. It's ~30 lines of scikit-learn. And it beat a 6-learner boosting ensemble that took an hour of Optuna to tune.

Why? Because it's *structurally correct* for the problem. Football goals are Poisson-ish; attack strength and defense weakness are the right latent variables to learn. When you match the model to the data-generating process, you don't need flexibility — you need the right inductive bias.

### The ensemble was diverse on purpose

Six learners: XGBoost, LightGBM, CatBoost, HistGradientBoosting, RandomForest, plus a `PoissonRegressor` GLM. The GLM was kept **specifically for decorrelation** — the boosters were pairwise correlated at 0.87–0.99 on their λ predictions. The GLM was the biggest outlier, which is the whole point of a stack.

Features were all leakage-safe and derived from results: as-of Elo, rolling last-5 form (`shift(1).rolling(5).mean()`), rest days, tournament tier, running head-to-head. Tuned per-learner via Optuna TPE with a joint recency-decay ξ across trials.

Result: plain averaging landed at 0.1674. Not bad — but not DC.

### The stacking meta lost. And this is the interesting bit.

I built a `PoissonRegressor` meta on out-of-fold predictions from time-ordered 3-fold expanding-window CV. Tuned the L2 penalty via Optuna. It landed at 0.1761 — **worse than plain averaging (0.1674)**.

Textbook small-data outcome. The base learners' λ's on the eval set carry an optimistic bias because they were tuned to minimise eval RPS. Plain averaging preserves that. The meta was trained on OOF λ's with a different noise structure than the eval-time λ's it was later scored on — so its learned weights were miscalibrated.

Same logic ruled out fitting a 2-feature meta over `(DC λ, ensemble λ)`. On ~2,500 eval matches, the number of degrees of freedom you can safely fit is very small. A 50/50 blend adds one bit of information (yes, blend these two things) and doesn't overfit. That won: 0.1640 RPS.

**The rule I kept coming back to: at this data scale, the simplest thing that could possibly work is often what actually works.**

## The Monte Carlo simulator

A match model gives you λ per side per fixture. Two λ's give you a scoreline distribution (independent Poissons → outer product), from which W/D/L falls out.

To go from single-match predictions to tournament predictions:

1. **State snapshot** — walk pre-WC matches chronologically, freeze each team's end-of-history Elo, rolling form, last-match date, and running head-to-head records.
2. **λ cache** — precompute λ's for all `C(48,2) = 1128` possible team pairings once. Simulator inner loop is dict lookups, not model calls. This reduces 50k iterations from hours to seconds.
3. **One tournament**: 12 groups × 6 matches → standings sorted by pts / GD / GF → top-2 direct + 8 best 3rd-placed → 32 R32 qualifiers → 5-round bracket. Draws in knockouts resolved by 50/50 shootout.
4. **Loop 50k times** → concentration of "who wins" turns into P(champion).

## Pre-tournament forecast (Path 1)

50k iterations from the draw. Top 10:

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

Top 6 concentrate ~55% of the equity. Argentina/Spain tied because their Elos were within 4 points (2043 vs 2039). Nothing absurd — no minnow above 3.5%. Broadly matches bookmaker consensus.

## Live-conditioning forecast (Path 2)

As the tournament progressed, real results were conditioned in — only unplayed matches got simulated. By the semi-final stage, the field had narrowed to 4 teams:

| Team | P(champion) |
|---|---|
| Argentina | 29.1% |
| Spain | 28.9% |
| France | 21.3% |
| England | 20.7% |

## The result

Model was consulted on all three remaining knockout matchups:

| Match | Model pick | W% | Actual |
|---|---|---|---|
| SF: France vs Spain | Spain | 54.6% | Spain won ✓ |
| SF: Argentina vs England | Argentina | 55.3% | Argentina won ✓ |
| **Final: Spain vs Argentina** | **Spain** | **50.5%** | **Spain won ✓** |

Three for three. The Final was effectively a coinflip in the model's view — Spain edged Argentina by half a percentage point, driven by DC's form/attack components in the blend rather than raw Elo (where Argentina was fractionally ahead). Spain lifted the trophy.

## What I punted, and why

Honest section, because portfolio pieces without one aren't portfolios, they're marketing.

- **Host advantage in group games.** USA/Canada/Mexico play their group games at home. I modelled all group games as neutral. <5% of group fixtures affected.
- **Real FIFA R32 bracket.** The 2026 bracket has a specific pairing lookup based on which 3rd-placed teams qualify. I random-paired the 32 qualifiers. Simpler, and Monte Carlo washes out bracket luck across 50k iterations for the champion probabilities I cared about.
- **Extra time before penalties.** Collapsed "90 + ET → shootout" into "single match, coinflip if drawn." Marginal effect.
- **Bayesian shrinkage on in-tournament λ's.** Designed but not implemented. The idea: nudge team λ's based on residuals from played WC matches (opponent-adjusted because the expectation already accounts for opponent). By deep in the bracket, shrinkage weight rises naturally as `n / (n + τ)`.

None of these change the pitch. They're worth flagging because portfolio-quality means knowing where your model isn't quite honest yet.

## What I'd tell someone building this again

- **RPS, not accuracy.** Accuracy hides overconfidence. RPS punishes it.
- **Time-ordered evaluation, always.** Random K-fold on time series leaks.
- **Try the simple model first.** DC is 30 lines and it beat a tuned ensemble. Structure matters more than flexibility at small n.
- **Simple averaging is a hard baseline for stacking.** Especially on ~2,500 eval matches. Try it, and be willing to ship it if it wins.
- **Calibration compounds.** The simulator runs the match model through 7 rounds. A model that's miscalibrated by 2% per match is 14%+ off by the Final. Tune for calibration, not top-1 correctness.

## Links

- Repo: [github.com/JohnKarounis/worldcup-2026-results-ml](https://github.com/JohnKarounis/worldcup-2026-results-ml)
- Dataset: [martj42/international_results](https://github.com/martj42/international_results)

---

*Built as a portfolio project by John Karounis. Feedback welcome.*
