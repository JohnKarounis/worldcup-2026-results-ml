"""Shared evaluation metrics — single source of truth across notebooks."""

import numpy as np
from scipy.stats import poisson


def ranked_probability_score(predicted_probs, actual_outcome):
    """Ordinal-aware score for W/D/L probabilities (lower = better).

    Compares cumulative predicted vs actual distributions, so a near-miss
    (predicted draw, home won) is penalised less than a far-miss
    (predicted away-win, home won). Normalised to [0, 1] by (n - 1).
    """
    cum_pred = np.cumsum(predicted_probs)
    cum_actual = np.cumsum(actual_outcome)
    n = len(predicted_probs)
    return np.sum((cum_pred - cum_actual) ** 2) / (n - 1)


def rps_from_lambdas(lambda_vec, df, max_goals=10):
    """Mean RPS across matches, given per-side Poisson rates.

    lambda_vec: shape (2n,) — home λ's stacked over away λ's, in df order.
                This is the standard output of predict() on the long-format
                reshape (attacker/defender rows concatenated).
    df:         DataFrame with 'home_score' and 'away_score' columns.

    Full pipeline per match: λ_home, λ_away → Poisson PMFs → outer product
    scoreline matrix → W/D/L (tril/diag/triu sums) → RPS vs one-hot actual.
    """
    n = len(df)
    lambda_home = lambda_vec[:n]
    lambda_away = lambda_vec[n:]

    home_score = df['home_score'].to_numpy()
    away_score = df['away_score'].to_numpy()

    rps_values = []
    for i in range(n):
        home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home[i])
        away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away[i])
        matrix = np.outer(home_probs, away_probs)
        probs = [
            np.tril(matrix, -1).sum(),
            np.trace(matrix),
            np.triu(matrix, 1).sum(),
        ]

        if home_score[i] > away_score[i]:
            actual = [1, 0, 0]
        elif home_score[i] == away_score[i]:
            actual = [0, 1, 0]
        else:
            actual = [0, 0, 1]

        rps_values.append(ranked_probability_score(probs, actual))

    return np.mean(rps_values)
