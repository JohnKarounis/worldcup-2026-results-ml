"""Shared evaluation metrics — single source of truth across notebooks."""

import numpy as np


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
