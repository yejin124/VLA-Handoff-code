"""Exact integer numerator for a binary gate and integer branch-outcome change."""
import numpy as np
from fractions import Fraction


def gate_covariance(gate, change):
    """Return mean(g*d)-mean(g)*mean(d), preserving exact zero without epsilon.

    A gate is binary; an outcome change is -1, 0, or 1. Compute the integer
    numerator before the final division, avoiding a rounding residual entering
    a sign test as an observation. Inputs must be nonempty aligned vectors.
    """
    return float(gate_covariance_fraction(gate, change))


def gate_covariance_fraction(gate, change):
    """Retain a rational result when several strata are averaged into a cluster."""
    g, d = np.asarray(gate), np.asarray(change)
    if g.ndim != 1 or d.shape != g.shape or not len(g):
        raise ValueError("Expected aligned nonempty one-dimensional vectors")
    if not np.isin(g, (0, 1)).all() or not np.isin(d, (-1, 0, 1)).all():
        raise ValueError("Gate must be binary and outcome change in {-1,0,1}")
    pairs = [(int(a), int(b)) for a, b in zip(g, d)]
    n = len(pairs)
    numerator = n * sum(a*b for a, b in pairs) - sum(a for a, _ in pairs) * sum(b for _, b in pairs)
    return Fraction(numerator, n*n)
