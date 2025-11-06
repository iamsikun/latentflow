from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ------------------------------
# Utility helpers
# ------------------------------

def _check_row_stochastic(trans_mat: np.ndarray, atol: float = 1e-8) -> None:
    """Raise ValueError if rows of the transition matrix are not (approximately) stochastic."""
    if trans_mat.ndim != 2 or trans_mat.shape[0] != trans_mat.shape[1]:
        raise ValueError("The transition matrix must be square (n_states x n_states).")
    row_sums = trans_mat.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"Each row of the transition matrix must sum to 1. Row sums: {row_sums}.")
    if (trans_mat < -atol).any():
        raise ValueError("The transition matrix has negative entries.")
    if (trans_mat > 1 + atol).any():
        raise ValueError("The transition matrix has entries > 1.")

def _check_simplex(p: np.ndarray, atol: float = 1e-8) -> None:
    """Raise ValueError if vector p is not on the probability simplex."""
    if p.ndim != 1:
        raise ValueError("Probability vector must be 1D.")
    if not np.allclose(p.sum(), 1.0, atol=atol):
        raise ValueError(f"Probability vector must sum to 1. Got {p.sum():.6f}")
    if (p < -atol).any() or (p > 1 + atol).any():
        raise ValueError("Probability vector has invalid entries outside [0,1].")


@dataclass
class GaussianHMMParams:
    start_probs: np.ndarray           # (n_states,)
    trans_mat: np.ndarray    # (n_states, n_states)
    means: np.ndarray        # (n_states, n_features)
    covars: np.ndarray       # (n_states, n_features, n_features)

    @property
    def n_states(self) -> int:
        return self.start_probs.shape[0]

    @property
    def n_features(self) -> int:
        return self.means.shape[1]

    def __post_init__(self):
        self.start_probs = np.asarray(self.start_probs, dtype=float)
        self.trans_mat = np.asarray(self.trans_mat, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.covars = np.asarray(self.covars, dtype=float)
        
        _check_simplex(self.start_probs)
        _check_row_stochastic(self.trans_mat)

        n_states = self.trans_mat.shape[0]
        if self.start_probs.shape[0] != n_states:
            raise ValueError("start_probs and trans_mat have inconsistent n_states.")
        if self.means.shape[0] != n_states or self.covars.shape[0] != n_states:
            raise ValueError("means/covars and trans_mat have inconsistent n_states.")
        if self.means.ndim != 2 or self.covars.ndim != 3:
            raise ValueError("means must be (n_states, n_features) and covars must be (n_states, n_features, n_features).")
        if self.covars.shape[1] != self.covars.shape[2]:
            raise ValueError("covars must have square (n_features x n_features) blocks.")
        if self.covars.shape[1] != self.means.shape[1]:
            raise ValueError("means and covars have inconsistent n_features.")
