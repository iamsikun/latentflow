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


def _check_rows_simplex(weights: np.ndarray, atol: float = 1e-8) -> None:
    """Validate that each row of a matrix lies on the probability simplex."""

    if weights.ndim != 2:
        raise ValueError("weights must be a 2D array of shape (n_states, n_mixtures)")
    for row in weights:
        _check_simplex(row, atol=atol)


@dataclass
class GaussianHMMParams:
    start_probs: np.ndarray  # (n_states,)
    trans_mat: np.ndarray  # (n_states, n_states)
    means: np.ndarray  # (n_states, n_features)
    covars: np.ndarray  # (n_states, n_features, n_features) or (n_states, n_features)

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
        if self.means.ndim != 2:
            raise ValueError("means must be (n_states, n_features).")

        if self.covars.ndim not in (2, 3):
            raise ValueError("covars must have shape (n_states, n_features) or (n_states, n_features, n_features).")

        n_features = self.means.shape[1]
        if self.covars.shape[1] != n_features:
            raise ValueError("means and covars have inconsistent n_features.")
        if self.covars.ndim == 3 and self.covars.shape[2] != n_features:
            raise ValueError("covars must have square (n_features x n_features) blocks when full covariance is used.")


@dataclass
class GaussianARHMMParams:
    start_probs: np.ndarray  # (n_states,)
    trans_mat: np.ndarray  # (n_states, n_states)
    coeffs: np.ndarray  # (n_states, n_features, order * n_features + 1)
    covars: np.ndarray  # (n_states, n_features, n_features) or (n_states, n_features)
    order: int

    @property
    def n_states(self) -> int:
        return self.start_probs.shape[0]

    @property
    def n_features(self) -> int:
        return self.coeffs.shape[1]

    def __post_init__(self) -> None:
        self.start_probs = np.asarray(self.start_probs, dtype=float)
        self.trans_mat = np.asarray(self.trans_mat, dtype=float)
        self.coeffs = np.asarray(self.coeffs, dtype=float)
        self.covars = np.asarray(self.covars, dtype=float)

        if not isinstance(self.order, int) or self.order < 0:
            raise ValueError("order must be a non-negative integer")

        _check_simplex(self.start_probs)
        _check_row_stochastic(self.trans_mat)

        n_states = self.trans_mat.shape[0]
        if self.start_probs.shape[0] != n_states:
            raise ValueError("start_probs and trans_mat have inconsistent n_states.")
        if self.coeffs.shape[0] != n_states or self.covars.shape[0] != n_states:
            raise ValueError("coeffs/covars and trans_mat have inconsistent n_states.")

        if self.coeffs.ndim != 3:
            raise ValueError("coeffs must have shape (n_states, n_features, order * n_features + 1).")
        if self.covars.ndim not in (2, 3):
            raise ValueError("covars must have shape (n_states, n_features) or (n_states, n_features, n_features).")

        n_features = self.coeffs.shape[1]
        if self.covars.shape[1] != n_features:
            raise ValueError("coeffs and covars have inconsistent n_features.")
        if self.covars.ndim == 3 and self.covars.shape[1] != self.covars.shape[2]:
            raise ValueError("covars must have square (n_features x n_features) blocks when full covariance is used.")

        expected_width = self.order * n_features + 1
        if self.coeffs.shape[2] != expected_width:
            raise ValueError(
                "The last dimension of coeffs must equal order * n_features + 1 (for intercept)."
            )


@dataclass
class GMMHMMParams:
    start_probs: np.ndarray  # (n_states,)
    trans_mat: np.ndarray  # (n_states, n_states)
    weights: np.ndarray  # (n_states, n_mixtures)
    means: np.ndarray  # (n_states, n_mixtures, n_features)
    covars: np.ndarray  # (n_states, n_mixtures, n_features[, n_features])

    @property
    def n_states(self) -> int:
        return self.start_probs.shape[0]

    @property
    def n_mixtures(self) -> int:
        return self.weights.shape[1]

    @property
    def n_features(self) -> int:
        return self.means.shape[2]

    def __post_init__(self) -> None:
        self.start_probs = np.asarray(self.start_probs, dtype=float)
        self.trans_mat = np.asarray(self.trans_mat, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.covars = np.asarray(self.covars, dtype=float)

        _check_simplex(self.start_probs)
        _check_row_stochastic(self.trans_mat)
        _check_rows_simplex(self.weights)

        n_states = self.trans_mat.shape[0]
        if self.start_probs.shape[0] != n_states:
            raise ValueError("start_probs and trans_mat have inconsistent n_states.")
        if self.weights.shape[0] != n_states or self.means.shape[0] != n_states:
            raise ValueError("weights/means and trans_mat have inconsistent n_states.")

        if self.covars.shape[0] != n_states:
            raise ValueError("covars and trans_mat have inconsistent n_states.")

        if self.weights.ndim != 2:
            raise ValueError("weights must have shape (n_states, n_mixtures).")
        if self.means.ndim != 3:
            raise ValueError("means must have shape (n_states, n_mixtures, n_features).")

        n_mixtures = self.weights.shape[1]
        if self.means.shape[1] != n_mixtures:
            raise ValueError("weights and means have inconsistent n_mixtures.")

        n_features = self.means.shape[2]
        if self.covars.ndim == 4:
            if self.covars.shape[1] != n_mixtures:
                raise ValueError("covars and weights have inconsistent n_mixtures.")
            if self.covars.shape[2] != n_features or self.covars.shape[3] != n_features:
                raise ValueError("covars must have shape (n_states, n_mixtures, n_features, n_features).")
        elif self.covars.ndim == 3:
            if self.covars.shape[1] != n_mixtures or self.covars.shape[2] != n_features:
                raise ValueError("covars must have shape (n_states, n_mixtures, n_features) when diagonal.")
        else:
            raise ValueError("covars must be either 3D (diag) or 4D (full).")


@dataclass
class GMMARHMMParams:
    start_probs: np.ndarray  # (n_states,)
    trans_mat: np.ndarray  # (n_states, n_states)
    weights: np.ndarray  # (n_states, n_mixtures)
    coeffs: np.ndarray  # (n_states, n_mixtures, n_features, order * n_features + 1)
    covars: np.ndarray  # (n_states, n_mixtures, n_features[, n_features])
    order: int

    @property
    def n_states(self) -> int:
        return self.start_probs.shape[0]

    @property
    def n_mixtures(self) -> int:
        return self.weights.shape[1]

    @property
    def n_features(self) -> int:
        return self.coeffs.shape[2]

    def __post_init__(self) -> None:
        self.start_probs = np.asarray(self.start_probs, dtype=float)
        self.trans_mat = np.asarray(self.trans_mat, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.coeffs = np.asarray(self.coeffs, dtype=float)
        self.covars = np.asarray(self.covars, dtype=float)

        if not isinstance(self.order, int) or self.order < 0:
            raise ValueError("order must be a non-negative integer")

        _check_simplex(self.start_probs)
        _check_row_stochastic(self.trans_mat)
        _check_rows_simplex(self.weights)

        n_states = self.trans_mat.shape[0]
        if self.start_probs.shape[0] != n_states:
            raise ValueError("start_probs and trans_mat have inconsistent n_states.")

        if self.weights.shape[0] != n_states or self.coeffs.shape[0] != n_states:
            raise ValueError("weights/coeffs and trans_mat have inconsistent n_states.")
        if self.covars.shape[0] != n_states:
            raise ValueError("covars and trans_mat have inconsistent n_states.")

        if self.weights.ndim != 2:
            raise ValueError("weights must have shape (n_states, n_mixtures).")
        if self.coeffs.ndim != 4:
            raise ValueError(
                "coeffs must have shape (n_states, n_mixtures, n_features, order * n_features + 1)."
            )

        n_mixtures = self.weights.shape[1]
        if self.coeffs.shape[1] != n_mixtures:
            raise ValueError("weights and coeffs have inconsistent n_mixtures.")

        if self.covars.ndim == 4:
            if self.covars.shape[1] != n_mixtures:
                raise ValueError("covars and weights have inconsistent n_mixtures.")
            if self.covars.shape[2] != self.covars.shape[3]:
                raise ValueError("full covariance matrices must be square.")
        elif self.covars.ndim == 3:
            if self.covars.shape[1] != n_mixtures:
                raise ValueError("covars and weights have inconsistent n_mixtures.")
        else:
            raise ValueError("covars must be either 3D (diag) or 4D (full).")

        n_features = self.coeffs.shape[2]
        if self.covars.shape[-1] != n_features:
            raise ValueError("coeffs and covars have inconsistent n_features.")

        expected_width = self.order * n_features + 1
        if self.coeffs.shape[3] != expected_width:
            raise ValueError(
                "The last dimension of coeffs must equal order * n_features + 1 (for intercept)."
            )
