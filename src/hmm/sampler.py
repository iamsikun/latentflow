from __future__ import annotations

from dataclasses import dataclass 
from typing import Optional, Tuple

import numpy as np 

from hmm.params import GaussianHMMParams


# ------------------------------
# Utility helpers
# ------------------------------

def _sample_categorical(p: np.ndarray, rng: np.random.Generator) -> int:
    """Draw a single categorical sample from probabilities p (1D)."""
    return rng.choice(len(p), p=p)


def _random_psd(d: int, rng: np.random.Generator, diag_min: float = 0.3) -> np.ndarray:
    """Quick helper: generate a random positive semi-definite covariance (d x d)."""
    M = rng.normal(size=(d, d))
    S = M @ M.T
    S += diag_min * np.eye(d)  # jitter for PD
    return S

# ------------------------------
# Sampling functions
# ------------------------------

def sample_gaussian_hmm(
    params: GaussianHMMParams,
    T: int,
    rng: Optional[np.random.Generator] = None,
    s0: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a trajectory from a Gaussian HMM.

    Params
    ------
    params: GaussianHMMParams
        Parameters of the Gaussian HMM.
    T: int
        Length of the trajectory.
    rng: Optional[np.random.Generator]
        Random number generator.
    s0: Optional[int]
        Initial state. If None, a random state is sampled from the initial state distribution.

    Returns
    -------
        states: np.ndarray, shape = (T,)
            States of the trajectory.
        y: np.ndarray, shape = (T, n_features)
            Observations of the trajectory.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize state and observation arrays
    states = np.empty(T, dtype=int)
    y = np.empty((T, params.n_features), dtype=float)
    
    # Sample initial state and observation
    states[0] = s0 if (s0 is not None) else _sample_categorical(params.start_probs, rng)
    y[0] = rng.multivariate_normal(params.means[states[0]], params.covars[states[0]])

    # Sample subsequent states and observations
    for t in range(1, T):
        states[t] = _sample_categorical(params.trans_mat[states[t - 1]], rng)
        y[t] = rng.multivariate_normal(params.means[states[t]], params.covars[states[t]])
    return states, y


# ------------------------------
# Convenience factories: quick random-but-stable params
# ------------------------------

def make_random_gaussian_hmm(n_states: int, n_features: int, rng: Optional[np.random.Generator] = None) -> GaussianHMMParams:
    """
    Create a random Gaussian HMM with stable parameters.

    Params
    ------
        n_states: int
            Number of states.
        n_features: int
            Number of features.
        rng: np.random.Generator
            Random number generator.

    Returns
    -------
        hmm: GaussianHMMParams
            Random Gaussian HMM.
    """
    if rng is None:
        rng = np.random.default_rng()
    pi = rng.dirichlet(np.ones(n_states))
    trans_mat = rng.dirichlet(np.ones(n_states), size=n_states)
    means = rng.normal(scale=2.0, size=(n_states, n_features))
    covars = np.stack([_random_psd(n_features, rng, diag_min=0.5) for _ in range(n_states)], axis=0)
    return GaussianHMMParams(pi, trans_mat, means, covars)