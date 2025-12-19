from __future__ import annotations

from dataclasses import dataclass 
from typing import Optional, Tuple, Union

import numpy as np 

from latentflow.params import (
    GaussianARHMMParams,
    GaussianHMMParams,
    GMMARHMMParams,
    GMMHMMParams,
)


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
    first_cov = params.covars[states[0]]
    if first_cov.ndim == 1:
        y[0] = params.means[states[0]] + rng.normal(size=params.n_features) * np.sqrt(first_cov)
    else:
        y[0] = rng.multivariate_normal(params.means[states[0]], first_cov)

    # Sample subsequent states and observations
    for t in range(1, T):
        states[t] = _sample_categorical(params.trans_mat[states[t - 1]], rng)
        cov = params.covars[states[t]]
        if cov.ndim == 1:
            y[t] = params.means[states[t]] + rng.normal(size=params.n_features) * np.sqrt(cov)
        else:
            y[t] = rng.multivariate_normal(params.means[states[t]], cov)
    return states, y


def sample_gmm_hmm(
    params: GMMHMMParams,
    T: int,
    rng: Optional[np.random.Generator] = None,
    s0: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a trajectory from a Gaussian mixture HMM."""

    if rng is None:
        rng = np.random.default_rng()

    states = np.empty(T, dtype=int)
    y = np.empty((T, params.n_features), dtype=float)

    states[0] = s0 if (s0 is not None) else _sample_categorical(params.start_probs, rng)
    mix0 = _sample_categorical(params.weights[states[0]], rng)
    cov0 = params.covars[states[0], mix0]
    mean0 = params.means[states[0], mix0]
    if cov0.ndim == 1:
        y[0] = mean0 + rng.normal(size=params.n_features) * np.sqrt(cov0)
    else:
        y[0] = rng.multivariate_normal(mean0, cov0)

    for t in range(1, T):
        states[t] = _sample_categorical(params.trans_mat[states[t - 1]], rng)
        mix = _sample_categorical(params.weights[states[t]], rng)
        mean = params.means[states[t], mix]
        cov = params.covars[states[t], mix]
        if cov.ndim == 1:
            y[t] = mean + rng.normal(size=params.n_features) * np.sqrt(cov)
        else:
            y[t] = rng.multivariate_normal(mean, cov)

    return states, y


# ------------------------------
# Convenience factories: quick random-but-stable params
# ------------------------------

def make_random_gaussian_hmm(
    n_states: int,
    n_features: int,
    rng: Optional[np.random.Generator] = None,
) -> GaussianHMMParams:
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
    pi = rng.dirichlet(np.ones(n_states))  # initial state distribution
    trans_mat = rng.dirichlet(np.ones(n_states), size=n_states)
    means = rng.normal(scale=2.0, size=(n_states, n_features))
    covars = np.stack(
        [_random_psd(n_features, rng, diag_min=0.5) for _ in range(n_states)],
        axis=0,
    )
    return GaussianHMMParams(pi, trans_mat, means, covars)


def make_random_gaussian_arhmm(
    n_states: int,
    n_features: int,
    order: int,
    rng: Optional[np.random.Generator] = None,
) -> GaussianARHMMParams:
    """Create random-but-stable parameters for an autoregressive Gaussian HMM."""

    if rng is None:
        rng = np.random.default_rng()

    if order < 0:
        raise ValueError("order must be non-negative")

    width = order * n_features + 1

    start_probs = rng.dirichlet(np.ones(n_states))
    trans_mat = rng.dirichlet(np.ones(n_states), size=n_states)

    coeffs = rng.normal(scale=0.3, size=(n_states, n_features, width))
    # Intercepts should have a wider spread to avoid degenerate means.
    coeffs[:, :, -1] = rng.normal(scale=1.0, size=(n_states, n_features))

    covars = np.stack([_random_psd(n_features, rng, diag_min=0.5) for _ in range(n_states)], axis=0)

    return GaussianARHMMParams(
        start_probs=start_probs,
        trans_mat=trans_mat,
        coeffs=coeffs,
        covars=covars,
        order=order,
    )


def make_random_gaussian_mixture_hmm(
    n_states: int,
    n_features: int,
    n_mixtures: int,
    *,
    covariance_type: str = "full",
    rng: Optional[np.random.Generator] = None,
) -> GMMHMMParams:
    """Create random-but-stable parameters for a Gaussian mixture HMM."""

    if rng is None:
        rng = np.random.default_rng()

    if covariance_type not in {"full", "diag"}:
        raise ValueError("covariance_type must be either 'full' or 'diag'")

    start_probs = rng.dirichlet(np.ones(n_states))
    trans_mat = rng.dirichlet(np.ones(n_states), size=n_states)
    weights = rng.dirichlet(np.ones(n_mixtures), size=n_states)
    means = rng.normal(scale=2.0, size=(n_states, n_mixtures, n_features))

    if covariance_type == "full":
        covars = np.empty((n_states, n_mixtures, n_features, n_features), dtype=float)
        for i in range(n_states):
            for j in range(n_mixtures):
                covars[i, j] = _random_psd(n_features, rng, diag_min=0.5)
    else:
        covars = rng.uniform(0.5, 1.5, size=(n_states, n_mixtures, n_features))

    return GMMHMMParams(
        start_probs=start_probs,
        trans_mat=trans_mat,
        weights=weights,
        means=means,
        covars=covars,
    )


def make_random_gaussian_mixture_arhmm(
    n_states: int,
    n_features: int,
    order: int,
    n_mixtures: int,
    *,
    covariance_type: str = "full",
    rng: Optional[np.random.Generator] = None,
) -> GMMARHMMParams:
    """Create random-but-stable parameters for a Gaussian mixture ARHMM."""

    if rng is None:
        rng = np.random.default_rng()

    if order < 0:
        raise ValueError("order must be non-negative")
    if covariance_type not in {"full", "diag"}:
        raise ValueError("covariance_type must be either 'full' or 'diag'")

    width = order * n_features + 1

    start_probs = rng.dirichlet(np.ones(n_states))
    trans_mat = rng.dirichlet(np.ones(n_states), size=n_states)
    weights = rng.dirichlet(np.ones(n_mixtures), size=n_states)

    coeffs = rng.normal(scale=0.3, size=(n_states, n_mixtures, n_features, width))
    coeffs[..., -1] = rng.normal(scale=1.0, size=(n_states, n_mixtures, n_features))

    if covariance_type == "full":
        covars = np.empty((n_states, n_mixtures, n_features, n_features), dtype=float)
        for i in range(n_states):
            for j in range(n_mixtures):
                covars[i, j] = _random_psd(n_features, rng, diag_min=0.5)
    else:
        covars = rng.uniform(0.5, 1.5, size=(n_states, n_mixtures, n_features))

    return GMMARHMMParams(
        start_probs=start_probs,
        trans_mat=trans_mat,
        weights=weights,
        coeffs=coeffs,
        covars=covars,
        order=order,
    )


def sample_gaussian_arhmm(
    params: GaussianARHMMParams,
    T: int,
    rng: Optional[np.random.Generator] = None,
    s0: Optional[int] = None,
    history: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a trajectory from an autoregressive Gaussian HMM."""

    if rng is None:
        rng = np.random.default_rng()

    order = params.order
    n_features = params.n_features

    if history is None:
        history_arr = np.zeros((order, n_features), dtype=float)
    else:
        history_arr = np.asarray(history, dtype=float)
        if history_arr.shape != (order, n_features):
            raise ValueError(
                f"history must have shape ({order}, {n_features}) when provided; got {history_arr.shape}."
            )

    states = np.empty(T, dtype=int)
    y = np.empty((T, n_features), dtype=float)

    states[0] = s0 if (s0 is not None) else _sample_categorical(params.start_probs, rng)

    def _design_vector(t: int) -> np.ndarray:
        if order == 0:
            return np.array([1.0])
        z = np.empty(order * n_features + 1, dtype=float)
        for lag in range(1, order + 1):
            idx = t - lag
            if idx >= 0:
                prev = y[idx]
            else:
                prev = history_arr[order + idx]
            start = (lag - 1) * n_features
            z[start : start + n_features] = prev
        z[-1] = 1.0
        return z

    for t in range(T):
        if t > 0:
            states[t] = _sample_categorical(params.trans_mat[states[t - 1]], rng)

        z_t = _design_vector(t)
        coeffs = params.coeffs[states[t]]
        mean = coeffs @ z_t

        cov = params.covars[states[t]]
        if cov.ndim == 1:
            noise = rng.normal(size=n_features) * np.sqrt(cov)
            y[t] = mean + noise
        else:
            y[t] = rng.multivariate_normal(mean, cov)

    return states, y


def sample_gmm_arhmm(
    params: GMMARHMMParams,
    T: int,
    rng: Optional[np.random.Generator] = None,
    s0: Optional[int] = None,
    history: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a trajectory from a Gaussian mixture autoregressive HMM."""

    if rng is None:
        rng = np.random.default_rng()

    order = params.order
    n_features = params.n_features

    if history is None:
        history_arr = np.zeros((order, n_features), dtype=float)
    else:
        history_arr = np.asarray(history, dtype=float)
        if history_arr.shape != (order, n_features):
            raise ValueError(
                f"history must have shape ({order}, {n_features}) when provided; got {history_arr.shape}."
            )

    def _design_vector(t: int) -> np.ndarray:
        if order == 0:
            return np.array([1.0])
        z = np.empty(order * n_features + 1, dtype=float)
        for lag in range(1, order + 1):
            idx = t - lag
            if idx >= 0:
                prev = y[idx]
            else:
                prev = history_arr[order + idx]
            start = (lag - 1) * n_features
            z[start : start + n_features] = prev
        z[-1] = 1.0
        return z

    states = np.empty(T, dtype=int)
    y = np.empty((T, n_features), dtype=float)

    states[0] = s0 if (s0 is not None) else _sample_categorical(params.start_probs, rng)
    mix0 = _sample_categorical(params.weights[states[0]], rng)

    for t in range(T):
        if t > 0:
            states[t] = _sample_categorical(params.trans_mat[states[t - 1]], rng)
            mix0 = _sample_categorical(params.weights[states[t]], rng)

        z_t = _design_vector(t)
        coeffs = params.coeffs[states[t], mix0]
        mean = coeffs @ z_t
        cov = params.covars[states[t], mix0]
        if cov.ndim == 1:
            y[t] = mean + rng.normal(size=n_features) * np.sqrt(cov)
        else:
            y[t] = rng.multivariate_normal(mean, cov)

    return states, y


def sample_any_hmm(
    params: Union[GaussianHMMParams, GaussianARHMMParams, GMMHMMParams, GMMARHMMParams],
    T: int,
    rng: Optional[np.random.Generator] = None,
    s0: Optional[int] = None,
    history: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Polymorphic dispatcher for sampling from any HMM type.
    """
    if isinstance(params, GaussianHMMParams):
        return sample_gaussian_hmm(params, T, rng, s0)
    elif isinstance(params, GMMHMMParams):
        return sample_gmm_hmm(params, T, rng, s0)
    elif isinstance(params, GaussianARHMMParams):
        return sample_gaussian_arhmm(params, T, rng, s0, history)
    elif isinstance(params, GMMARHMMParams):
        return sample_gmm_arhmm(params, T, rng, s0, history)
    else:
        raise TypeError(f"Unsupported parameter type: {type(params)}")
