"""Shared Hidden Markov Model utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from latentflow.utils import _check_random_state

try:  # pragma: no cover - optional Cython acceleration
    from . import _hmm_cy  # type: ignore
except Exception:  # pragma: no cover - fallback when Cython extension missing
    _hmm_cy = None

# Suppress divide by zero warnings which happen when taking log of 0 probability
np.seterr(divide='ignore')


def logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Stable log-sum-exp implementation."""

    m = np.max(a, axis=axis, keepdims=True)
    res = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if axis is not None:
        res = np.squeeze(res, axis=axis)
    return res


def split_sequences(X: np.ndarray, lengths: Optional[Sequence[int]]) -> List[np.ndarray]:
    """Split a concatenated observation matrix into sequences of given lengths."""

    if lengths is None:
        return [X]
    out: List[np.ndarray] = []
    start = 0
    for L in lengths:
        if L < 0:
            raise ValueError("lengths must be non-negative")
        out.append(X[start : start + L])
        start += L
    if start != len(X):
        raise ValueError("sum(lengths) != len(X)")
    return out


def _forward_log_py(
    start_probs: np.ndarray, trans_mat: np.ndarray, log_b: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Pure Python forward algorithm implementation."""

    T, n_states = log_b.shape
    log_alpha = np.empty((T, n_states), dtype=float)
    log_start = np.log(start_probs)
    log_trans = np.log(trans_mat)

    log_alpha[0] = log_start + log_b[0]
    for t in range(1, T):
        tmp = log_alpha[t - 1][:, None] + log_trans
        log_alpha[t] = logsumexp(tmp, axis=0) + log_b[t]
    loglik = float(logsumexp(log_alpha[-1], axis=0))
    return log_alpha, loglik


def forward_log(
    start_probs: np.ndarray, trans_mat: np.ndarray, log_b: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Run the forward algorithm in log-space."""

    if _hmm_cy is not None:
        return _hmm_cy.forward_log(start_probs, trans_mat, log_b)
    return _forward_log_py(start_probs, trans_mat, log_b)


def _backward_log_py(trans_mat: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Pure Python backward algorithm implementation."""

    T, n_states = log_b.shape
    log_beta = np.empty((T, n_states), dtype=float)
    log_trans = np.log(trans_mat)
    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        tmp = log_trans + log_b[t + 1] + log_beta[t + 1]
        log_beta[t] = logsumexp(tmp, axis=1)
    return log_beta


def backward_log(trans_mat: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Run the backward algorithm in log-space."""

    if _hmm_cy is not None:
        return _hmm_cy.backward_log(trans_mat, log_b)
    return _backward_log_py(trans_mat, log_b)


def _viterbi_py(start_probs: np.ndarray, trans_mat: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Pure Python Viterbi algorithm implementation."""

    T, n_states = log_b.shape
    log_start = np.log(start_probs)
    log_trans = np.log(trans_mat)
    delta = np.empty((T, n_states), dtype=float)
    psi = np.empty((T, n_states), dtype=int)

    delta[0] = log_start + log_b[0]
    psi[0] = 0
    for t in range(1, T):
        tmp = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(tmp, axis=0)
        delta[t] = tmp[psi[t], np.arange(n_states)] + log_b[t]
    states = np.empty(T, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = int(psi[t + 1, states[t + 1]])
    return states


def viterbi(start_probs: np.ndarray, trans_mat: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Compute the most-likely state path via the Viterbi algorithm."""

    if _hmm_cy is not None:
        return _hmm_cy.viterbi(start_probs, trans_mat, log_b)
    return _viterbi_py(start_probs, trans_mat, log_b)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a 1-D array onto the probability simplex."""

    s = v.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(v) / len(v)
    return v / s


def default_clone_params(params: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Create a defensive copy of EM parameter dictionaries."""

    return {key: np.array(value, copy=True) for key, value in params.items()}


def run_em(
    sequences: Sequence[np.ndarray],
    *,
    init_params: Callable[[np.random.Generator], Mapping[str, np.ndarray]],
    e_step: Callable[[Mapping[str, np.ndarray]], Tuple[float, Mapping[str, np.ndarray]]],
    m_step: Callable[[Mapping[str, np.ndarray], Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
    clone_params: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]] = default_clone_params,
    n_init: int,
    n_iter: int,
    tol: float,
    random_state: Optional[int | np.random.Generator],
    verbose: bool = False,
) -> Tuple[Mapping[str, np.ndarray], bool, float, List[float]]:
    """
    Generic EM driver shared by different HMM variants.

    Returns the best parameter set (by log-likelihood), a convergence flag,
    the final log-likelihood, and the history of log-likelihoods for the best run.
    """

    if n_init <= 0:
        raise ValueError("n_init must be positive")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive")

    rng_master = _check_random_state(random_state)

    best_params: Optional[Mapping[str, np.ndarray]] = None
    best_loglik = -np.inf
    best_history: List[float] = []
    best_converged = False

    for trial in range(n_init):
        rng = _check_random_state(rng_master.integers(0, 2**32 - 1))
        params = dict(init_params(rng))
        prev_loglik = -np.inf
        history: List[float] = []
        converged = False
        loglik = -np.inf

        for it in range(n_iter):
            loglik, stats = e_step(params)
            params = dict(m_step(stats, params))
            history.append(loglik)

            if verbose:
                print(f"EM iter {it + 1:03d}: loglik={loglik:.6f}")

            if it > 0 and loglik - prev_loglik < tol:
                converged = True
                break
            prev_loglik = loglik

        if loglik > best_loglik:
            best_loglik = loglik
            best_params = clone_params(params)
            best_history = history.copy()
            best_converged = converged

    if best_params is None:
        raise RuntimeError("EM failed to produce parameter estimates")

    return best_params, best_converged, float(best_loglik), best_history
