from __future__ import annotations

import numpy as np 
from numpy.typing import ArrayLike
from dataclasses import field
from sklearn.base import BaseEstimator
from typing import List, Sequence, Tuple, Optional, Any

from sklearn.cluster import KMeans

from hmm.utils import _check_random_state
from hmm.params import GaussianHMMParams
from hmm.sampler import sample_gaussian_hmm

def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    res = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if axis is not None:
        res = np.squeeze(res, axis=axis)
    return res


def _split_sequences(X: np.ndarray, lengths: Optional[Sequence[int]]) -> List[np.ndarray]:
    """
    Split a sequence into segments of given lengths.
    """
    if lengths is None:
        return [X]
    out = []
    start = 0
    for L in lengths:
        if L < 0:
            raise ValueError("lengths must be non-negative")
        out.append(X[start : start + L])
        start += L
    if start != len(X):
        raise ValueError("sum(lengths) != len(X)")
    return out


# ---------------------------------------------------------------------------
# Main estimators
# ---------------------------------------------------------------------------
class GaussianHMM:
    def __init__(
        self,
        n_components: int,
        *,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int | np.random.Generator] = None,
        init: str = "kmeans",
        n_init: int = 1,
    ) -> None:
        """Gaussian Hidden Markov Model (HMM).

        Parameters
        ----------
        n_components : int
            Number of hidden states.
        covariance_type : {"full", "diag"}, default="full"
            Form of the covariance matrices for each state.
        n_iter : int, default=100
            Maximum number of EM iterations.
        tol : float, default=1e-4
            Convergence threshold on the improvement of total log-likelihood.
        reg_covar : float, default=1e-6
            Non-negative regularization added to the diagonal of covariance
            matrices to ensure they stay positive definite / non-singular.
        random_state : int or Generator, optional
            Controls randomness for initialization and sampling.
        init : {"kmeans", "random"}, default="kmeans"
            Initialization strategy for means. If scikit-learn is not available,
            falls back to "random" automatically.
        n_init : int, default=1
            Number of random initializations to try; the best (highest total
            log-likelihood after training) is retained.

        Attributes (learned after `fit`)
        -------------------------------
        start_probs : (K,) ndarray
            Initial state distribution.
        trans_mat : (K, K) ndarray
            State transition matrix. Rows sum to 1.
        means : (K, D) ndarray
            Means of each state's Gaussian.
        covars : (K, D, D) or (K, D) ndarray
            Covariance matrices. Shape depends on `covariance_type`.
        converged : bool
            Whether EM converged.
        n_iter : int
            Number of iterations run for the best initialization.
        loglik : float
            Final total log-likelihood of the training data under the model.
        """
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if covariance_type not in {"full", "diag"}:
            raise ValueError("covariance_type must be 'full' or 'diag'")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        if init not in {"kmeans", "random"}:
            raise ValueError("init must be 'kmeans' or 'random'")
        if n_init <= 0:
            raise ValueError("n_init must be positive")

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init = init
        self.n_init = n_init

        # learned params (post-fit)
        self.converged: bool | None = None
        self.loglik: float | None = None
        self.history: list[float] = field(default_factory=list)
        self.params: GaussianHMMParams | None = None

    # ------------------------------------------------------------------
    # scikit-learn compatible API
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
            "init": self.init,
            "n_init": self.n_init,
        }

    def set_params(self, **params) -> GaussianHMM:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, verbose: bool = False) -> GaussianHMM:
        """Fit the model by EM / Baum–Welch.

        Parameters
        ----------
        X: np.ndarray, shape = (T, n_features)
            Input data.
        lengths : optional list of ints
            Segment lengths for each sequence in the concatenated `X`.
        verbose : bool, default=False
            If True, prints iteration progress and log-likelihood.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (T, n_features)")
        T, n_features = X.shape

        # Split sequences into segments
        sequences = _split_sequences(X, lengths)

        # Initialize best parameters
        best_loglik = -np.inf
        best_params = None
        rng_master = _check_random_state(self.random_state)

        # Try multiple initializations
        for trial in range(self.n_init):
            rng = _check_random_state(rng_master.integers(0, 2**32 - 1))
            # initialize
            start_probs, trans_mat, means, covars = self._init_params(
                sequences, n_features, rng
            )

            # Initialize log-likelihood and convergence flags for this initialization
            prev_loglik = -np.inf
            converged = False
            history = []

            # Run EM iterations
            for it in range(self.n_iter):
                loglik, stats = self._e_step(sequences, start_probs, trans_mat, means, covars)
                start_probs, trans_mat, means, covars = self._m_step(stats, n_features)

                history.append(loglik)

                if verbose:
                    print(f"EM iter {it+1:03d}: loglik={loglik:.6f}")

                if it > 0 and loglik - prev_loglik < self.tol:
                    converged = True
                    break
                prev_loglik = loglik

            # Update best parameters if this initialization is better
            if loglik > best_loglik:
                best_loglik = loglik
                best_params = (start_probs, trans_mat, means, covars, converged, best_loglik, history)

        # store best
        (
            start_probs,
            trans_mat,
            means,
            covars,
            self.converged,
            self.loglik,
            self.history,
        ) = best_params
        self.params = GaussianHMMParams(
            start_probs=start_probs,
            trans_mat=trans_mat,
            means=means,
            covars=covars,
        )
        return self

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> float:
        """
        Compute the average log-likelihood per sample.

        This returns the total log-likelihood divided by the number of
        observations, so it is comparable across datasets.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = _split_sequences(X, lengths)
        loglik = 0.0
        total_T = 0
        for seq in sequences:
            log_b = self._log_emission(seq, self.params.means, self.params.covars)
            _, loglik_seq = self._forward_log(self.params.start_probs, self.params.trans_mat, log_b)
            loglik += loglik_seq
            total_T += len(seq)
        return float(loglik / max(total_T, 1))

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        """
        Most likely state sequence via Viterbi.

        Returns a 1D integer array of length `n_samples`, concatenating all
        sequences provided.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = _split_sequences(X, lengths)
        paths: List[np.ndarray] = []
        for seq in sequences:
            log_b = self._log_emission(seq, self.params.means, self.params.covars)
            path = self._viterbi(self.params.start_probs, self.params.trans_mat, log_b)
            paths.append(path)
        return np.concatenate(paths, axis=0)

    # ------------------------------------------------------------------
    def predict_proba(
        self, X: np.ndarray, lengths: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """Posterior state probabilities ("gamma").

        Returns an array of shape (n_samples, n_components), concatenating
        all sequences.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = _split_sequences(X, lengths)
        gammas: List[np.ndarray] = []
        for seq in sequences:
            log_b = self._log_emission(seq, self.params.means, self.params.covars)
            log_alpha, loglik = self._forward_log(self.params.start_probs, self.params.trans_mat, log_b)
            log_beta = self._backward_log(self.params.trans_mat, log_b)
            gamma = np.exp(log_alpha + log_beta - loglik)
            gammas.append(gamma)
        return np.vstack(gammas)

    # ------------------------------------------------------------------
    def sample(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single synthetic sequence of length `T`.

        Returns
        -------
        states: np.ndarray, shape = (T,)
            States of the trajectory.
        y: np.ndarray, shape = (T, n_features)
            Observations of the trajectory.
        """
        self._check_fitted()

        return sample_gaussian_hmm(self.params, T, self.random_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_fitted(self):
        if self.params is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def _init_params(
        self,
        sequences: List[np.ndarray],
        n_features: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_states = self.n_components
        # startprob: slightly perturbed uniform
        start_probs = np.ones(n_states) / n_states
        start_probs = start_probs + rng.random(n_states) * 1e-3
        start_probs /= start_probs.sum()

        # transitions: random row-stochastic with small bias toward self-transitions
        trans_mat = rng.random((n_states, n_states)) + np.eye(n_states) * n_states
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)

        # stack all data for initialization
        X_all = np.vstack(sequences)

        # means
        if self.init == "kmeans":
            km = KMeans(n_clusters=n_states, n_init=10, random_state=int(rng.integers(0, 2**32 - 1)))
            labels = km.fit_predict(X_all)
            means = km.cluster_centers_.astype(float)
        else:
            # random selection of observations
            idx = rng.choice(len(X_all), size=n_states, replace=False)
            means = X_all[idx].copy()

        # covariances
        if self.covariance_type == "full":
            covars = np.zeros((n_states, n_features, n_features), dtype=float)
            global_cov = np.cov(X_all.T, bias=True) + self.reg_covar * np.eye(n_features)
            for k in range(n_states):
                covars[k] = global_cov.copy()
        else:  # diag
            var = np.var(X_all, axis=0) + self.reg_covar
            covars = np.tile(var, (n_states, 1))

        return start_probs, trans_mat, means, covars

    # ------------------------------------------------------------------
    @staticmethod
    def _log_emission(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
        """Compute log p(x_t | state=j) for all t, j.

        Returns array shape (T, K).
        """
        T, n_features = X.shape
        n_states = means.shape[0]
        log_b = np.empty((T, n_states), dtype=float)

        if covars.ndim == 3:  # full
            for j in range(n_states):
                mu = means[j]
                S = covars[j]
                # ensure symmetry and PD with small jitter already included
                try:
                    L = np.linalg.cholesky(S)
                except np.linalg.LinAlgError:
                    # add jitter if needed
                    S = S + 1e-8 * np.eye(n_features)
                    L = np.linalg.cholesky(S)
                diff = X - mu
                # solve L y = diff^T  => y = L^{-1} diff^T
                sol = np.linalg.solve(L, diff.T)
                quad = np.sum(sol**2, axis=0)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                log_norm = -0.5 * (n_features * np.log(2 * np.pi) + log_det + quad)
                log_b[:, j] = log_norm
        else:  # diag
            for j in range(n_states):
                mu = means[j]
                var = covars[j]
                inv_var = 1.0 / var
                diff2 = (X - mu) ** 2
                quad = np.sum(diff2 * inv_var, axis=1)
                log_det = np.sum(np.log(var))
                log_b[:, j] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + quad)

        return log_b

    # ------------------------------------------------------------------
    @staticmethod
    def _forward_log(start_probs: np.ndarray, trans_mat: np.ndarray, log_b: np.ndarray) -> Tuple[np.ndarray, float]:
        T, n_states = log_b.shape
        log_alpha = np.empty((T, n_states), dtype=float)
        log_start = np.log(start_probs)
        log_trans = np.log(trans_mat)

        log_alpha[0] = log_start + log_b[0]
        for t in range(1, T):
            # logsumexp over previous states i
            tmp = log_alpha[t - 1][:, None] + log_trans  # shape (n_states, n_states)
            log_alpha[t] = _logsumexp(tmp, axis=0) + log_b[t]
        loglik = float(_logsumexp(log_alpha[-1], axis=0))
        return log_alpha, loglik

    @staticmethod
    def _backward_log(trans_mat: np.ndarray, log_b: np.ndarray) -> np.ndarray:
        T, n_states = log_b.shape
        log_beta = np.empty((T, n_states), dtype=float)
        log_trans = np.log(trans_mat)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            tmp = log_trans + log_b[t + 1] + log_beta[t + 1]  # (n_states, n_states) by broadcast
            log_beta[t] = _logsumexp(tmp, axis=1)
        return log_beta

    # ------------------------------------------------------------------
    @staticmethod
    def _viterbi(start_probs: np.ndarray, trans_mat: np.ndarray, log_b: np.ndarray) -> np.ndarray:
        T, n_states = log_b.shape
        log_start = np.log(start_probs)
        log_trans = np.log(trans_mat)
        delta = np.empty((T, n_states), dtype=float)
        psi = np.empty((T, n_states), dtype=int)

        delta[0] = log_start + log_b[0]
        psi[0] = 0
        for t in range(1, T):
            tmp = delta[t - 1][:, None] + log_trans  # (n_states, n_states)
            psi[t] = np.argmax(tmp, axis=0)
            delta[t] = tmp[psi[t], np.arange(n_states)] + log_b[t]
        states = np.empty(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = int(psi[t + 1, states[t + 1]])
        return states

    # ------------------------------------------------------------------
    def _e_step(
        self,
        sequences: List[np.ndarray],
        start_probs: np.ndarray,
        trans_mat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
    ):
        n_states = start_probs.shape[0]
        n_features = means.shape[1]

        # Sufficient statistics
        stats = {
            "post": np.zeros(n_states),  # sum_t gamma_tk
            "x": np.zeros((n_states, n_features)),  # sum_t gamma_tk * x_t
            "xx": np.zeros((n_states, n_features, n_features)) if covars.ndim == 3 else np.zeros((n_states, n_features)),  # sum_t gamma_tk * x_t x_t^T
            "start": np.zeros(n_states),  # expected start states
            "trans": np.zeros((n_states, n_states)),  # sum over xi_tij
        }

        total_loglik = 0.0

        for seq in sequences:
            T = len(seq)
            log_b = self._log_emission(seq, means, covars)
            log_alpha, loglik = self._forward_log(start_probs, trans_mat, log_b)
            log_beta = self._backward_log(trans_mat, log_b)
            total_loglik += loglik

            # gamma: (T, n_states)
            gamma = np.exp(log_alpha + log_beta - loglik)

            stats["post"] += gamma.sum(axis=0)
            stats["x"] += gamma.T @ seq
            if covars.ndim == 3:  # full
                for k in range(n_states):
                    diff = seq - means[k]
                    # accumulate weighted scatter
                    stats["xx"][k] += (diff * gamma[:, [k]]).T @ diff
            else:  # diag
                for k in range(n_states):
                    diff2 = (seq - means[k]) ** 2
                    stats["xx"][k] += np.sum(gamma[:, [k]] * diff2, axis=0)

            stats["start"] += gamma[0]

            # xi for transitions; handle T<=1 gracefully
            if T > 1:
                log_trans = np.log(trans_mat)
                # shape (T-1, K, K)
                log_xi = (
                    log_alpha[:-1, :, None]
                    + log_trans[None, :, :]
                    + log_b[1:, None, :]
                    + log_beta[1:, None, :]
                    - loglik
                )
                xi = np.exp(log_xi)
                stats["trans"] += xi.sum(axis=0)

        return float(total_loglik), stats

    # ------------------------------------------------------------------
    def _m_step(self, stats, n_features: int):
        n_states = self.n_components
        reg = self.reg_covar

        start_probs = stats["start"].copy()
        start_probs = self._normalize(start_probs)

        trans_mat = stats["trans"].copy()
        # Row-normalize; add small epsilon to avoid zeros
        trans_mat += 1e-12
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)
        # ensure rows sum to 1 even if a state was never visited
        bad = ~np.isfinite(trans_mat).any(axis=1)
        if bad.any():
            trans_mat[bad] = 1.0 / n_states

        post = stats["post"] + 1e-12  # avoid divide-by-zero if a state unused

        means = stats["x"] / post[:, None]

        if stats["xx"].ndim == 3:  # full covariance
            covars = np.empty((n_states, n_features, n_features), dtype=float)
            for k in range(n_states):
                cov_k = stats["xx"][k] / post[k]
                # add reg on diagonal
                cov_k.flat[:: n_features + 1] += reg
                # symmetrize for numerical safety
                covars[k] = 0.5 * (cov_k + cov_k.T)
        else:  # diag
            var = stats["xx"] / post[:, None]
            var = np.maximum(var, reg)
            covars = var

        return start_probs, trans_mat, means, covars

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        s = v.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback to uniform
            return np.ones_like(v) / len(v)
        return v / s

    # ------------------------------------------------------------------
    def _sample_gaussian(self, state: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.params.means[state]
        if self.params.covars.ndim == 3:  # full
            S = self.params.covars[state]
            return rng.multivariate_normal(mu, S)
        else:  # diag
            var = self.params.covars[state]
            return mu + rng.normal(size=mu.shape) * np.sqrt(var)