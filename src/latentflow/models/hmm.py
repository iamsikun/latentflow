from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

from latentflow.core import (
    backward_log,
    forward_log,
    logsumexp,
    normalize,
    run_em,
    split_sequences,
    viterbi,
)
from latentflow.params import (
    GaussianARHMMParams,
    GaussianHMMParams,
    GMMARHMMParams,
    GMMHMMParams,
)
from latentflow.sampler import (
    sample_gaussian_arhmm,
    sample_gaussian_hmm,
    sample_gmm_arhmm,
    sample_gmm_hmm,
)


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
        self.history: list[float] = []
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

        sequences = split_sequences(X, lengths)

        def init_fn(rng: np.random.Generator) -> Dict[str, np.ndarray]:
            return self._init_params(sequences, n_features, rng)

        def e_step_fn(params: Dict[str, np.ndarray]):
            return self._e_step(
                sequences,
                params["start_probs"],
                params["trans_mat"],
                params["means"],
                params["covars"],
            )

        def m_step_fn(stats, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            return self._m_step(stats, n_features)

        best_params, self.converged, self.loglik, self.history = run_em(
            sequences,
            init_params=init_fn,
            e_step=e_step_fn,
            m_step=m_step_fn,
            n_init=self.n_init,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=verbose,
        )

        params_dict = dict(best_params)
        self.params = GaussianHMMParams(
            start_probs=params_dict["start_probs"],
            trans_mat=params_dict["trans_mat"],
            means=params_dict["means"],
            covars=params_dict["covars"],
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
        sequences = split_sequences(X, lengths)
        loglik = 0.0
        total_T = 0
        for seq in sequences:
            log_b = self._log_emission(seq, self.params.means, self.params.covars)
            _, loglik_seq = forward_log(self.params.start_probs, self.params.trans_mat, log_b)
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
        sequences = split_sequences(X, lengths)
        paths: List[np.ndarray] = []
        for seq in sequences:
            log_b = self._log_emission(seq, self.params.means, self.params.covars)
            path = viterbi(self.params.start_probs, self.params.trans_mat, log_b)
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
        sequences = split_sequences(X, lengths)
        gammas: List[np.ndarray] = []
        for seq in sequences:
            log_b = self._log_emission(seq, self.params.means, self.params.covars)
            log_alpha, loglik = forward_log(self.params.start_probs, self.params.trans_mat, log_b)
            log_beta = backward_log(self.params.trans_mat, log_b)
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
    ) -> Dict[str, np.ndarray]:
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
            km.fit(X_all)
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

        return {
            "start_probs": start_probs,
            "trans_mat": trans_mat,
            "means": means,
            "covars": covars,
        }

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
            "xx": np.zeros((n_states, n_features, n_features))
            if covars.ndim == 3
            else np.zeros((n_states, n_features)),  # sum_t gamma_tk * x_t x_t^T
            "start": np.zeros(n_states),  # expected start states
            "trans": np.zeros((n_states, n_states)),  # sum over xi_tij
        }

        total_loglik = 0.0

        for seq in sequences:
            T = len(seq)
            log_b = self._log_emission(seq, means, covars)
            log_alpha, loglik = forward_log(start_probs, trans_mat, log_b)
            log_beta = backward_log(trans_mat, log_b)
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
    def _m_step(self, stats, n_features: int) -> Dict[str, np.ndarray]:
        n_states = self.n_components
        reg = self.reg_covar

        start_probs = stats["start"].copy()
        start_probs = normalize(start_probs)

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

        return {
            "start_probs": start_probs,
            "trans_mat": trans_mat,
            "means": means,
            "covars": covars,
        }

    # ------------------------------------------------------------------
    def _sample_gaussian(self, state: int, rng: np.random.Generator) -> np.ndarray:
        mu = self.params.means[state]
        if self.params.covars.ndim == 3:  # full
            S = self.params.covars[state]
            return rng.multivariate_normal(mu, S)
        else:  # diag
            var = self.params.covars[state]
            return mu + rng.normal(size=mu.shape) * np.sqrt(var)


class GaussianARHMM:
    """Autoregressive Hidden Markov Model with Gaussian emissions."""

    def __init__(
        self,
        n_components: int,
        *,
        order: int = 1,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int | np.random.Generator] = None,
        init: str = "kmeans",
        n_init: int = 1,
    ) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if order < 0:
            raise ValueError("order must be non-negative")
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
        self.order = order
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init = init
        self.n_init = n_init

        self.converged: bool | None = None
        self.loglik: float | None = None
        self.history: list[float] = []
        self.params: GaussianARHMMParams | None = None

    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "order": self.order,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
            "init": self.init,
            "n_init": self.n_init,
        }

    def set_params(self, **params) -> "GaussianARHMM":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, verbose: bool = False) -> "GaussianARHMM":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (T, n_features)")
        T, n_features = X.shape
        if self.order * n_features + 1 <= 0:
            raise ValueError("Invalid combination of order and n_features")

        sequences = split_sequences(X, lengths)

        def init_fn(rng: np.random.Generator) -> Dict[str, np.ndarray]:
            return self._init_params(sequences, n_features, rng)

        def e_step_fn(params: Dict[str, np.ndarray]):
            return self._e_step(
                sequences,
                params["start_probs"],
                params["trans_mat"],
                params["coeffs"],
                params["covars"],
            )

        def m_step_fn(stats, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            return self._m_step(stats, n_features)

        best_params, self.converged, self.loglik, self.history = run_em(
            sequences,
            init_params=init_fn,
            e_step=e_step_fn,
            m_step=m_step_fn,
            n_init=self.n_init,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=verbose,
        )

        params_dict = dict(best_params)
        self.params = GaussianARHMMParams(
            start_probs=params_dict["start_probs"],
            trans_mat=params_dict["trans_mat"],
            coeffs=params_dict["coeffs"],
            covars=params_dict["covars"],
            order=self.order,
        )
        return self

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> float:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = split_sequences(X, lengths)
        loglik = 0.0
        total_T = 0
        for seq in sequences:
            Z = self._design_matrix(seq)
            log_b = self._log_emission(seq, Z, self.params.coeffs, self.params.covars)
            _, loglik_seq = forward_log(self.params.start_probs, self.params.trans_mat, log_b)
            loglik += loglik_seq
            total_T += len(seq)
        return float(loglik / max(total_T, 1))

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = split_sequences(X, lengths)
        paths: List[np.ndarray] = []
        for seq in sequences:
            Z = self._design_matrix(seq)
            log_b = self._log_emission(seq, Z, self.params.coeffs, self.params.covars)
            path = viterbi(self.params.start_probs, self.params.trans_mat, log_b)
            paths.append(path)
        return np.concatenate(paths, axis=0)

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = split_sequences(X, lengths)
        gammas: List[np.ndarray] = []
        for seq in sequences:
            Z = self._design_matrix(seq)
            log_b = self._log_emission(seq, Z, self.params.coeffs, self.params.covars)
            log_alpha, loglik = forward_log(self.params.start_probs, self.params.trans_mat, log_b)
            log_beta = backward_log(self.params.trans_mat, log_b)
            gamma = np.exp(log_alpha + log_beta - loglik)
            gammas.append(gamma)
        return np.vstack(gammas)

    # ------------------------------------------------------------------
    def sample(
        self,
        T: int,
        *,
        history: Optional[np.ndarray] = None,
        s0: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        return sample_gaussian_arhmm(self.params, T, self.random_state, s0=s0, history=history)

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.params is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    # ------------------------------------------------------------------
    def _design_matrix(self, seq: np.ndarray) -> np.ndarray:
        T, n_features = seq.shape
        order = self.order
        width = order * n_features + 1
        Z = np.zeros((T, width), dtype=float)
        if order == 0:
            Z[:, -1] = 1.0
            return Z
        for t in range(T):
            for lag in range(1, order + 1):
                idx = t - lag
                start = (lag - 1) * n_features
                if idx >= 0:
                    Z[t, start : start + n_features] = seq[idx]
                else:
                    # zero padding for unavailable history
                    Z[t, start : start + n_features] = 0.0
            Z[t, -1] = 1.0
        return Z

    # ------------------------------------------------------------------
    def _log_emission(
        self,
        seq: np.ndarray,
        Z: np.ndarray,
        coeffs: np.ndarray,
        covars: np.ndarray,
    ) -> np.ndarray:
        T = len(seq)
        n_states = coeffs.shape[0]
        log_b = np.empty((T, n_states), dtype=float)

        for k in range(n_states):
            B = coeffs[k]  # (n_features, width)
            mu = Z @ B.T
            cov = covars[k]
            log_b[:, k] = self._log_gaussian_density(seq, mu, cov)
        return log_b

    # ------------------------------------------------------------------
    def _e_step(
        self,
        sequences: List[np.ndarray],
        start_probs: np.ndarray,
        trans_mat: np.ndarray,
        coeffs: np.ndarray,
        covars: np.ndarray,
    ):
        n_states = start_probs.shape[0]
        n_features = coeffs.shape[1]
        width = coeffs.shape[2]

        stats = {
            "post": np.zeros(n_states),
            "start": np.zeros(n_states),
            "trans": np.zeros((n_states, n_states)),
            "zz": np.zeros((n_states, width, width)),
            "yz": np.zeros((n_states, n_features, width)),
            "yy": np.zeros((n_states, n_features, n_features)),
        }

        total_loglik = 0.0

        for seq in sequences:
            Z = self._design_matrix(seq)
            log_b = self._log_emission(seq, Z, coeffs, covars)
            log_alpha, loglik = forward_log(start_probs, trans_mat, log_b)
            log_beta = backward_log(trans_mat, log_b)
            total_loglik += loglik

            gamma = np.exp(log_alpha + log_beta - loglik)

            stats["post"] += gamma.sum(axis=0)
            stats["start"] += gamma[0]

            if len(seq) > 1:
                log_trans = np.log(trans_mat)
                log_xi = (
                    log_alpha[:-1, :, None]
                    + log_trans[None, :, :]
                    + log_b[1:, None, :]
                    + log_beta[1:, None, :]
                    - loglik
                )
                xi = np.exp(log_xi)
                stats["trans"] += xi.sum(axis=0)

            for k in range(n_states):
                gamma_k = gamma[:, k][:, None]
                weighted_Z = Z * gamma_k
                weighted_seq = seq * gamma_k
                stats["zz"][k] += Z.T @ weighted_Z
                stats["yz"][k] += weighted_seq.T @ Z
                stats["yy"][k] += weighted_seq.T @ seq

        return float(total_loglik), stats

    # ------------------------------------------------------------------
    def _m_step(self, stats, n_features: int) -> Dict[str, np.ndarray]:
        n_states = self.n_components
        width = self.order * n_features + 1
        reg = self.reg_covar

        start_probs = stats["start"].copy()
        start_probs = normalize(start_probs)

        trans_mat = stats["trans"].copy()
        trans_mat += 1e-12
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)
        bad = ~np.isfinite(trans_mat).any(axis=1)
        if bad.any():
            trans_mat[bad] = 1.0 / n_states

        coeffs = np.zeros((n_states, n_features, width), dtype=float)
        if self.covariance_type == "full":
            covars = np.zeros((n_states, n_features, n_features), dtype=float)
        else:
            covars = np.zeros((n_states, n_features), dtype=float)

        post = stats["post"] + 1e-12

        eye = np.eye(width)

        for k in range(n_states):
            zz = stats["zz"][k] + reg * eye
            yz = stats["yz"][k]
            try:
                B = yz @ np.linalg.inv(zz)
            except np.linalg.LinAlgError:
                B = yz @ np.linalg.pinv(zz)

            coeffs[k] = B

            cov_k = stats["yy"][k] - B @ stats["yz"][k].T - stats["yz"][k] @ B.T + B @ stats["zz"][k] @ B.T
            cov_k /= post[k]
            cov_k = 0.5 * (cov_k + cov_k.T)
            cov_k.flat[:: n_features + 1] += reg

            if self.covariance_type == "full":
                covars[k] = cov_k
            else:
                covars[k] = np.maximum(np.diag(cov_k), reg)

        return {
            "start_probs": start_probs,
            "trans_mat": trans_mat,
            "coeffs": coeffs,
            "covars": covars,
        }

    # ------------------------------------------------------------------
    def _init_params(
        self,
        sequences: List[np.ndarray],
        n_features: int,
        rng: np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        n_states = self.n_components
        order = self.order
        width = order * n_features + 1

        start_probs = np.ones(n_states) / n_states
        start_probs = start_probs + rng.random(n_states) * 1e-3
        start_probs /= start_probs.sum()

        trans_mat = rng.random((n_states, n_states)) + np.eye(n_states) * n_states
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)

        X_all = np.vstack(sequences)

        if self.init == "kmeans":
            km = KMeans(n_clusters=n_states, n_init=10, random_state=int(rng.integers(0, 2**32 - 1)))
            km.fit(X_all)
            means = km.cluster_centers_.astype(float)
        else:
            idx = rng.choice(len(X_all), size=n_states, replace=False)
            means = X_all[idx].copy()

        coeffs = np.zeros((n_states, n_features, width), dtype=float)
        coeffs[:, :, -1] = means

        if self.covariance_type == "full":
            covars = np.zeros((n_states, n_features, n_features), dtype=float)
            global_cov = np.cov(X_all.T, bias=True) + self.reg_covar * np.eye(n_features)
            for k in range(n_states):
                covars[k] = global_cov.copy()
        else:
            var = np.var(X_all, axis=0) + self.reg_covar
            covars = np.tile(var, (n_states, 1))

        return {
            "start_probs": start_probs,
            "trans_mat": trans_mat,
            "coeffs": coeffs,
            "covars": covars,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _log_gaussian_density(y: np.ndarray, mu: np.ndarray, cov) -> np.ndarray:
        n_features = y.shape[1]
        if np.ndim(cov) == 1:  # diagonal covariance
            var = cov
            inv_var = 1.0 / var
            diff = y - mu
            quad = np.sum(diff**2 * inv_var, axis=1)
            log_det = np.sum(np.log(var))
            return -0.5 * (n_features * np.log(2 * np.pi) + log_det + quad)

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]))

        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        log_prob = np.empty(y.shape[0], dtype=float)
        for t in range(y.shape[0]):
            diff = y[t] - mu[t]
            sol = np.linalg.solve(L, diff)
            quad = sol @ sol
            log_prob[t] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + quad)
        return log_prob


class GMMHMM:
    """Hidden Markov Model with Gaussian mixture emissions."""

    def __init__(
        self,
        n_components: int,
        *,
        n_mixtures: int,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int | np.random.Generator] = None,
        init: str = "kmeans",
        n_init: int = 1,
    ) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if n_mixtures <= 0:
            raise ValueError("n_mixtures must be positive")
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
        self.n_mixtures = n_mixtures
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init = init
        self.n_init = n_init

        self.converged: bool | None = None
        self.loglik: float | None = None
        self.history: list[float] = []
        self.params: GMMHMMParams | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "n_mixtures": self.n_mixtures,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
            "init": self.init,
            "n_init": self.n_init,
        }

    def set_params(self, **params) -> GMMHMM:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(
        self,
        X: np.ndarray,
        lengths: Optional[Sequence[int]] = None,
        verbose: bool = False,
    ) -> GMMHMM:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (T, n_features)")

        sequences = split_sequences(X, lengths)
        _, n_features = X.shape

        def init_fn(rng: np.random.Generator) -> Dict[str, np.ndarray]:
            return self._init_params(sequences, n_features, rng)

        def e_step_fn(params: Dict[str, np.ndarray]):
            return self._e_step(
                sequences,
                params["start_probs"],
                params["trans_mat"],
                params["weights"],
                params["means"],
                params["covars"],
            )

        def m_step_fn(stats, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            return self._m_step(stats, n_features)

        best_params, self.converged, self.loglik, self.history = run_em(
            sequences,
            init_params=init_fn,
            e_step=e_step_fn,
            m_step=m_step_fn,
            n_init=self.n_init,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=verbose,
        )

        params_dict = dict(best_params)
        self.params = GMMHMMParams(
            start_probs=params_dict["start_probs"],
            trans_mat=params_dict["trans_mat"],
            weights=params_dict["weights"],
            means=params_dict["means"],
            covars=params_dict["covars"],
        )
        return self

    def score(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> float:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = split_sequences(X, lengths)
        loglik = 0.0
        total_T = 0
        for seq in sequences:
            log_b = self._log_emission(
                seq,
                self.params.means,
                self.params.covars,
                self.params.weights,
            )
            _, loglik_seq = forward_log(
                self.params.start_probs, self.params.trans_mat, log_b
            )
            loglik += loglik_seq
            total_T += len(seq)
        return float(loglik / max(total_T, 1))

    def predict(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = split_sequences(X, lengths)
        paths: List[np.ndarray] = []
        for seq in sequences:
            log_b = self._log_emission(
                seq,
                self.params.means,
                self.params.covars,
                self.params.weights,
            )
            path = viterbi(self.params.start_probs, self.params.trans_mat, log_b)
            paths.append(path)
        return np.concatenate(paths, axis=0)

    def predict_proba(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = split_sequences(X, lengths)
        gammas: List[np.ndarray] = []
        for seq in sequences:
            log_b = self._log_emission(
                seq,
                self.params.means,
                self.params.covars,
                self.params.weights,
            )
            log_alpha, loglik = forward_log(
                self.params.start_probs, self.params.trans_mat, log_b
            )
            log_beta = backward_log(self.params.trans_mat, log_b)
            gamma = np.exp(log_alpha + log_beta - loglik)
            gammas.append(gamma)
        return np.vstack(gammas)

    def sample(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        return sample_gmm_hmm(self.params, T, self.random_state)

    def _check_fitted(self) -> None:
        if self.params is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def _init_params(
        self,
        sequences: List[np.ndarray],
        n_features: int,
        rng: np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        n_states = self.n_components
        n_mix = self.n_mixtures

        start_probs = np.ones(n_states) / n_states
        start_probs = start_probs + rng.random(n_states) * 1e-3
        start_probs = normalize(start_probs)

        trans_mat = rng.random((n_states, n_states)) + np.eye(n_states) * n_states
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)

        X_all = np.vstack(sequences)
        n_clusters = n_states * n_mix
        centers: np.ndarray
        if self.init == "kmeans" and len(X_all) >= n_clusters:
            km = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            km.fit(X_all)
            centers = km.cluster_centers_.astype(float)
        else:
            replace = len(X_all) < n_clusters
            idx = rng.choice(len(X_all), size=n_clusters, replace=replace)
            centers = X_all[idx].astype(float)

        means = centers.reshape(n_states, n_mix, n_features)
        weights = np.ones((n_states, n_mix), dtype=float) / n_mix

        if self.covariance_type == "full":
            covars = np.zeros((n_states, n_mix, n_features, n_features), dtype=float)
            global_cov = np.cov(X_all.T, bias=True) + self.reg_covar * np.eye(n_features)
            for k in range(n_states):
                for m in range(n_mix):
                    covars[k, m] = global_cov.copy()
        else:
            var = np.var(X_all, axis=0) + self.reg_covar
            covars = np.tile(var, (n_states, n_mix, 1))

        return {
            "start_probs": start_probs,
            "trans_mat": trans_mat,
            "weights": weights,
            "means": means,
            "covars": covars,
        }

    def _log_emission(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
        weights: np.ndarray,
        *,
        return_resp: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        T, _ = X.shape
        n_states = means.shape[0]
        n_mix = means.shape[1]
        log_b = np.empty((T, n_states), dtype=float)
        resp = np.empty((T, n_states, n_mix), dtype=float) if return_resp else None

        for k in range(n_states):
            log_comp = np.empty((T, n_mix), dtype=float)
            log_w = np.log(np.maximum(weights[k], 1e-12))
            if covars.ndim == 4:
                for m in range(n_mix):
                    log_comp[:, m] = log_w[m] + self._log_gaussian_full(
                        X, means[k, m], covars[k, m]
                    )
            else:
                for m in range(n_mix):
                    log_comp[:, m] = log_w[m] + self._log_gaussian_diag(
                        X, means[k, m], covars[k, m]
                    )
            log_b[:, k] = logsumexp(log_comp, axis=1)
            if return_resp and resp is not None:
                resp[:, k, :] = np.exp(log_comp - log_b[:, k][:, None])

        if return_resp and resp is not None:
            return log_b, resp
        return log_b

    @staticmethod
    def _log_gaussian_diag(X: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        var_safe = np.maximum(var, 1e-12)
        diff = X - mean
        inv_var = 1.0 / var_safe
        quad = np.sum(diff**2 * inv_var, axis=1)
        log_det = np.sum(np.log(var_safe))
        n_features = X.shape[1]
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + quad)

    @staticmethod
    def _log_gaussian_full(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov + 1e-8 * np.eye(n_features))
        diff = X - mean
        sol = np.linalg.solve(L, diff.T)
        quad = np.sum(sol**2, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (n_features * np.log(2 * np.pi) + log_det + quad)

    def _e_step(
        self,
        sequences: List[np.ndarray],
        start_probs: np.ndarray,
        trans_mat: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
    ):
        n_states = start_probs.shape[0]
        n_mix = weights.shape[1]
        n_features = means.shape[2]

        stats = {
            "post": np.zeros(n_states),
            "start": np.zeros(n_states),
            "trans": np.zeros((n_states, n_states)),
            "mix_post": np.zeros((n_states, n_mix)),
            "mix_x": np.zeros((n_states, n_mix, n_features)),
            "mix_xx": np.zeros((n_states, n_mix, n_features, n_features)),
        }

        total_loglik = 0.0

        for seq in sequences:
            log_b, resp = self._log_emission(
                seq, means, covars, weights, return_resp=True
            )
            log_alpha, loglik = forward_log(start_probs, trans_mat, log_b)
            log_beta = backward_log(trans_mat, log_b)
            total_loglik += loglik

            gamma = np.exp(log_alpha + log_beta - loglik)

            stats["post"] += gamma.sum(axis=0)
            stats["start"] += gamma[0]

            if len(seq) > 1:
                log_trans = np.log(trans_mat)
                log_xi = (
                    log_alpha[:-1, :, None]
                    + log_trans[None, :, :]
                    + log_b[1:, None, :]
                    + log_beta[1:, None, :]
                    - loglik
                )
                xi = np.exp(log_xi)
                stats["trans"] += xi.sum(axis=0)

            for k in range(n_states):
                gamma_k = gamma[:, k][:, None]
                resp_k = resp[:, k, :]
                gamma_mix = gamma_k * resp_k
                stats["mix_post"][k] += gamma_mix.sum(axis=0)
                for m in range(n_mix):
                    weights_tm = gamma_mix[:, m][:, None]
                    weighted_seq = seq * weights_tm
                    stats["mix_x"][k, m] += weighted_seq.sum(axis=0)
                    stats["mix_xx"][k, m] += weighted_seq.T @ seq

        return float(total_loglik), stats

    def _m_step(self, stats, n_features: int) -> Dict[str, np.ndarray]:
        n_states = self.n_components
        n_mix = self.n_mixtures
        reg = self.reg_covar

        start_probs = normalize(stats["start"].copy())

        trans_mat = stats["trans"].copy()
        trans_mat += 1e-12
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)
        bad = ~np.isfinite(trans_mat).any(axis=1)
        if bad.any():
            trans_mat[bad] = 1.0 / n_states

        weights = stats["mix_post"].copy() + 1e-12
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_sum[weights_sum <= 0] = 1.0
        weights = weights / weights_sum
        if not np.all(np.isfinite(weights)):
            weights = np.ones_like(weights) / n_mix

        post = stats["mix_post"] + 1e-12
        means = stats["mix_x"] / post[:, :, None]

        if self.covariance_type == "full":
            covars = np.zeros((n_states, n_mix, n_features, n_features), dtype=float)
        else:
            covars = np.zeros((n_states, n_mix, n_features), dtype=float)

        for k in range(n_states):
            for m in range(n_mix):
                denom = post[k, m]
                mean = means[k, m]
                cov_k = stats["mix_xx"][k, m] / denom - np.outer(mean, mean)
                cov_k = 0.5 * (cov_k + cov_k.T)
                if self.covariance_type == "full":
                    cov_k.flat[:: n_features + 1] += reg
                    covars[k, m] = cov_k
                else:
                    diag_cov = np.maximum(np.diag(cov_k), reg)
                    covars[k, m] = diag_cov

        return {
            "start_probs": start_probs,
            "trans_mat": trans_mat,
            "weights": weights,
            "means": means,
            "covars": covars,
        }


class GMMARHMM:
    """Autoregressive HMM with Gaussian mixture emissions."""

    def __init__(
        self,
        n_components: int,
        *,
        n_mixtures: int,
        order: int = 1,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: Optional[int | np.random.Generator] = None,
        init: str = "kmeans",
        n_init: int = 1,
    ) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if n_mixtures <= 0:
            raise ValueError("n_mixtures must be positive")
        if order < 0:
            raise ValueError("order must be non-negative")
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
        self.n_mixtures = n_mixtures
        self.order = order
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init = init
        self.n_init = n_init

        self.converged: bool | None = None
        self.loglik: float | None = None
        self.history: list[float] = []
        self.params: GMMARHMMParams | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "n_mixtures": self.n_mixtures,
            "order": self.order,
            "covariance_type": self.covariance_type,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
            "init": self.init,
            "n_init": self.n_init,
        }

    def set_params(self, **params) -> GMMARHMM:
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(
        self,
        X: np.ndarray,
        lengths: Optional[Sequence[int]] = None,
        verbose: bool = False,
    ) -> GMMARHMM:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D array of shape (T, n_features)")

        sequences = split_sequences(X, lengths)
        _, n_features = X.shape

        def init_fn(rng: np.random.Generator) -> Dict[str, np.ndarray]:
            return self._init_params(sequences, n_features, rng)

        def e_step_fn(params: Dict[str, np.ndarray]):
            return self._e_step(
                sequences,
                params["start_probs"],
                params["trans_mat"],
                params["weights"],
                params["coeffs"],
                params["covars"],
            )

        def m_step_fn(stats, params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            return self._m_step(stats, n_features)

        best_params, self.converged, self.loglik, self.history = run_em(
            sequences,
            init_params=init_fn,
            e_step=e_step_fn,
            m_step=m_step_fn,
            n_init=self.n_init,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=verbose,
        )

        params_dict = dict(best_params)
        self.params = GMMARHMMParams(
            start_probs=params_dict["start_probs"],
            trans_mat=params_dict["trans_mat"],
            weights=params_dict["weights"],
            coeffs=params_dict["coeffs"],
            covars=params_dict["covars"],
            order=self.order,
        )
        return self

    def score(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> float:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = split_sequences(X, lengths)
        loglik = 0.0
        total_T = 0
        for seq in sequences:
            Z = self._design_matrix(seq)
            log_b = self._log_emission(
                seq,
                Z,
                self.params.coeffs,
                self.params.covars,
                self.params.weights,
            )
            _, loglik_seq = forward_log(
                self.params.start_probs, self.params.trans_mat, log_b
            )
            loglik += loglik_seq
            total_T += len(seq)
        return float(loglik / max(total_T, 1))

    def predict(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = split_sequences(X, lengths)
        paths: List[np.ndarray] = []
        for seq in sequences:
            Z = self._design_matrix(seq)
            log_b = self._log_emission(
                seq,
                Z,
                self.params.coeffs,
                self.params.covars,
                self.params.weights,
            )
            path = viterbi(self.params.start_probs, self.params.trans_mat, log_b)
            paths.append(path)
        return np.concatenate(paths, axis=0)

    def predict_proba(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        sequences = split_sequences(X, lengths)
        gammas: List[np.ndarray] = []
        for seq in sequences:
            Z = self._design_matrix(seq)
            log_b = self._log_emission(
                seq,
                Z,
                self.params.coeffs,
                self.params.covars,
                self.params.weights,
            )
            log_alpha, loglik = forward_log(
                self.params.start_probs, self.params.trans_mat, log_b
            )
            log_beta = backward_log(self.params.trans_mat, log_b)
            gamma = np.exp(log_alpha + log_beta - loglik)
            gammas.append(gamma)
        return np.vstack(gammas)

    def sample(
        self,
        T: int,
        *,
        history: Optional[np.ndarray] = None,
        s0: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        return sample_gmm_arhmm(
            self.params, T, self.random_state, s0=s0, history=history
        )

    def _check_fitted(self) -> None:
        if self.params is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def _init_params(
        self,
        sequences: List[np.ndarray],
        n_features: int,
        rng: np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        n_states = self.n_components
        n_mix = self.n_mixtures
        width = self.order * n_features + 1

        start_probs = np.ones(n_states, dtype=float) / n_states
        start_probs = start_probs + rng.random(n_states) * 1e-3
        start_probs = normalize(start_probs)

        trans_mat = rng.random((n_states, n_states)) + np.eye(n_states) * n_states
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)

        X_all = np.vstack(sequences)

        n_clusters = n_states * n_mix
        if self.init == "kmeans" and len(X_all) >= n_clusters:
            km = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            km.fit(X_all)
            centers = km.cluster_centers_.astype(float)
        else:
            replace = len(X_all) < n_clusters
            idx = rng.choice(len(X_all), size=n_clusters, replace=replace)
            centers = X_all[idx].astype(float)

        weights = np.ones((n_states, n_mix), dtype=float) / n_mix

        coeffs = np.zeros((n_states, n_mix, n_features, width), dtype=float)
        coeffs[:, :, :, -1] = centers.reshape(n_states, n_mix, n_features)

        if self.covariance_type == "full":
            covars = np.zeros((n_states, n_mix, n_features, n_features), dtype=float)
            global_cov = np.cov(X_all.T, bias=True) + self.reg_covar * np.eye(n_features)
            for k in range(n_states):
                for m in range(n_mix):
                    covars[k, m] = global_cov.copy()
        else:
            var = np.var(X_all, axis=0) + self.reg_covar
            covars = np.tile(var, (n_states, n_mix, 1))

        return {
            "start_probs": start_probs,
            "trans_mat": trans_mat,
            "weights": weights,
            "coeffs": coeffs,
            "covars": covars,
        }

    def _design_matrix(self, seq: np.ndarray) -> np.ndarray:
        T, n_features = seq.shape
        order = self.order
        width = order * n_features + 1
        Z = np.zeros((T, width), dtype=float)
        if order == 0:
            Z[:, -1] = 1.0
            return Z
        for t in range(T):
            for lag in range(1, order + 1):
                idx = t - lag
                start = (lag - 1) * n_features
                if idx >= 0:
                    Z[t, start : start + n_features] = seq[idx]
                else:
                    Z[t, start : start + n_features] = 0.0
            Z[t, -1] = 1.0
        return Z

    def _log_emission(
        self,
        seq: np.ndarray,
        Z: np.ndarray,
        coeffs: np.ndarray,
        covars: np.ndarray,
        weights: np.ndarray,
        *,
        return_resp: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        T = len(seq)
        n_states = coeffs.shape[0]
        n_mix = coeffs.shape[1]
        log_b = np.empty((T, n_states), dtype=float)
        resp = np.empty((T, n_states, n_mix), dtype=float) if return_resp else None

        for k in range(n_states):
            log_comp = np.empty((T, n_mix), dtype=float)
            log_w = np.log(np.maximum(weights[k], 1e-12))
            for m in range(n_mix):
                B = coeffs[k, m]
                mu = Z @ B.T
                cov = covars[k, m]
                log_comp[:, m] = log_w[m] + GaussianARHMM._log_gaussian_density(
                    seq, mu, cov
                )
            log_b[:, k] = logsumexp(log_comp, axis=1)
            if return_resp and resp is not None:
                resp[:, k, :] = np.exp(log_comp - log_b[:, k][:, None])

        if return_resp and resp is not None:
            return log_b, resp
        return log_b

    def _e_step(
        self,
        sequences: List[np.ndarray],
        start_probs: np.ndarray,
        trans_mat: np.ndarray,
        weights: np.ndarray,
        coeffs: np.ndarray,
        covars: np.ndarray,
    ):
        n_states = start_probs.shape[0]
        n_mix = weights.shape[1]
        n_features = coeffs.shape[2]
        width = coeffs.shape[3]

        stats = {
            "post": np.zeros(n_states),
            "start": np.zeros(n_states),
            "trans": np.zeros((n_states, n_states)),
            "mix_post": np.zeros((n_states, n_mix)),
            "zz": np.zeros((n_states, n_mix, width, width)),
            "yz": np.zeros((n_states, n_mix, n_features, width)),
            "yy": np.zeros((n_states, n_mix, n_features, n_features)),
        }

        total_loglik = 0.0

        for seq in sequences:
            Z = self._design_matrix(seq)
            log_b, resp = self._log_emission(
                seq, Z, coeffs, covars, weights, return_resp=True
            )
            log_alpha, loglik = forward_log(start_probs, trans_mat, log_b)
            log_beta = backward_log(trans_mat, log_b)
            total_loglik += loglik

            gamma = np.exp(log_alpha + log_beta - loglik)

            stats["post"] += gamma.sum(axis=0)
            stats["start"] += gamma[0]

            if len(seq) > 1:
                log_trans = np.log(trans_mat)
                log_xi = (
                    log_alpha[:-1, :, None]
                    + log_trans[None, :, :]
                    + log_b[1:, None, :]
                    + log_beta[1:, None, :]
                    - loglik
                )
                xi = np.exp(log_xi)
                stats["trans"] += xi.sum(axis=0)

            for k in range(n_states):
                gamma_k = gamma[:, k][:, None]
                resp_k = resp[:, k, :]
                gamma_mix = gamma_k * resp_k
                stats["mix_post"][k] += gamma_mix.sum(axis=0)
                for m in range(n_mix):
                    weights_tm = gamma_mix[:, m][:, None]
                    weighted_Z = Z * weights_tm
                    weighted_seq = seq * weights_tm
                    stats["zz"][k, m] += Z.T @ weighted_Z
                    stats["yz"][k, m] += weighted_seq.T @ Z
                    stats["yy"][k, m] += weighted_seq.T @ seq

        return float(total_loglik), stats

    def _m_step(self, stats, n_features: int) -> Dict[str, np.ndarray]:
        n_states = self.n_components
        n_mix = self.n_mixtures
        width = self.order * n_features + 1
        reg = self.reg_covar

        start_probs = normalize(stats["start"].copy())

        trans_mat = stats["trans"].copy()
        trans_mat += 1e-12
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)
        bad = ~np.isfinite(trans_mat).any(axis=1)
        if bad.any():
            trans_mat[bad] = 1.0 / n_states

        weights = stats["mix_post"].copy() + 1e-12
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_sum[weights_sum <= 0] = 1.0
        weights = weights / weights_sum
        if not np.all(np.isfinite(weights)):
            weights = np.ones_like(weights) / n_mix

        coeffs = np.zeros((n_states, n_mix, n_features, width), dtype=float)
        if self.covariance_type == "full":
            covars = np.zeros((n_states, n_mix, n_features, n_features), dtype=float)
        else:
            covars = np.zeros((n_states, n_mix, n_features), dtype=float)

        post = stats["mix_post"] + 1e-12
        eye = np.eye(width)

        for k in range(n_states):
            for m in range(n_mix):
                zz = stats["zz"][k, m] + reg * eye
                yz = stats["yz"][k, m]
                try:
                    B = yz @ np.linalg.inv(zz)
                except np.linalg.LinAlgError:
                    B = yz @ np.linalg.pinv(zz)
                coeffs[k, m] = B

                cov_k = (
                    stats["yy"][k, m]
                    - B @ stats["yz"][k, m].T
                    - stats["yz"][k, m] @ B.T
                    + B @ stats["zz"][k, m] @ B.T
                )
                cov_k /= post[k, m]
                cov_k = 0.5 * (cov_k + cov_k.T)

                if self.covariance_type == "full":
                    cov_k.flat[:: n_features + 1] += reg
                    covars[k, m] = cov_k
                else:
                    diag_cov = np.maximum(np.diag(cov_k), reg)
                    covars[k, m] = diag_cov

        return {
            "start_probs": start_probs,
            "trans_mat": trans_mat,
            "weights": weights,
            "coeffs": coeffs,
            "covars": covars,
        }

