from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.stats


class RandomVariable(ABC):
    @abstractmethod
    def cdf(self, x) -> float:
        pass

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        pass


class ContinuousRandomVariable(RandomVariable):
    @abstractmethod
    def pdf(self, x) -> float:
        pass


class DiscreteRandomVariable(RandomVariable):
    @abstractmethod
    def pmf(self, x) -> float:
        pass


class UnivariateGaussian(ContinuousRandomVariable):
    def __init__(self, mean: float, std: float):
        if std <= 0:
            raise ValueError(f"std must be a positive number. Got {std}.")

        self.mean = mean
        self.std = std

    def pdf(self, x: float) -> float:
        return (
            1
            / (self.std * np.sqrt(2 * np.pi))
            * np.exp(-((x - self.mean) ** 2) / (2 * self.std**2))
        )

    def cdf(self, x: float) -> float:
        return scipy.stats.norm.cdf(x, loc=self.mean, scale=self.std)

    def sample(self, n: int) -> np.ndarray:
        return np.random.normal(self.mean, self.std, n)


class MultivariateGaussian(ContinuousRandomVariable):
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        if mean.ndim != 1:
            raise ValueError(f"mean must be a 1D array. Got {mean.ndim} dimensions.")
        if cov.ndim != 2:
            raise ValueError(f"cov must be a 2D array. Got {cov.ndim} dimensions.")
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(
                f"cov must be a square matrix. Got {cov.shape[0]}x{cov.shape[1]}."
            )
        if not np.all(np.linalg.eigvals(cov) > 0):
            raise ValueError(
                "cov must be a positive definite matrix. Got a matrix with negative eigenvalues."
            )
        if cov.shape[0] != mean.shape[0]:
            raise ValueError(
                f"mean and cov must have the same number of dimensions. Got {mean.shape[0]} and {cov.shape[0]}."
            )

        self.mean = mean
        self.cov = cov

    def pdf(self, x: np.ndarray) -> float:
        return (
            1
            / (np.sqrt(2 * np.pi) ** self.mean.shape[0])
            * np.exp(
                -0.5 * ((x - self.mean) @ np.linalg.inv(self.cov) @ (x - self.mean).T)
            )
        )

    def cdf(self, x: np.ndarray) -> float:
        return scipy.stats.multivariate_normal.cdf(x, loc=self.mean, cov=self.cov)

    def sample(self, n: int) -> np.ndarray:
        return np.random.multivariate_normal(self.mean, self.cov, n)


class MixtureGaussian(ContinuousRandomVariable):
    """Finite mixture of Gaussians with component weights."""

    def __init__(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
    ) -> None:
        weights = np.asarray(weights, dtype=float)
        means = np.asarray(means, dtype=float)
        covars = np.asarray(covars, dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array.")
        if means.ndim != 2:
            raise ValueError("means must be a 2D array (n_components, n_features).")
        if covars.ndim not in (2, 3):
            raise ValueError(
                "covars must be (n_components, n_features) for diag or "
                "(n_components, n_features, n_features) for full."
            )
        if means.shape[0] != weights.shape[0] or covars.shape[0] != weights.shape[0]:
            raise ValueError("weights, means, and covars must have same n_components.")

        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("weights must sum to 1.")
        if (weights < 0).any():
            raise ValueError("weights must be non-negative.")

        n_features = means.shape[1]
        if covars.ndim == 2 and covars.shape[1] != n_features:
            raise ValueError("diag covars must have shape (n_components, n_features).")
        if covars.ndim == 3:
            if covars.shape[1] != n_features or covars.shape[2] != n_features:
                raise ValueError("full covars must have shape (n_components, n_features, n_features).")

        self.weights = weights
        self.means = means
        self.covars = covars

    def _log_gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        d = mean.shape[0]
        diff = x - mean[None, :]
        if cov.ndim == 1:
            inv = 1.0 / cov
            quad = np.sum(diff * diff * inv[None, :], axis=1)
            logdet = np.sum(np.log(cov))
        else:
            chol = np.linalg.cholesky(cov)
            solve = np.linalg.solve(chol, diff.T)
            quad = np.sum(solve * solve, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        from latentflow.core import logsumexp

        x = np.atleast_2d(x)
        log_terms = []
        for w, m, c in zip(self.weights, self.means, self.covars):
            log_terms.append(np.log(w) + self._log_gaussian_pdf(x, m, c))
        return logsumexp(np.vstack(log_terms), axis=0)

    def pdf(self, x: np.ndarray) -> float:
        return float(np.exp(self.logpdf(np.asarray(x)))[0])

    def cdf(self, x: np.ndarray) -> float:
        # Component CDFs are weighted; for multivariate, rely on scipy's approximation.
        import scipy.stats

        x = np.asarray(x)
        total = 0.0
        for w, m, c in zip(self.weights, self.means, self.covars):
            if c.ndim == 1:
                total += w * scipy.stats.norm.cdf(x, loc=m, scale=np.sqrt(c))
            else:
                total += w * scipy.stats.multivariate_normal.cdf(x, loc=m, cov=c)
        return float(total)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        choices = rng.choice(len(self.weights), size=n, p=self.weights)
        samples = []
        for k in range(len(self.weights)):
            count = int(np.sum(choices == k))
            if count == 0:
                continue
            cov = self.covars[k]
            if cov.ndim == 1:
                samples.append(rng.normal(self.means[k], np.sqrt(cov), size=(count, self.means.shape[1])))
            else:
                samples.append(rng.multivariate_normal(self.means[k], cov, size=count))
        if not samples:
            return np.zeros((0, self.means.shape[1]))
        return np.vstack(samples)
