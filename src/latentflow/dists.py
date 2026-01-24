from __future__ import annotations

import numpy as np

from latentflow.variables import MixtureGaussian, UnivariateGaussian, MultivariateGaussian


def kl_div_gaussian(
    p: UnivariateGaussian | MultivariateGaussian,
    q: UnivariateGaussian | MultivariateGaussian,
) -> float:
    if isinstance(p, UnivariateGaussian) and isinstance(q, UnivariateGaussian):
        return (
            np.log(p.std / q.std)
            + (q.std**2 + (p.mean - q.mean) ** 2) / (2 * q.std**2)
            - 0.5
        )
    elif isinstance(p, MultivariateGaussian) and isinstance(q, MultivariateGaussian):
        return (
            np.log(np.linalg.det(p.cov) / np.linalg.det(q.cov))
            + np.trace(np.linalg.inv(p.cov) @ q.cov)
            - p.mean.shape[0]
            + np.sum((p.mean - q.mean) ** 2) / np.linalg.det(p.cov)
        )
    else:
        raise ValueError(f"Unsupported types: {type(p)} and {type(q)}")


def symmetric_kl_div_gaussian(
    p: UnivariateGaussian | MultivariateGaussian,
    q: UnivariateGaussian | MultivariateGaussian,
) -> float:
    return (kl_div_gaussian(p, q) + kl_div_gaussian(q, p)) / 2


def kl_div_mixture_gaussian(
    p: MixtureGaussian,
    q: MixtureGaussian,
    *,
    n_samples: int = 2048,
    rng: np.random.Generator | None = None,
) -> float:
    """Approximate KL(p || q) via Monte Carlo sampling from p."""
    if rng is None:
        rng = np.random.default_rng(0)
    samples = p.sample(n_samples, rng=rng)
    log_p = p.logpdf(samples)
    log_q = q.logpdf(samples)
    return float(np.mean(log_p - log_q))


def symmetric_kl_div_mixture_gaussian(
    p: MixtureGaussian,
    q: MixtureGaussian,
    *,
    n_samples: int = 2048,
    rng: np.random.Generator | None = None,
) -> float:
    return (kl_div_mixture_gaussian(p, q, n_samples=n_samples, rng=rng)
            + kl_div_mixture_gaussian(q, p, n_samples=n_samples, rng=rng)) / 2


def symmetric_kl_div(
    p: UnivariateGaussian | MultivariateGaussian | MixtureGaussian,
    q: UnivariateGaussian | MultivariateGaussian | MixtureGaussian,
    *,
    n_samples: int = 2048,
    rng: np.random.Generator | None = None,
) -> float:
    if isinstance(p, (UnivariateGaussian, MultivariateGaussian)) and isinstance(
        q, (UnivariateGaussian, MultivariateGaussian)
    ):
        return symmetric_kl_div_gaussian(p, q)
    if isinstance(p, MixtureGaussian) and isinstance(q, MixtureGaussian):
        return symmetric_kl_div_mixture_gaussian(p, q, n_samples=n_samples, rng=rng)
    raise ValueError(f"Unsupported types: {type(p)} and {type(q)}")
