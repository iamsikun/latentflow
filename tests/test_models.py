from __future__ import annotations

import numpy as np

from latentflow import (
    GaussianARHMM,
    GaussianHMM,
    GMMARHMM,
    GMMHMM,
    make_random_gaussian_arhmm,
    make_random_gaussian_hmm,
    make_random_gaussian_mixture_arhmm,
    make_random_gaussian_mixture_hmm,
    sample_gaussian_arhmm,
    sample_gaussian_hmm,
    sample_gmm_arhmm,
    sample_gmm_hmm,
)
from latentflow.matching import match_states
from latentflow.variables import UnivariateGaussian
from latentflow.dists import symmetric_kl_div_gaussian


def _assert_probabilities(proba: np.ndarray) -> None:
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(proba >= 0.0)


def test_gaussian_hmm_fit_predict_roundtrip():
    rng = np.random.default_rng(0)
    params = make_random_gaussian_hmm(n_states=3, n_features=2, rng=rng)
    true_states, obs = sample_gaussian_hmm(params, T=60, rng=rng)

    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=15,
        tol=1e-3,
        reg_covar=1e-5,
        init="random",
        random_state=0,
    )
    model.fit(obs)

    preds = model.predict(obs)
    assert preds.shape == true_states.shape

    proba = model.predict_proba(obs)
    assert proba.shape == (len(obs), 3)
    _assert_probabilities(proba)

    score = model.score(obs)
    assert np.isfinite(score)


def test_gaussian_arhmm_handles_autoregressive_design():
    rng = np.random.default_rng(1)
    params = make_random_gaussian_arhmm(n_states=2, n_features=2, order=1, rng=rng)
    _, obs = sample_gaussian_arhmm(params, T=70, rng=rng)

    model = GaussianARHMM(
        n_components=2,
        order=1,
        covariance_type="full",
        n_iter=12,
        tol=1e-3,
        init="random",
        random_state=1,
    )
    model.fit(obs)

    proba = model.predict_proba(obs)
    assert proba.shape == (len(obs), 2)
    _assert_probabilities(proba)

    sampled_states, sampled_obs = model.sample(T=10)
    assert sampled_states.shape == (10,)
    assert sampled_obs.shape[0] == 10


def test_gmm_hmm_supports_mixture_emissions():
    rng = np.random.default_rng(2)
    params = make_random_gaussian_mixture_hmm(
        n_states=2,
        n_features=2,
        n_mixtures=2,
        covariance_type="diag",
        rng=rng,
    )
    _, obs = sample_gmm_hmm(params, T=50, rng=rng)

    model = GMMHMM(
        n_components=2,
        n_mixtures=2,
        covariance_type="diag",
        n_iter=12,
        tol=1e-3,
        init="random",
        random_state=2,
    )
    model.fit(obs)

    preds = model.predict(obs)
    assert preds.shape == (len(obs),)

    proba = model.predict_proba(obs)
    assert proba.shape == (len(obs), 2)
    _assert_probabilities(proba)


def test_gmm_arhmm_runs_end_to_end():
    rng = np.random.default_rng(3)
    params = make_random_gaussian_mixture_arhmm(
        n_states=2,
        n_features=2,
        order=1,
        n_mixtures=2,
        covariance_type="diag",
        rng=rng,
    )
    _, obs = sample_gmm_arhmm(params, T=45, rng=rng)

    model = GMMARHMM(
        n_components=2,
        n_mixtures=2,
        order=1,
        covariance_type="diag",
        n_iter=10,
        tol=1e-3,
        init="random",
        random_state=3,
    )
    model.fit(obs)

    proba = model.predict_proba(obs)
    assert proba.shape == (len(obs), 2)
    _assert_probabilities(proba)
