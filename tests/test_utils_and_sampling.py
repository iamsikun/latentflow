from __future__ import annotations

import numpy as np

from latentflow.sampler import (
    make_random_gaussian_hmm,
    sample_any_hmm,
    sample_gaussian_hmm,
)
from latentflow.utils import _check_random_state


def test_check_random_state_accepts_int_and_generator():
    rng_from_int = _check_random_state(42)
    assert isinstance(rng_from_int, np.random.Generator)
    value_int = rng_from_int.random()

    base_rng = np.random.default_rng(42)
    rng_from_rng = _check_random_state(base_rng)
    value_rng = rng_from_rng.random()

    assert np.isfinite(value_int)
    assert np.isfinite(value_rng)


def test_sample_any_hmm_dispatches_gaussian_hmm():
    rng = np.random.default_rng(5)
    params = make_random_gaussian_hmm(n_states=2, n_features=2, rng=rng)
    states, obs = sample_any_hmm(params, T=5, rng=rng)

    assert states.shape == (5,)
    assert obs.shape == (5, params.n_features)

    states_direct, obs_direct = sample_gaussian_hmm(params, T=5, rng=rng)
    assert states_direct.shape == states.shape
    assert obs_direct.shape == obs.shape
