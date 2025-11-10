import sys
from pathlib import Path

# Add src directory to path for editable installs
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np

from latentflow.models.hmm import GMMARHMM
from latentflow.sampler import (
    make_random_gaussian_mixture_arhmm,
    sample_gmm_arhmm,
)
from latentflow.visualize import plot_hmm_series_with_states


if __name__ == "__main__":
    # Hyperparameters
    n_states = 3
    n_mixtures = 2
    n_features = 2
    order = 2
    T = 300
    seed = 321

    rng = np.random.default_rng(seed)

    # Sample random parameters
    arhmm_params = make_random_gaussian_mixture_arhmm(
        n_states=n_states,
        n_features=n_features,
        order=order,
        n_mixtures=n_mixtures,
        rng=rng,
    )
    print("True parameters:", arhmm_params)

    history = rng.normal(scale=0.5, size=(order, n_features)) if order > 0 else None

    # Sample trajectory
    true_states, obs = sample_gmm_arhmm(
        arhmm_params,
        T=T,
        rng=rng,
        history=history,
    )

    # Create Gaussian Mixture ARHMM model
    arhmm = GMMARHMM(
        n_components=n_states,
        n_mixtures=n_mixtures,
        order=order,
        random_state=seed,
    )
    arhmm.fit(obs, verbose=False)

    print("Learned parameters:", arhmm.params)

    # Predict states
    pred_states = arhmm.predict(obs)

    # Visualize
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    _, axes[0] = plot_hmm_series_with_states(
        list(range(T)),
        obs,
        true_states,
        covariate_names=[f"cov{i+1}" for i in range(n_features)],
        title="True Gaussian Mixture ARHMM Process",
        annotate_states=False,
        boundary_markers=True,
        ax=axes[0],
    )

    _, axes[1] = plot_hmm_series_with_states(
        list(range(T)),
        obs,
        pred_states,
        covariate_names=[f"cov{i+1}" for i in range(n_features)],
        title="Predicted Gaussian Mixture ARHMM Process",
        annotate_states=False,
        boundary_markers=True,
        ax=axes[1],
    )

    plt.tight_layout()
    plt.show()
