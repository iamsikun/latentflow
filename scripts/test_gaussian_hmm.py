import sys
from pathlib import Path

# Add src directory to path for editable installs
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from hmm.sampler import make_random_gaussian_hmm, sample_gaussian_hmm
from hmm.models.hmm import GaussianHMM
from hmm.visualize import plot_hmm_series_with_states, plot_faceted_hmm_series_with_states


if __name__ == "__main__":
    # Hyperparameters
    n_states = 3
    n_features = 2
    T = 200

    # Sample random parameters
    hmm_params = make_random_gaussian_hmm(n_states=n_states, n_features=n_features)
    print("True parameters:", hmm_params)

    # Sample trajectory
    true_states, obs = sample_gaussian_hmm(hmm_params, T=T)

    # Create HMM model
    hmm = GaussianHMM(n_components=n_states)
    hmm.fit(obs, verbose=False)

    print("Learned parameters:", hmm.params)

    # predict states 
    pred_states = hmm.predict(obs)

    # visualize
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    _, axes[0] = plot_hmm_series_with_states(
        list(range(T)), obs, true_states,
        covariate_names=['cov1', 'cov2'],
        title='True Gaussian HMM Process',
        annotate_states=False,
        boundary_markers=True,
        ax=axes[0],
    )

    _, axes[1] = plot_hmm_series_with_states(
        list(range(T)), obs, pred_states,
        covariate_names=['cov1', 'cov2'],
        title='Predicted Gaussian HMM Process',
        annotate_states=False,
        boundary_markers=True,
        ax=axes[1],
    )

    plt.tight_layout()
    plt.show()

