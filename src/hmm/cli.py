"""Command-line interface for the hmm package."""

from hmm.sampler import make_random_gaussian_hmm, sample_gaussian_hmm
from hmm.models.hmm import GaussianHMM
from hmm.visualize import plot_hmm_series_with_states


def main():
    """Main entry point for the hmm CLI."""
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


def sampler_main():
    """Entry point for the hmm-sampler CLI."""
    import numpy as np
    
    # Hyperparameters
    n_states = 3
    n_features = 2
    T = 200
    
    # Sample random parameters
    hmm_params = make_random_gaussian_hmm(n_states=n_states, n_features=n_features)
    print("Sampled HMM parameters:", hmm_params)
    
    # Sample trajectory
    true_states, obs = sample_gaussian_hmm(hmm_params, T=T)
    
    print(f"Sampled {T} observations with {n_states} hidden states")
    print(f"State sequence: {true_states[:20]}... (showing first 20)")
    print(f"Observations shape: {obs.shape}")


if __name__ == "__main__":
    main()

