import argparse
import sys
import webbrowser
from dataclasses import fields, is_dataclass
from pathlib import Path

# Add src directory to path for editable installs
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import matplotlib.pyplot as plt

from latentflow.config import load_experiment_config
from latentflow.analysis import ResultAnalyzer
from latentflow.sampler import sample_any_hmm
from latentflow.models.hmm import GaussianHMM, GaussianARHMM, GMMHMM, GMMARHMM
from latentflow.params import GaussianHMMParams, GaussianARHMMParams, GMMHMMParams, GMMARHMMParams
from latentflow.reporting import HTMLReport, make_timeseries_section
from latentflow.visualize import plot_hmm_series_with_states


def get_model_class(params):
    if isinstance(params, GaussianHMMParams):
        return GaussianHMM
    elif isinstance(params, GMMHMMParams):
        return GMMHMM
    elif isinstance(params, GaussianARHMMParams):
        return GaussianARHMM
    elif isinstance(params, GMMARHMMParams):
        return GMMARHMM
    else:
        raise TypeError(f"Unsupported parameter type: {type(params)}")


def _flatten_params_for_metrics(param_obj) -> dict[str, float]:
    """
    Flatten a parameter dataclass into a mapping of label -> value.

    This preserves field ordering and uses C-order flattening for arrays so that
    true and estimated parameters align for metric computation.
    """
    if not is_dataclass(param_obj):
        raise TypeError("Parameter object must be a dataclass.")

    flat: dict[str, float] = {}
    for field in fields(param_obj):
        value = getattr(param_obj, field.name)
        arr = np.asarray(value, dtype=float).ravel()
        for idx, val in enumerate(arr):
            flat[f"{field.name}[{idx}]"] = float(val)
    return flat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified HMM Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--verbose", action="store_true", help="Print EM progress")
    args = parser.parse_args()

    # Load configuration
    params, run_config = load_experiment_config(args.config)
    print(f"Loaded {type(params).__name__}")
    print("Parameters:", params)

    # Run configuration
    T = run_config.get("T", 200)
    seed = run_config.get("seed", 42)
    rng = np.random.default_rng(seed)

    # Need history for ARHMM sampling
    history = None
    if hasattr(params, 'order') and params.order > 0:
        history = rng.normal(scale=0.5, size=(params.order, params.n_features))

    # Sample trajectory
    true_states, obs = sample_any_hmm(
        params,
        T=T,
        rng=rng,
        history=history,
    )

    # Instantiate model
    model_cls = get_model_class(params)
    
    # Prepare init kwargs
    init_kwargs = {
        "n_components": params.n_states,
        "random_state": seed,
    }
    if hasattr(params, 'n_mixtures'):
        init_kwargs["n_mixtures"] = params.n_mixtures
    if hasattr(params, 'order'):
        init_kwargs["order"] = params.order
        
    model = model_cls(**init_kwargs)
    
    # Fit model
    print(f"Fitting {model_cls.__name__}...")
    model.fit(obs, verbose=args.verbose)

    print("Learned parameters:", model.params)

    # Predict states
    pred_states = model.predict(obs)

    # Visualize
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    model_name_display = type(params).__name__.replace("Params", "")
    
    _, axes[0] = plot_hmm_series_with_states(
        list(range(T)),
        obs,
        true_states,
        covariate_names=[f"cov{i+1}" for i in range(params.n_features)],
        title=f"True {model_name_display} Process",
        annotate_states=False,
        boundary_markers=True,
        ax=axes[0],
    )

    _, axes[1] = plot_hmm_series_with_states(
        list(range(T)),
        obs,
        pred_states,
        covariate_names=[f"cov{i+1}" for i in range(params.n_features)],
        title=f"Predicted {model_name_display} Process",
        annotate_states=False,
        boundary_markers=True,
        ax=axes[1],
    )

    plt.tight_layout()
    if run_config.get("output_file"):
        print(f"Saving plot to {run_config['output_file']}")
        plt.savefig(run_config["output_file"])
    else:
        # Only show if explicit request or no output file?
        # Standard behavior: show() if interactive or no file specified.
        plt.show()

    # ------------------------------------------------------------------
    # Interactive HTML report with metrics
    # ------------------------------------------------------------------
    analyzer = ResultAnalyzer()
    metrics_table = {}

    if model.params is not None:
        try:
            true_flat = _flatten_params_for_metrics(params)
            est_flat = _flatten_params_for_metrics(model.params)
            metrics_table = analyzer.evaluate_parameters(true_flat, est_flat)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"Warning: could not compute parameter metrics: {exc}")

    # Include simple prediction accuracy over hidden states
    metrics_table["state_accuracy"] = float(np.mean(pred_states == true_states))
    if model.loglik is not None:
        metrics_table["log_likelihood"] = float(model.loglik)

    report = HTMLReport(title=f"{model_name_display} Experiment Results")
    cov_names = [f"cov{i+1}" for i in range(params.n_features)]

    report.add_section(
        make_timeseries_section(
            title=f"True {model_name_display} Process",
            x=list(range(T)),
            Y=obs,
            states=true_states,
            covariate_names=cov_names,
            description="Observed covariates with true hidden-state shading.",
        )
    )
    report.add_section(
        make_timeseries_section(
            title=f"Predicted {model_name_display} Process",
            x=list(range(T)),
            Y=obs,
            states=pred_states,
            covariate_names=cov_names,
            description="Observed covariates with predicted hidden-state shading.",
        )
    )

    if metrics_table:
        report.add_table(metrics_table, title="Parameter & Prediction Metrics")

    html_output = Path(run_config.get("html_output", Path(args.config).with_suffix(".html")))
    saved_path = report.save(html_output)
    print(f"Saved interactive report to {saved_path}")
    try:
        webbrowser.open(saved_path.resolve().as_uri())
    except Exception as exc:  # pragma: no cover - best-effort open
        print(f"Warning: could not open report automatically: {exc}")
