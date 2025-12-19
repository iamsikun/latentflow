"""
Configuration loader for LatentFlow experiments.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np

from latentflow.params import (
    GaussianHMMParams,
    GaussianARHMMParams,
    GMMHMMParams,
    GMMARHMMParams,
)
from latentflow.sampler import (
    make_random_gaussian_hmm,
    make_random_gaussian_arhmm,
    make_random_gaussian_mixture_hmm,
    make_random_gaussian_mixture_arhmm,
)

def _parse_numpy(data: Any) -> np.ndarray:
    """Recursively convert lists to numpy arrays."""
    return np.asarray(data)

def load_experiment_config(
    path: Union[str, Path]
) -> Tuple[Union[GaussianHMMParams, GaussianARHMMParams, GMMHMMParams, GMMARHMMParams], Dict[str, Any]]:
    """
    Load experiment configuration from a YAML file.

    Returns:
        params: The model parameters (either randomly generated or loaded from config).
        run_config: Dictionary containing run configuration (e.g. T, output settings).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Validate top-level keys
    if "model_type" not in config:
        raise ValueError("Config must specify 'model_type'.")
    
    model_type = config["model_type"]
    mode = config.get("mode", "manual") # 'manual' or 'random'
    run_config = config.get("run_config", {})
    
    # Ensure T is in run_config if not present (optional default)
    if "T" not in run_config:
        run_config["T"] = 100

    rng = np.random.default_rng(config.get("seed", None))

    if mode == "random":
        # Random generation mode
        gen_config = config.get("generation_config", {})
        
        if model_type == "GaussianHMM":
            params = make_random_gaussian_hmm(
                n_states=gen_config.get("n_states", 3),
                n_features=gen_config.get("n_features", 2),
                rng=rng,
            )
        elif model_type == "GaussianARHMM":
            params = make_random_gaussian_arhmm(
                n_states=gen_config.get("n_states", 3),
                n_features=gen_config.get("n_features", 2),
                order=gen_config.get("order", 1),
                rng=rng,
            )
        elif model_type == "GMMHMM":
            params = make_random_gaussian_mixture_hmm(
                n_states=gen_config.get("n_states", 3),
                n_features=gen_config.get("n_features", 2),
                n_mixtures=gen_config.get("n_mixtures", 2),
                covariance_type=gen_config.get("covariance_type", "full"),
                rng=rng,
            )
        elif model_type == "GMMARHMM":
            params = make_random_gaussian_mixture_arhmm(
                n_states=gen_config.get("n_states", 3),
                n_features=gen_config.get("n_features", 2),
                n_mixtures=gen_config.get("n_mixtures", 2),
                order=gen_config.get("order", 1),
                covariance_type=gen_config.get("covariance_type", "full"),
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    elif mode == "manual":
        # Manual specification mode
        p_config = config.get("params", {})
        
        if model_type == "GaussianHMM":
            params = GaussianHMMParams(
                start_probs=_parse_numpy(p_config["start_probs"]),
                trans_mat=_parse_numpy(p_config["trans_mat"]),
                means=_parse_numpy(p_config["means"]),
                covars=_parse_numpy(p_config["covars"]),
            )
        elif model_type == "GaussianARHMM":
            params = GaussianARHMMParams(
                start_probs=_parse_numpy(p_config["start_probs"]),
                trans_mat=_parse_numpy(p_config["trans_mat"]),
                coeffs=_parse_numpy(p_config["coeffs"]),
                covars=_parse_numpy(p_config["covars"]),
                order=p_config["order"],
            )
        elif model_type == "GMMHMM":
            params = GMMHMMParams(
                start_probs=_parse_numpy(p_config["start_probs"]),
                trans_mat=_parse_numpy(p_config["trans_mat"]),
                weights=_parse_numpy(p_config["weights"]),
                means=_parse_numpy(p_config["means"]),
                covars=_parse_numpy(p_config["covars"]),
            )
        elif model_type == "GMMARHMM":
            params = GMMARHMMParams(
                start_probs=_parse_numpy(p_config["start_probs"]),
                trans_mat=_parse_numpy(p_config["trans_mat"]),
                weights=_parse_numpy(p_config["weights"]),
                coeffs=_parse_numpy(p_config["coeffs"]),
                covars=_parse_numpy(p_config["covars"]),
                order=p_config["order"],
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'random' or 'manual'.")

    return params, run_config
