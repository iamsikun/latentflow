from __future__ import annotations

"""
LatentFlow public API.

This package exposes the main estimators, sampling helpers, and configuration
loader so downstream projects can import them directly from ``latentflow``.
"""

from latentflow.config import load_experiment_config
from latentflow.models.hmm import GaussianARHMM, GaussianHMM, GMMARHMM, GMMHMM
from latentflow.analysis import ResultAnalyzer, ResultRecord, ResultStore, MetricRegistry
from latentflow.reporting import HTMLReport, ReportSection, build_timeseries_report, make_timeseries_section
from latentflow.sampler import (
    make_random_gaussian_arhmm,
    make_random_gaussian_hmm,
    make_random_gaussian_mixture_arhmm,
    make_random_gaussian_mixture_hmm,
    sample_any_hmm,
    sample_gaussian_arhmm,
    sample_gaussian_hmm,
    sample_gmm_arhmm,
    sample_gmm_hmm,
)

__all__ = [
    "GaussianHMM",
    "GaussianARHMM",
    "GMMHMM",
    "GMMARHMM",
    "ResultAnalyzer",
    "ResultRecord",
    "ResultStore",
    "MetricRegistry",
    "HTMLReport",
    "ReportSection",
    "build_timeseries_report",
    "make_timeseries_section",
    "sample_gaussian_hmm",
    "sample_gaussian_arhmm",
    "sample_gmm_hmm",
    "sample_gmm_arhmm",
    "sample_any_hmm",
    "make_random_gaussian_hmm",
    "make_random_gaussian_arhmm",
    "make_random_gaussian_mixture_hmm",
    "make_random_gaussian_mixture_arhmm",
    "load_experiment_config",
]
