from __future__ import annotations

"""
Result analysis utilities for LatentFlow.

This module makes it easy to persist evaluation results from tests or
experiments and compute summary statistics over estimated parameters. It ships
with a handful of common metrics (RMSE, MAE, MAPE) and exposes a registry so
callers can register additional metrics without having to change the core
package. Results are stored as JSON on disk to keep them portable and easy to
inspect.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import json

import numpy as np

MetricFunc = Callable[[np.ndarray, np.ndarray], float]


def _as_array(values: Sequence[float] | Mapping[str, float], *, expected_keys: Optional[List[str]] = None) -> tuple[np.ndarray, Optional[List[str]]]:
    """Convert parameter containers into an array while preserving ordering.

    If a mapping is provided, keys are sorted alphabetically to ensure a stable
    ordering. When ``expected_keys`` is provided we will reorder to match the
    expected keys and raise if any key is missing.
    """

    if isinstance(values, Mapping):
        if expected_keys is None:
            keys = sorted(values)
        else:
            missing = set(expected_keys) - set(values)
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"Estimated values are missing keys: {missing_list}")
            keys = list(expected_keys)
        arr = np.asarray([values[k] for k in keys], dtype=float)
        return arr, keys

    arr = np.asarray(values, dtype=float)
    return arr, None


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_pred - y_true) / denom)) * 100.0)


class MetricRegistry:
    """Registry of metric functions.

    The registry ships with a few defaults and allows users to register their
    own metrics. Each metric function should accept two numpy arrays (true
    values, predicted values) and return a float.
    """

    def __init__(self) -> None:
        self._metrics: MutableMapping[str, MetricFunc] = {
            "rmse": _rmse,
            "mae": _mae,
            "mape": _mape,
        }

    def register(self, name: str, func: MetricFunc, *, overwrite: bool = False) -> None:
        if not overwrite and name in self._metrics:
            raise ValueError(f"Metric '{name}' is already registered. Use overwrite=True to replace it.")
        self._metrics[name] = func

    def compute(self, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' is not registered.")
        return self._metrics[name](y_true, y_pred)

    def compute_many(self, names: Iterable[str], y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {name: self.compute(name, y_true, y_pred) for name in names}

    @property
    def metric_names(self) -> List[str]:
        return list(self._metrics)


class ResultAnalyzer:
    """Compute metrics over estimated parameters.

    Examples
    --------
    >>> analyzer = ResultAnalyzer()
    >>> metrics = analyzer.evaluate_parameters([1.0, 2.0], [1.1, 1.9])
    >>> sorted(metrics)
    ['mae', 'mape', 'rmse']
    """

    def __init__(self, registry: Optional[MetricRegistry] = None) -> None:
        self.registry = registry or MetricRegistry()

    def evaluate_parameters(
        self,
        true_params: Sequence[float] | Mapping[str, float],
        estimated_params: Sequence[float] | Mapping[str, float],
        *,
        metrics: Optional[Iterable[str]] = None,
    ) -> Dict[str, float]:
        y_true, keys = _as_array(true_params)
        y_pred, _ = _as_array(estimated_params, expected_keys=keys)

        if y_true.shape != y_pred.shape:
            raise ValueError("true_params and estimated_params must have the same shape after alignment.")

        metric_names = list(metrics) if metrics is not None else self.registry.metric_names
        return self.registry.compute_many(metric_names, y_true, y_pred)

    def register_metric(self, name: str, func: MetricFunc, *, overwrite: bool = False) -> None:
        """Convenience wrapper around ``MetricRegistry.register``."""

        self.registry.register(name, func, overwrite=overwrite)


@dataclass
class ResultRecord:
    test_name: str
    true_params: Sequence[float] | Mapping[str, float]
    estimated_params: Sequence[float] | Mapping[str, float]
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def _make_serializable(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (list, tuple)):
                return list(obj)
            if isinstance(obj, Mapping):
                return {k: _make_serializable(v) for k, v in obj.items()}
            return obj

        return {
            "test_name": self.test_name,
            "true_params": _make_serializable(self.true_params),
            "estimated_params": _make_serializable(self.estimated_params),
            "metrics": _make_serializable(self.metrics),
            "metadata": _make_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResultRecord":
        return cls(
            test_name=payload["test_name"],
            true_params=payload["true_params"],
            estimated_params=payload["estimated_params"],
            metrics=payload.get("metrics", {}),
            metadata=payload.get("metadata", {}),
        )


class ResultStore:
    """Persist and reload ``ResultRecord`` objects.

    Each store writes to a single JSON file containing a list of result
    dictionaries. The file is created if it does not exist yet.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: ResultRecord) -> None:
        records = self._load_json()
        records.append(record.to_dict())
        self._write_json(records)

    def load(self) -> List[ResultRecord]:
        return [ResultRecord.from_dict(entry) for entry in self._load_json()]

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def _load_json(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        text = self.path.read_text()
        if not text.strip():
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse result store at {self.path}") from exc
        if not isinstance(payload, list):
            raise ValueError("Result store must contain a list of result dictionaries.")
        return payload

    def _write_json(self, payload: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))

