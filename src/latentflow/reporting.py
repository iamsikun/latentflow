from __future__ import annotations

"""
HTML reporting utilities for LatentFlow visualizations and statistics.

This module builds interactive plots and summary tables in a single HTML file
so results can be shared or inspected later. It is intentionally lightweight
and pure-Python to avoid adding heavyweight notebook dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .visualize import _as_numpy_1d, _as_numpy_2d, _get_index_like


def _default_table(data: Mapping[str, Any]) -> go.Figure:
    headers = list(data)
    values = [[data[h]] if not isinstance(data[h], list) else data[h] for h in headers]
    return go.Figure(
        data=[
            go.Table(
                header=dict(values=headers, fill_color="#0f172a", font=dict(color="white"), align="left"),
                cells=dict(values=values, fill_color="#e2e8f0", align="left"),
            )
        ]
    )


def _timeseries_figure(
    x: Sequence[Any],
    Y: Sequence[Sequence[float]],
    series_names: Sequence[str],
    *,
    title: str | None = None,
    state_spans: Optional[List[Dict[str, Any]]] = None,
) -> go.Figure:
    fig = go.Figure()
    for col, name in zip(np.asarray(Y).T, series_names):
        fig.add_trace(go.Scatter(x=x, y=col, mode="lines", name=name))

    if state_spans:
        for span in state_spans:
            fig.add_vrect(**span)

    fig.update_layout(title=title, template="plotly_white", hovermode="x unified")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Value")
    return fig


def _compute_state_spans(x_values: Sequence[Any], states: Sequence[int], palette: Mapping[int, str]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if len(x_values) == 0:
        return segments
    start = 0
    current_state = states[0]
    for i in range(1, len(states)):
        if states[i] != current_state:
            segments.append(
                dict(
                    x0=x_values[start],
                    x1=x_values[i],
                    fillcolor=palette[current_state],
                    opacity=0.16,
                    line_width=0,
                    annotation_text=f"State {current_state}",
                )
            )
            start = i
            current_state = states[i]
    segments.append(
        dict(
            x0=x_values[start],
            x1=x_values[-1],
            fillcolor=palette[current_state],
            opacity=0.16,
            line_width=0,
            annotation_text=f"State {current_state}",
        )
    )
    return segments


def _infer_palette(states: Sequence[int]) -> Dict[int, str]:
    cmap = go.Figure().to_dict()["layout"].get("colorway", []) or [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]
    unique_states = sorted(set(states))
    return {s: cmap[i % len(cmap)] for i, s in enumerate(unique_states)}


@dataclass
class ReportSection:
    title: str
    figure: go.Figure
    description: Optional[str] = None

    def to_html(self) -> str:
        html = self.figure.to_html(include_plotlyjs=False, full_html=False)
        desc_block = f"<p>{self.description}</p>" if self.description else ""
        return f"<section><h2>{self.title}</h2>{desc_block}{html}</section>"


@dataclass
class HTMLReport:
    title: str
    sections: List[ReportSection] = field(default_factory=list)
    tables: List[Mapping[str, Any]] = field(default_factory=list)
    additional_html: List[str] = field(default_factory=list)

    def add_section(self, section: ReportSection) -> None:
        self.sections.append(section)

    def add_table(self, table: Mapping[str, Any], *, title: str | None = None) -> None:
        fig = _default_table(table)
        self.sections.append(ReportSection(title=title or "Summary", figure=fig))

    def to_html(self) -> str:
        body = []
        for section in self.sections:
            body.append(section.to_html())
        body.extend(self.additional_html)
        return "\n".join(
            [
                "<html>",
                "<head>",
                f"<title>{self.title}</title>",
                '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>',
                '<style>body{font-family:Inter,system-ui,sans-serif;margin:24px;} h1,h2{color:#0f172a;} section{margin-bottom:36px;} table{border-collapse:collapse;}</style>',
                "</head>",
                "<body>",
                f"<h1>{self.title}</h1>",
                "\n".join(body),
                "</body>",
                "</html>",
            ]
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_html())
        return path


def build_timeseries_report(
    *,
    x: Sequence[Any],
    Y: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    states: Sequence[int],
    covariate_names: Optional[Sequence[str]] = None,
    metrics_table: Optional[Mapping[str, Any]] = None,
    title: str = "LatentFlow Results",
    description: Optional[str] = None,
    include_state_annotations: bool = True,
) -> HTMLReport:
    x_values = _get_index_like(x)
    y_values = _as_numpy_2d(Y)
    if y_values.shape[0] != len(x_values):
        raise ValueError("Length of x must match rows in Y.")

    if hasattr(Y, "columns"):
        names = list(map(str, Y.columns))
    elif covariate_names is not None:
        names = list(map(str, covariate_names))
    else:
        names = [f"y{i}" for i in range(y_values.shape[1])]

    palette = _infer_palette(states)
    spans = _compute_state_spans(x_values, states, palette) if include_state_annotations else None

    fig = _timeseries_figure(x_values, y_values, names, title="Time series with hidden states", state_spans=spans)
    section_desc = description or "Interactive view of observed series with hidden-state shading."
    report = HTMLReport(title=title, sections=[ReportSection(title="Time series", figure=fig, description=section_desc)])

    if metrics_table:
        report.add_table(metrics_table, title="Metrics")

    return report


def make_timeseries_section(
    *,
    title: str,
    x: Sequence[Any],
    Y: Sequence[Sequence[float]] | np.ndarray | pd.DataFrame,
    states: Sequence[int],
    covariate_names: Optional[Sequence[str]] = None,
    description: Optional[str] = None,
    include_state_annotations: bool = True,
) -> ReportSection:
    """
    Convenience helper to build a ``ReportSection`` with a Plotly time-series figure
    and optional hidden-state shading.
    """
    x_values = _get_index_like(x)
    y_values = _as_numpy_2d(Y)
    if y_values.shape[0] != len(x_values):
        raise ValueError("Length of x must match rows in Y.")

    if hasattr(Y, "columns"):
        names = list(map(str, Y.columns))
    elif covariate_names is not None:
        names = list(map(str, covariate_names))
    else:
        names = [f"y{i}" for i in range(y_values.shape[1])]

    palette = _infer_palette(states)
    spans = _compute_state_spans(x_values, states, palette) if include_state_annotations else None
    fig = _timeseries_figure(x_values, y_values, names, title=title, state_spans=spans)
    section_desc = description or "Interactive time-series view."
    return ReportSection(title=title, figure=fig, description=section_desc)
