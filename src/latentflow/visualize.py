from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ArrayLike1D = Union[Sequence[float], np.ndarray]
ArrayLike2D = Union[Sequence[Sequence[float]], np.ndarray]

# ----------------------------- helpers -----------------------------

def _as_numpy_1d(x: Union[ArrayLike1D, pd.Series]) -> np.ndarray:
    if hasattr(x, 'to_numpy'):
        return x.to_numpy()
    return np.asarray(x)

def _as_numpy_2d(Y: Union[ArrayLike2D, pd.DataFrame]) -> np.ndarray:
    arr = None
    if hasattr(Y, 'to_numpy'):
        arr = Y.to_numpy()
    else:
        arr = np.asarray(Y)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr

def _get_index_like(x: Union[ArrayLike1D, pd.Index, pd.Series]) -> np.ndarray:
    """Return values for the x-axis. If pandas objects are provided,
    return the index (for Series/DataFrame) or the values (for Index).
    """
    if hasattr(x, 'index') and hasattr(x.index, 'to_numpy'):
        # pandas Series/DataFrame: use their index
        return x.index.to_numpy()
    if hasattr(x, 'to_numpy') and 'Index' in type(x).__name__:
        return x.to_numpy()
    # plain array-like of x-values
    return _as_numpy_1d(x)

def _find_segments(states: Sequence[int]) -> List[Tuple[int, int, int]]:
    """
    Find segments of constant states in the sequence.

    Parameters
    ----------
    states : Sequence[int]
        Sequence of state indices.

    Returns
    -------
    List[Tuple[int, int, int]]
        List of (start_idx, end_idx_exclusive, state) segments.
    """
    states_arr = np.asarray(states)
    if states_arr.size == 0:
        return []
    start_idx_list = [0]
    for i in range(1, len(states_arr)):
        if states_arr[i] != states_arr[i-1]:
            start_idx_list.append(i)
    segs = []
    for i, start_idx in enumerate(start_idx_list):
        end_idx = start_idx_list[i+1] if i+1 < len(start_idx_list) else len(states_arr)
        segs.append((start_idx, end_idx, int(states_arr[start_idx])))
    return segs

def _infer_state_colors(states: Sequence[int], state_colors: Optional[Mapping[int, str]] = None):
    """
    Infer colors for states based on their unique values.

    Parameters
    ----------
    states : Sequence[int]
        Sequence of state indices.
    state_colors : Optional[Mapping[int, str]]
        Optional mapping from state id -> color.

    Returns
    -------
    Mapping[int, str]
        Mapping from state id -> color.
    """
    unique_states = sorted(np.unique(states).tolist())
    if state_colors is not None:
        # Preserve provided mapping; fallback to colormap for missing keys.
        cmap = plt.get_cmap('tab20')
        auto_colors = {s: cmap(i / max(1, (len(unique_states)-1))) for i, s in enumerate(unique_states)}
        auto_colors.update(state_colors)
        return auto_colors
    else:
        cmap = plt.get_cmap('tab20')
        return {s: cmap(i / max(1, (len(unique_states)-1))) for i, s in enumerate(unique_states)}

# ----------------------------- core API -----------------------------

def shade_states(
    ax: plt.Axes,
    x_values: Union[ArrayLike1D, pd.Index, pd.Series],
    states: Sequence[int],
    *,
    state_labels: Optional[Mapping[int, str]] = None,
    state_colors: Optional[Mapping[int, str]] = None,
    alpha: float = 0.14,
    annotate: bool = False,
    label_ypos: float = 0.98,
    label_kwargs: Optional[dict] = None,
    boundary_markers: bool = False, 
    boundary_kwargs: Optional[dict] = None,
) -> None:
    """Shade the background by hidden states over `x_values`.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes to shade.
    x_values : array-like or pandas Index/Series
        X-axis values (same length as `states`), can be datetime-like.
    states : Sequence[int]
        Hidden state for each x.
    state_labels : Optional[Mapping[int, str]]
        Optional mapping from state id -> label string for annotations.
    state_colors : Optional[Mapping[int, str]]
        Optional mapping from state id -> color. Defaults to a tab20 palette.
    alpha : float
        Opacity of the shaded spans.
    annotate : bool
        If True, add state labels centered in each segment.
    label_ypos : float
        Vertical position for labels in axes-relative coordinates (0..1).
    label_kwargs : dict
        Extra kwargs for ax.text (e.g., fontsize=9, fontweight='bold').
    boundary_markers : bool
        If True, add markers at the boundaries of each segment.
    boundary_kwargs : dict
        Extra kwargs for ax.plot (e.g., marker='o', markersize=4, markerfacecolor='white', markeredgecolor='black').
    """
    x_values = _get_index_like(x_values)
    states = np.asarray(states)
    if len(x_values) != len(states):
        raise ValueError("x_values and states must have the same length.")

    palette = _infer_state_colors(states, state_colors)
    segments = _find_segments(states)

    if label_kwargs is None:
        label_kwargs = dict(ha='center', va='top')

    if boundary_kwargs is None:
        boundary_kwargs = dict(linestyle=':', linewidth=1.0, alpha=0.6)

    for start_idx, end_idx, state in segments:
        x0 = x_values[start_idx]
        x1 = x_values[end_idx] if end_idx < len(x_values) else x_values[-1]
        # Expand to right edge a bit by using next tick if available
        ax.axvspan(x0, x1, facecolor=palette[state], alpha=alpha, lw=0)

        if annotate:
            xm = x_values[(start_idx + end_idx)//2]
            lbl = state_labels[state] if (state_labels and state in state_labels) else f"State {state}"
            ax.text(xm, label_ypos, lbl, transform=ax.get_xaxis_transform(), **label_kwargs)

    # Draw boundary markers at each change (except at index 0)
    if boundary_markers and len(segments) > 1:
        for (start_idx, end_idx, state), (next_start_idx, next_end_idx, next_state) in zip(segments[:-1], segments[1:]):
            # boundary at start of next segment
            ax.axvline(x_values[next_start_idx], zorder=0.5, **boundary_kwargs)

def plot_hmm_series_with_states(
    x: Union[ArrayLike1D, pd.Index, pd.Series],
    Y: Union[ArrayLike2D, pd.DataFrame, pd.Series],
    states: Sequence[int],
    *,
    covariate_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    state_labels: Optional[Mapping[int, str]] = None,
    state_colors: Optional[Mapping[int, str]] = None,
    alpha: float = 0.14,
    annotate_states: bool = False,
    figsize: Tuple[float, float] = (12, 4),
    legend: bool = True,
    grid: bool = True,
    boundary_markers: bool = False,
    boundary_kwargs: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot observed covariates as line charts with hidden-state background.

    Parameters
    ----------
    x : array-like or pandas Index/Series
        X-axis values (e.g., time); can be datetime-like.
    Y : array-like 2D, pandas DataFrame/Series
        Observed covariates; if 1D, treated as a single series.
    states : Sequence[int]
        Hidden state per time step.
    covariate_names : list[str], optional
        Names for each column in Y. If Y is a DataFrame, its columns are used.
    title : str, optional
        Figure title.
    y_label : str, optional
        Y-axis label.
    state_labels : Mapping[int, str], optional
        Mapping of hidden state -> label.
    state_colors : Mapping[int, str], optional
        Mapping of hidden state -> color.
    alpha : float, default 0.14
        Opacity for shaded background spans.
    annotate_states : bool, default False
        If True, annotate state name/ID on each segment.
    figsize : tuple, default (12, 4)
        Figure size.
    legend : bool, default True
        Show legend for covariates.
    grid : bool, default True
        Show grid lines.
    ax : matplotlib Axes, optional
        If provided, plot on this axes; otherwise create a new figure/axes.
    boundary_markers : bool
        If True, add markers at the boundaries of each segment.
    boundary_kwargs : dict
        Extra kwargs for ax.plot (e.g., marker='o', markersize=4, markerfacecolor='white', markeredgecolor='black').
    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    x_values = _get_index_like(x)
    y_values = _as_numpy_2d(Y)
    states = np.asarray(states)
    if len(x_values) != len(states) or len(x_values) != y_values.shape[0]:
        raise ValueError("x, Y, and states must have compatible lengths.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # shade first so lines draw on top
    shade_states(
        ax, x_values, states, 
        state_labels=state_labels, 
        state_colors=state_colors, 
        alpha=alpha, 
        annotate=annotate_states, 
        boundary_markers=boundary_markers, 
        boundary_kwargs=boundary_kwargs,
    )

    # pick names
    if hasattr(Y, 'columns'):
        names = list(map(str, Y.columns))
    else:
        if covariate_names is not None:
            if len(covariate_names) != y_values.shape[1]:
                raise ValueError("covariate_names length must match number of columns in Y.")
            names = list(map(str, covariate_names))
        else:
            names = [f"y{i}" for i in range(y_values.shape[1])]

    # plot lines
    for j in range(y_values.shape[1]):
        ax.plot(x_values, y_values[:, j], label=names[j], linewidth=1.6)

    if title:
        ax.set_title(title)
    if y_label:
        ax.set_ylabel(y_label)
    ax.set_xlim(x_values[0], x_values[-1])
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    if legend and y_values.shape[1] > 1:
        ax.legend(frameon=False, ncols=min(3, y_values.shape[1]))

    return fig, ax

def plot_faceted_hmm_series_with_states(
    x: Union[ArrayLike1D, pd.Index, pd.Series],
    Y: Union[ArrayLike2D, pd.DataFrame, pd.Series],
    states: Sequence[int],
    *,
    covariate_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    state_labels: Optional[Mapping[int, str]] = None,
    state_colors: Optional[Mapping[int, str]] = None,
    alpha: float = 0.14,
    annotate_states: bool = False,
    figsize: Tuple[float, float] = (12, 6),
    sharey: bool = False,
    grid: bool = True,
    boundary_markers: bool = False,
    boundary_kwargs: Optional[dict] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Facet by covariate: one panel per series with consistent background shading.

    Parameters are similar to `plot_hmm_series_with_states`, except we create
    a vertical stack of subplots (one per covariate).
    """
    x_values = _get_index_like(x)
    y_values = _as_numpy_2d(Y)
    states = np.asarray(states)
    if len(x_values) != len(states) or len(x_values) != y_values.shape[0]:
        raise ValueError("x, Y, and states must have compatible lengths.")

    if hasattr(Y, 'columns'):
        names = list(map(str, Y.columns))
    else:
        if covariate_names is not None:
            if len(covariate_names) != y_values.shape[1]:
                raise ValueError("covariate_names length must match number of columns in Y.")
            names = list(map(str, covariate_names))
        else:
            names = [f"y{i}" for i in range(y_values.shape[1])]

    fig, axes = plt.subplots(y_values.shape[1], 1, sharex=True, sharey=sharey, figsize=figsize)
    if y_values.shape[1] == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        shade_states(
            ax, x_values, states, 
            state_labels=state_labels, 
            state_colors=state_colors, 
            alpha=alpha, 
            annotate=annotate_states, 
            boundary_markers=boundary_markers, 
            boundary_kwargs=boundary_kwargs,
        )
        ax.plot(x_values, y_values[:, j], linewidth=1.6)
        ax.set_ylabel(names[j])
        if grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    axes[-1].set_xlim(x_values[0], x_values[-1])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


if __name__ == '__main__':
    # Self-demo if you run: python hmm_viz.py
    n = 300
    t = pd.date_range('2025-01-01', periods=n, freq='h')

    # make a few segments for hidden states
    states = np.repeat([0, 1, 2, 1, 0], repeats=[60, 50, 80, 40, 70])[:n]

    rng = np.random.default_rng(7)
    y1 = np.sin(np.linspace(0, 15, n)) + (states==1)*0.5 + (states==2)*1.0 + 0.1*rng.standard_normal(n)
    y2 = np.cos(np.linspace(0, 9, n)) + (states==2)*0.3 - (states==0)*0.2 + 0.1*rng.standard_normal(n)
    Y = np.column_stack([y1, y2])

    labels = {0: 'Baseline', 1: 'Warm', 2: 'Hot'}

    fig, ax = plot_hmm_series_with_states(
        t, Y, states,
        covariate_names=['cov1', 'cov2'],
        title='Gaussian HMM covariates with hidden-state shading',
        state_labels=labels,
        annotate_states=True,
        boundary_markers=True,
    )

    fig, axes = plot_faceted_hmm_series_with_states(
        t, Y, states,
        covariate_names=['cov1', 'cov2'],
        title='Gaussian HMM covariates with hidden-state shading',
        state_labels=labels,
        annotate_states=True,
        boundary_markers=True,
    )
    plt.show()
