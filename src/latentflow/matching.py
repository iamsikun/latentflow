from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from latentflow.variables import RandomVariable


def compute_cost_matrix(
    vars1: list[RandomVariable],
    vars2: list[RandomVariable],
    dist_fn: Callable[[RandomVariable, RandomVariable], float],
) -> np.ndarray:
    """Compute the cost matrix between two lists of variables.

    Args:
        vars1: List of variables.
        vars2: List of variables.
        dist_fn: Distance function.

    Returns:
        Cost matrix.
    """
    n1 = len(vars1)
    n2 = len(vars2)
    cost_arr = np.zeros((n1, n2), dtype=float)
    for i in range(n1):
        for j in range(n2):
            cost_arr[i, j] = dist_fn(vars1[i], vars2[j])
    return cost_arr


def match_states(
    vars1: list[RandomVariable],
    vars2: list[RandomVariable],
    dist_fn: Callable[[RandomVariable, RandomVariable], float],
) -> dict[str, Any]:
    """
    Match states between two lists of variables.

    Args:
        vars1: List of variables.
        vars2: List of variables.
        dist_fn: Distance function.

    Returns:
        Dictionary containing the assignment, cost per state, and total cost.
        The ``assignment`` is a length ``len(vars1)`` array where each entry
        is the matched index in ``vars2`` or ``-1`` if unmatched. ``row_ind``
        and ``col_ind`` follow scipy's `linear_sum_assignment` convention.
    """
    if not vars1 or not vars2:
        return {
            "assignment": np.full(len(vars1), -1, dtype=int),
            "cost_per_state": np.array([], dtype=float),
            "total_cost": 0.0,
            "row_ind": np.array([], dtype=int),
            "col_ind": np.array([], dtype=int),
        }

    cost_arr = compute_cost_matrix(vars1, vars2, dist_fn)
    row_ind, col_ind = linear_sum_assignment(cost_arr)

    # compute total cost of the assignment
    assign_cost_arr = cost_arr[row_ind, col_ind]
    total_cost = np.sum(assign_cost_arr)
    assignment = np.full(len(vars1), -1, dtype=int)
    assignment[row_ind] = col_ind

    return {
        "assignment": assignment,
        "cost_per_state": assign_cost_arr,
        "total_cost": total_cost,
        "row_ind": row_ind,
        "col_ind": col_ind,
    }
