from __future__ import annotations

from typing import Optional

import numpy as np

def _check_random_state(random_state: Optional[int | np.random.Generator]) -> np.random.Generator:
    """
    Check if the random state is a valid random state.
    """
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(int(random_state))
    if isinstance(random_state, np.random.Generator):
        return random_state
    raise TypeError("random_state must be None, int or np.random.Generator")