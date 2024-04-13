"""Methods for system initialization."""

from __future__ import annotations

import numpy as np

from updown.types import Lattice


def random(nrow: int, ncol: int) -> Lattice:
    """Initialize a lattice with random spins.

    Args:
        nrow: The number of lattice rows.
        ncol: The number of lattice columns.

    Returns:
        The initialized lattice.
    """
    thresh = 0.5
    return np.where(np.random.random((nrow, ncol)) > thresh, 1, -1)


def uniform(nrow: int, ncol: int, spin: int = 1) -> Lattice:
    """Initialize a lattice spins all in the same direction.

    Args:
        nrow: The number of lattice rows.
        ncol: The number of lattice columns.
        spin: The direction to initialize with. Defaults to 1.

    Returns:
        The initialized lattice.
    """
    return np.ones((nrow, ncol), dtype=np.int64) * spin
