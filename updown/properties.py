"""Methods for calculating system properties."""
from __future__ import annotations

import numpy as np
from numba import jit

from updown.types import Lattice


@jit(nopython=True)  # type: ignore[misc]
def energy(lattice: Lattice, spin_inter: float, ext_field: float) -> float:
    """Calculate the total energy of a given lattice.

    Args:
        lattice: The lattice to calculate the total energy of.
        spin_inter: The spin interaction parameter.
        ext_field: The external magnetic field.

    Returns:
        The total energy of the input lattice.
    """
    nrow, ncol = lattice.shape
    tE = 0
    for iy in range(nrow):
        for ix in range(ncol):
            spin = lattice[iy, ix]
            nbors = (
                lattice[(iy + 1) % nrow, ix]
                + lattice[iy, (ix + 1) % ncol]
                + lattice[(iy - 1) % nrow, ix]
                + lattice[iy, (ix - 1) % ncol]
            )
            tE += -1 * ((spin_inter * spin * nbors) + (ext_field * spin))
    return tE / 4


@jit(nopython=True)  # type: ignore[misc]
def magnetization(lattice: Lattice) -> int:
    """Calculate the magnetization of a lattice.

    Args:
        lattice: The lattice to calculate the magnetization of.

    Returns:
        The magnetization the input lattice.
    """
    magnetization: int = np.sum(lattice)
    return magnetization
