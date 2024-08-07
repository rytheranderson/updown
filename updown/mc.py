"""Methods for Metropolis Monte Carlo simulations."""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from numba import jit

from updown.properties import energy, magnetization
from updown.types import Lattice


@jit(nopython=True)  # type: ignore[misc]
def mc_cycle(
    lattice: Lattice, spin_inter: float, ext_field: float, temp: float
) -> tuple[Lattice, float, float]:
    """Run a Metropolis Monte Carlo cycle for an input lattice.

    Args:
        lattice: The system lattice.
        spin_inter: The spin interaction parameter.
        ext_field: The external magnetic field.
        temp: The temperature.

    Returns:
        The lattice, energy, and magnetization after the cycle.
    """
    temp = float(temp)
    nrow, ncol = lattice.shape
    cE = energy(lattice, spin_inter, ext_field)
    cM = magnetization(lattice)

    for iy in range(nrow):
        for ix in range(ncol):
            spin = lattice[iy, ix]
            nbors = (
                lattice[(iy + 1) % nrow, ix]
                + lattice[iy, (ix + 1) % ncol]
                + lattice[(iy - 1) % nrow, ix]
                + lattice[iy, (ix - 1) % ncol]
            )
            dE = 2 * spin_inter * spin * nbors + 2 * ext_field * spin
            accept = np.random.random()
            if dE < 0.0 or accept < np.exp((-1.0 * dE) / temp):
                spin *= -1
                cE += dE
                cM += 2 * spin
            lattice[iy, ix] = spin

    return lattice, cE, cM


@jit(nopython=True)  # type: ignore[misc]
def run(
    lattice: Lattice,
    ncycles: int,
    spin_inter: float = 1.0,
    ext_field: float = 0.0,
    temp: float = 1.0,
) -> Iterator[Lattice]:
    """Run a series of Monte Carlo cycles.

    Args:
        lattice: The starting lattice. Likely initialized using one of the
                 methods in the initialize module.
        ncycles: The number of Monte Carlo cycles to run.
        spin_inter: The spin interaction parameter to use. Defaults to 1.0.
        ext_field: The external magnetic field to apply. Defaults to 0.0.
        temp: The temperature. Defaults to 1.0.

    Yields:
        The lattice after each cycle.
    """
    for _ in range(ncycles):
        lattice, _, _ = mc_cycle(lattice, spin_inter, ext_field, temp)
        yield lattice


def run_temp_sequence(
    lattice: Lattice,
    temps: Sequence[float],
    ncycles: int,
    spin_inter: float = 1.0,
    ext_field: float = 0.0,
) -> Iterator[Lattice]:
    """Perform a series of runs at different temperatures, stitching together.

    Args:
        lattice: The starting lattice. Likely initialized using one of the
                 methods in the initialize module.
        temps: The temperatures to run at.
        ncycles: The number of Monte Carlo cycles to run per temperature.
        spin_inter: The spin interaction parameter to use. Defaults to 1.0.
        ext_field: The external magnetic field to apply. Defaults to 0.0.

    Yields:
        The frames for each temperature, stitched together.
    """
    for temp in temps:
        yield from run(
            lattice, ncycles, spin_inter=spin_inter, ext_field=ext_field, temp=temp
        )
