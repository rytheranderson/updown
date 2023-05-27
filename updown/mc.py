"""Methods for Metropolis Monte Carlo simulations."""
from __future__ import annotations

from typing import Sequence

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
        The lattice after running a Metropolis Monte Carlo cycle.
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
) -> tuple[Lattice, list[Lattice]]:
    """Run a series of Monte Carlo cycles.

    Args:
        lattice: The starting lattice. Likely initialized using one of the
                 methods in the initialize module.
        ncycles: The number of Monte Carlo cycles to run.
        spin_inter: The spin interaction parameter to use. Defaults to 1.0.
        ext_field: The external magnetic field to apply. Defaults to 0.0.
        temp: The temperature. Defaults to 1.0.

    Returns:
        The last frame of the run and the lattice after each cycle.
    """
    nrow, ncol = lattice.shape
    frames = [np.zeros((nrow, ncol)) for i in range(ncycles)]
    for cyc in range(ncycles):
        lattice, _, _ = mc_cycle(lattice, spin_inter, ext_field, temp)
        frames[cyc] += lattice

    return lattice, frames


def run_temp_sequence(
    lattice: Lattice,
    temps: Sequence[float],
    ncycles: int,
    spin_inter: float = 1.0,
    ext_field: float = 0.0,
) -> list[Lattice]:
    """Perform a series of runs at different temperatures, stitching together.

    Args:
        lattice: The starting lattice. Likely initialized using one of the
                 methods in the initialize module.
        temps: The temperatures to run at.
        ncycles: The number of Monte Carlo cycles to run per temperature.
        spin_inter: The spin interaction parameter to use. Defaults to 1.0.
        ext_field: The external magnetic field to apply. Defaults to 0.0.

    Returns:
        The frames for each temperature, stitched together.
    """
    all_frames = []
    for temp in temps:
        print(f"temperature = {np.round(temp,3)}")
        _, frames = run(
            lattice, ncycles, spin_inter=spin_inter, ext_field=ext_field, temp=temp
        )
        all_frames.extend(frames)

    return all_frames
