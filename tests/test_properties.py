"""Tests for the properties calculated in the properties module."""
import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes

from updown.properties import energy


@settings(deadline=None)
@given(
    array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=1000),
    st.sampled_from([-1, 1]),
    st.floats(min_value=-1.0e6, max_value=1.0e6, allow_subnormal=False),
    st.floats(min_value=-1.0e6, max_value=1.0e6, allow_subnormal=False),
)
def test_energy_correct_for_lattices_with_aligned_spins(
    lattice_shape: tuple[int, int], direction: int, spin_inter: float, ext_field: float
) -> None:
    """Ensure the energy method calculates energies correctly for aligned spins.

    Args:
        lattice_shape: The shape of the lattice to test.
        direction: The spin direction to test.
        spin_inter: The spin interaction parameter to use.
        ext_field: An external magnetic field to apply.
    """
    assume(not math.isnan(spin_inter) and not math.isnan(ext_field))
    nrows, ncols = lattice_shape
    lattice = direction * np.ones(lattice_shape, dtype=np.int64)
    nspins = nrows * ncols
    expected_energy = -2 * spin_inter * nspins - direction * ext_field * nspins
    assert math.isclose(energy(lattice, spin_inter, ext_field), expected_energy)
