"""Types used throughout updown."""
from nptyping import Int64, NDArray, Shape

Lattice = NDArray[Shape["Height, Width"], Int64]
