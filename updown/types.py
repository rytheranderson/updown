"""Types used throughout updown."""
from typing import NewType

from nptyping import Int64, NDArray, Shape

Width = NewType("Width", int)
Height = NewType("Height", int)
Lattice = NDArray[Shape["Height, Width"], Int64]
