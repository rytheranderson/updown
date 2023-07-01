"""Methods for visualizing results."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from updown.types import Lattice


def animate_run(
    frames: Iterator[Lattice], file: Path, size: tuple[int, int] = (1000, 1000)
) -> None:
    """Animate a sequence of lattices.

    Args:
        frames: The sequence of lattices to animate.
        file: The file to write the output (.gif format) to.
        size: The size in pixels of the output .gif. Defaults to (200, 200).
    """
    gif = (
        Image.fromarray(np.uint8(frame)).resize(size).convert("RGB") for frame in frames
    )
    next(gif).save(file, save_all=True, optimize=True, append_images=gif, loop=0)
