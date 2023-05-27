"""Script for running an Ising model MC simulation."""
import argparse
import time
from pathlib import Path

import numpy as np

from updown.initialize import random
from updown.mc import run, run_temp_sequence
from updown.visualize import animate_run


def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments to control the script behavoir.

    Returns:
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        help="The width (in number of spins) of the lattice to run.",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        help="The height (in number of spins) of the lattice to run.",
    )
    parser.add_argument(
        "-N", "--ncycles", type=int, help="The number of cycles to run per temperature."
    )
    parser.add_argument(
        "-O",
        "--output",
        type=Path,
        default=Path("run_animation.gif"),
        help="The output to write to. Defaults to run_animation.gif",
    )
    parser.add_argument(
        "--start-temp",
        type=float,
        default=1.0,
        help="The starting temperature. Defaults to 1.0.",
    )
    parser.add_argument(
        "--end-temp",
        type=float,
        default=1.0,
        help=(
            "The ending temperature. Defaults to 1.0. If different from the starting"
            " temperature a simulation is run for a linear sequence of temperatures"
            " starting at --start-temp and ending at --end-temp. The number of"
            " temperatures run is controlled by the --ntemps argument."
        ),
    )
    parser.add_argument(
        "--ntemps",
        type=int,
        default=20,
        help=(
            "The number of temperatures to run. Only accessed if --start-temp does not"
            " equal --end-temp. Defaults to 20."
        ),
    )
    return parser


def main() -> None:
    """Run an Ising model MC simulation according to command line arguments."""
    args = parse_args().parse_args()
    start_time = time.time()
    lattice = random(args.height, args.width)
    if args.start_temp != args.end_temp:
        temps = np.linspace(args.start_temp, args.end_temp, args.ntemps)
        frames = run_temp_sequence(lattice, temps, args.ncycles)
    else:
        _, frames = run(lattice, args.ncycles, temp=args.start_temp)
    animate_run(frames, file=args.output, size=(args.height, args.width))
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
