# updown

Python for running 1 or 2D Ising models and visualizing the results. The code
was designed to be simple and fast, allowing for the simulation of relatively
large lattices.

<p align="center">
<img src="./run_animation.gif" width="500" height="500"/>
</p>

An example run animated. This run corresponds to a ferromagnetic 2000 x 2000
lattice that was randomly initialized. The temperature was decreased linearly
from 2 to 0.5 in 20 stages (20 cycles each). The included `run.py` script was
used to create the animation, thus:

```
python run.py -W 2000 -H 2000 -N 20 --start-temp 2.0 --end-temp 0.5
```
