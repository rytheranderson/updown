# A Fast Python Implementation of the Ising Model

Python 3 code for running 1 to 3D Ising models and visualizing the results. 
This code was designed to be simple and fast, allowing for the simulation of relatively large lattices.
Keep in mind that the runs are fast, but the animations can take awhile.

<p align="center">
<img src="./run_cooling.gif" width="500" height="500"/>
</p>

An example run animated. This run corresponds to a ferromagnetic 1000x1000 lattice that was randomly initialized.
The temperature was decreased linearly from 2 to 0.5 in 20 stages (20 cycles each).
Here is the code used to produce the image:

```
L = initialize_lattice_random(1000, 1000)
summary, frames = cooling(L, np.linspace(2,0.5,20), 20)
fast_animate_run(frames, size=(500,500))
```

this animation function uses PIL Image objects to build a .gif, which is much faster than
matplotlib animation, but has less functionality (I think).
