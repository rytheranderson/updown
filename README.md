# Ising_model

Python 3 code for running 1 to 3D Ising models and visualizing the results. 
This code was designed to be simple and fast, allowing for the simulations of relatively large lattices.
Keep in mind that the runs are fast, but the animation is not.

<p align="center">
<img src="./run.gif" width="800" height="600"/>
</p>

An example run animated. This run corresponds to a ferromagnetic 200x200 lattice that was randomly initialized.
Here is the code used to produce the image:

```
L = initialize_lattice_random(200, 200)
FL, EvS, MvS, LvS = run(L, 300, J=0.8, H=0, T=1.0, standard_output=True)
animate_run(LvS)
```
