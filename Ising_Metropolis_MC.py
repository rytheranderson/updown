from __future__ import print_function
import numpy as np
from numba import jit
import time
from PIL import Image

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation

#------------------------------------------------------------------------------# 
# Color maps
#------------------------------------------------------------------------------#

cmap1 = colors.ListedColormap(('#8e82fe','#580f41')) #periwinkle and plum

#------------------------------------------------------------------------------# 
# Functions
#------------------------------------------------------------------------------#

@jit
def initialize_lattice_random(nrow, ncol):

    """
        initialize a lattice with randomly oriented spins
    """

    return np.where(np.random.random((nrow,ncol))>0.5,1,-1)

@jit
def initialize_lattice_uniform(nrow, ncol, spin=1):

    """
        initialize a lattice with spins all in the same direction
    """

    return np.ones((nrow,ncol), dtype=np.int64) * spin

@jit
def system_energy(lattice, J, H):

    """
        J is the spin interaction parameter, J > 0 = ferromagnetic, J < 0 = antiferromagnetic
        H is an external magnetic field (constant)
    """

    nrow,ncol = lattice.shape
    E = 0.0

    for i in range(nrow):
        for j in range(ncol):

            S  = lattice[i,j]
            NS = lattice[(i+1)%nrow, j] + lattice[i,(j+1)%ncol] + lattice[(i-1)%nrow, j] + lattice[i,(j-1)%ncol]
            E += -1 * ((J * S * NS) + (H * S))

    return E/4

@jit
def system_magnetization(lattice):

    """
        calculate the system magnetization, just a rename of np.sum for physical interpretation
    """

    return np.sum(lattice)

@jit
def MC_cycle(lattice, J, H, T):

    """
        A single MC cycle (considering all lattice points)
        T is the temperature
    """

    T = float(T)
    naccept = 0 
    nrow,ncol = lattice.shape 
    E = system_energy(lattice, J, H) 
    M = system_magnetization(lattice)

    for i in range(nrow): 
        for j in range(ncol):

            S = lattice[i,j]
            NS = lattice[(i+1)%nrow, j] + lattice[i,(j+1)%ncol] + lattice[(i-1)%nrow, j] + lattice[i,(j-1)%ncol]
            dE = 2*J*S*NS + 2*H*S
            accept = np.random.random()

            if dE < 0.0 or accept < np.exp((-1.0 * dE)/T):
                naccept += 1
                S *= -1
                E += dE
                M += 2*S

            lattice[i,j] = S

    return lattice, E, M, naccept

@jit
def run(lattice, N_cycles, J=1, H=0, T=1.0, standard_output=False):

    """
        The summary function, which runs an MC simulation of N_cycles
    """

    nrow,ncol = lattice.shape

    lattice_evolve = [np.zeros((nrow,ncol)) for i in range(N_cycles)]
    energy_vs_step = []
    magnet_vs_step = []

    for cyc in range(N_cycles):

        if standard_output:
            print('cycle', cyc + 1, 'out of', N_cycles)

        lattice, E, M, naccept = MC_cycle(lattice, J, H, T)
        lattice_evolve[cyc] += lattice
        energy_vs_step.append(E)
        magnet_vs_step.append(M)

    return lattice, energy_vs_step, magnet_vs_step, lattice_evolve

def cooling(lattice, T_range, N_cycles, J=1, H=0):

    """
        a series of runs at decreasing temperatures
    """

    summary = []
    frames = []
    FL = lattice

    for T in T_range:

        print(f'Temperature = {np.round(T,3)}')
        FL, EvS, MvS, LvS = run(lattice, N_cycles, J=J, H=H, T=T) # make sure the lattice is the same for next T
        summary.append([J, EvS, MvS])
        frames.extend(LvS)

    return summary, frames

def animate_run(LvS, size=(5,5), fps=15, bitrate=1800, filename='run', ticks='off', dpi=100, cmap=cmap1):

    """
        function used to produce run animations, takes awhile, suitable for small/short runs,
        has the advantage of producing clear images of any size
    """

    writer = animation.PillowWriter(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
    FIG, ax = plt.subplots(figsize=size)
    ims = []
    plt.axis(ticks)
    
    for L in LvS:
        
        L = np.where(L==1.0, 0, 255)
        im = ax.imshow(L, origin='lower', cmap=cmap)
        ims.append([im])
        
    ani = animation.ArtistAnimation(FIG, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename + '.gif', dpi=dpi, writer=writer)

def fast_animate_run(LvS, resize=True, size=(200,200), fastest=True, filename='run', ticks='off', cmap=cmap1):

    """
        use this to animate very large/long runs
        fastest = True writes a black and white image, but is much faster than using a colormap
        recommended for systems larger than 200x200 lattice points
        size is in pixels
    """

    if fastest:
        gif = [Image.fromarray(np.uint8(L)).convert('RGB') for L in LvS]
    else:
        gif = [Image.fromarray(np.uint8(cmap(L)*255)) for L in LvS]
    
    if resize:
        gif = [img.resize(size) for img in gif]

    gif[0].save(filename + '.gif', save_all=True, optimize=False, append_images=gif[1:], loop=0)

#------------------------------------------------------------------------------# 
# Usage example
#------------------------------------------------------------------------------#

if __name__ == '__main__':

    start_time = time.time()
    L = initialize_lattice_random(1000, 1000)
    summary, frames = cooling(L, np.linspace(2,0.5,20), 20)

    fast_animate_run(frames, size=(500,500))
    print('--- %s seconds ---' % (time.time() - start_time))
    