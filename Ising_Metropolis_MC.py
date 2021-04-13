from __future__ import print_function
import numpy as np
from numba import jit
import time

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

	return np.where(np.random.random((nrow,ncol))>0.5,1,-1)

@jit
def initialize_lattice_uniform(nrow, ncol, spin=1):

	return np.ones((nrow,ncol), dtype=np.int64) * spin

@jit
def system_energy(lattice, J, H):

	nrow,ncol = lattice.shape
	E = 0.0

	for i in range(nrow):
		for j in range(ncol):

			S  = lattice[i,j]
			NS = lattice[(i+1)%nrow, j] + lattice[i,(j+1)%ncol] + lattice[(i-1)%nrow, j] + lattice[i,(j-1)%ncol]
			E += -1 * ( (J * S * NS) + (H * S) )

	return E/4

@jit
def system_magnetization(lattice):

	return np.sum(lattice)

@jit
def MC_cycle(lattice, J, H, T):

	T = float(T)
	naccept = 0 
	nrow,ncol = lattice.shape 
	E = system_energy(lattice, J, H) 
	M = system_magnetization(lattice)

	for i in range(nrow): 
		for j in range(ncol):

			S  = lattice[i,j]
			NS = lattice[(i+1)%nrow, j] + lattice[i,(j+1)%ncol] + lattice[(i-1)%nrow, j] + lattice[i,(j-1)%ncol]
			dE = 2*J*S*NS + 2*H*S
			accept = np.random.random()

			if dE < 0.0 or accept < np.exp( (-1.0 * dE) / T ):
				naccept += 1
				S *= -1
				E += dE
				M += 2*S

			lattice[i,j] = S

	return lattice, E, M, naccept

@jit
def run(lattice, J, H, T, N_cycles, standard_output=False):

	nrow,ncol = lattice.shape

	lattice_evolve = [np.zeros((nrow,ncol)) for i in range(N_cycles)]
	energy_vs_step = []
	magnet_vs_step = []

	for cyc in range(N_cycles):

		if standard_output:
			print('cycle', cyc + 1, 'out of', N_cycles)

		lattice,E,M,naccept = MC_cycle(lattice, J, H, T)
		lattice_evolve[cyc] += lattice
		energy_vs_step.append(E)
		magnet_vs_step.append(M)

	return lattice, energy_vs_step, magnet_vs_step, lattice_evolve

@jit
def cooling(lattice, T_range, N_cycles, J=1, H=0):

	summary = []
	for T in T_range:

		FL, EvS, MvS, LvS , N_converge = run(lattice, J, H, T, N_cycles)
		summary.append([J, EvS, MvS, LvS])

	return summary

def animate_run(LvS, size=(5,5), fps=15, bitrate=1800, filename='run', ticks='off', dpi=100, cmap=cmap1):

	writer = animation.PillowWriter(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
	FIG = plt.figure(figsize=size)
	ims = []
	plt.axis(ticks)
	
	for L in LvS:
		
		L = np.where(L==1.0, 0, 255)
		im = plt.imshow(L, origin='lower', cmap=cmap)
		ims.append([im])

	ani = animation.ArtistAnimation(FIG, ims, interval=50, blit=True, repeat_delay=1000)
	ani.save(filename + '.gif', dpi=dpi, writer=writer)

def animate_cooling(summary, size=(5,5), fps=15, bitrate=1800, filename='vary', ticks='off', dpi=100, cmap=cmap1):

	writer = animation.PillowWriter(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
	FIG = plt.figure(figsize=size)
	ims = []
	plt.axis(ticks)
	
	for line in summary:

		var, EvS, MvS, LvS = line

		for L in LvS:

			L = np.where(L==1.0, 0, 255)
			im = plt.imshow(L, origin='lower', cmap=cmap)
			ims.append([im])

	ani = animation.ArtistAnimation(FIG, ims, interval=50, blit=True, repeat_delay=1000)
	ani.save(filename + '.gif', dpi=dpi, writer=writer)

#------------------------------------------------------------------------------# 
# Usage example
#------------------------------------------------------------------------------#

start_time = time.time()
L = initialize_lattice_random(100, 100)
#L = initialize_lattice_uniform(200, 200)
FL, EvS, MvS, LvS = run(L, 0.8, 0, 1.0, 1000, standard_output=True)
animate_run(LvS)

### Plot of E vs step

fig = plt.figure(figsize=(4,3))
plt.xlabel('MC Cycles')
plt.ylabel('Energy')
plt.plot(range(len(EvS)), EvS)
plt.savefig('EvS.tiff', dpi=300)
plt.close(fig)

fig = plt.figure(figsize=(4,3))
plt.xlabel('MC Cycles')
plt.ylabel('Magnetization')
plt.plot(range(len(MvS)), MvS)
plt.savefig('MvS.tiff', dpi=300)
plt.close(fig)

print('Normal termination after')
print('--- %s seconds ---' % (time.time() - start_time))
