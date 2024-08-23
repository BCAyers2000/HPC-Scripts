"""
XY-Averaged Electrostatic Potential Script

This script reads a CUBE file containing electrostatic potential data,
performs xy-plane averaging, and plots the result along the z-axis.

Usage:
1. Ensure you have a CUBE file with electrostatic potential data (Environ verbose = 2).
2. Run this script to generate the xy-averaged electrostatic potential plot.

Dependencies:
- NumPy
- Matplotlib
- ASE (Atomic Simulation Environment)
"""

import numpy as np
from pathlib import Path
from ase.units import Rydberg
import matplotlib.pyplot as plt
import matplotlib.style as style
from ase.io.cube import read_cube
import matplotlib.font_manager as fm

font_dir = Path('/scratch/ba3g18/Installations/mambaforge/envs/Simulations/fonts')
for font_file in font_dir.glob('*.ttf'):
    fm.fontManager.addfont(str(font_file))
style.use('/iridisfs/home/ba3g18/.matplotlib_style')

cube_file = '/scratch/ba3g18/QE/Soft-spheres/4.01/Analysis/Cube_files/1.0_mol/pzc-10/velectrostatic.cube'
with open(cube_file, 'r') as cf:
    cube_dict = read_cube(cf, read_data=True, verbose=True)

atoms = cube_dict['atoms']
data = cube_dict['data']
nx, ny, nz = data.shape
cell = atoms.get_cell()
C = np.linalg.norm(cell[2])
dz = C / nz
z = np.arange(0, C, dz)
plan_avg = np.mean(data, axis=(0,1)) * Rydberg

plt.figure(figsize=(10, 6))
plt.plot(z, plan_avg)
plt.grid(True)
plt.xlabel('z [Å]')  
plt.ylabel(r'$\phi(z)$ [eV]')
plt.title('XY-Plane Averaged Electrostatic Potential')
plt.tight_layout()
plt.show()
