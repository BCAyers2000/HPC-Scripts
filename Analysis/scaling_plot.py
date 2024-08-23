"""
Quantum ESPRESSO Energy vs Cell Size Scaling Analysis

This script processes Quantum ESPRESSO output files from multiple directories,
extracts energy data, and plots energy versus cell size scaling.

Usage:
1. Set the 'main_directory_path' to the root directory containing your QE output files.
2. Run the script to process the files and generate the plot.

Dependencies:
- ASE (Atomic Simulation Environment)
- NumPy
- Matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import style
from pathlib import Path
from ase.io import read

font_dir = Path('/scratch/ba3g18/Installations/mambaforge/envs/Simulations/fonts')
for font_file in font_dir.glob('*.ttf'):
    fm.fontManager.addfont(str(font_file))
style.use('/iridisfs/home/ba3g18/.matplotlib_style')

def process_espresso_files(main_dir):
    data = []
    for root, dirs, files in os.walk(main_dir):
        if 'espresso.pwo' in files:
            file_path = os.path.join(root, 'espresso.pwo')
            atoms = read(file_path)
            energy = atoms.get_potential_energy()

            subdir = os.path.basename(root)
            scale_factor = float(subdir.split('_')[-1])  
            data.append((scale_factor, energy))
            print(f"Scale factor: {scale_factor:.2f}, Energy: {energy:.6f} eV")
    
    data.sort(key=lambda x: x[0])
    return np.array(data)

def plot_scale_vs_energy(data):
    scale_factors, energies = data[:, 0], data[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(scale_factors, energies, '-o')
    plt.xlabel('Scale Factor')
    plt.ylabel('Energy (eV)')
    plt.title('Energy vs Cell Size Scaling (including -muN)')
    plt.grid(True)
    # plt.savefig('E-vs-s-grand.png')
    plt.show()

main_directory_path = '/scratch/ba3g18/QE/Grandcanonical/[100]/mono-2v'
data = process_espresso_files(main_directory_path)
plot_scale_vs_energy(data)
