"""
Quantum ESPRESSO Grand Free Energy Analysis

This script processes Quantum ESPRESSO output files, converts Helmholtz free energy
to Grand free energy, and plots Grand free energy versus scale factor.

Usage:
1. Set the 'main_directory_path' to the root directory containing your QE output files.
2. Run the script to process the files, perform energy conversions, and generate the plot.

Dependencies:
- ASE (Atomic Simulation Environment)
- NumPy
- Matplotlib
"""

import os
import re
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

def get_electrons_from_output(output_file):
    electrons_pattern = r'number of electrons\s*=\s*([-+]?\d*\.?\d+)'
    with open(output_file, 'r') as f:
        for line in f:
            match = re.search(electrons_pattern, line)
            if match:
                return float(match.group(1))

def process_espresso_files(main_dir):
    data = []
    for root, dirs, files in os.walk(main_dir):
        if 'espresso.pwo' in files:
            file_path = os.path.join(root, 'espresso.pwo')
            atoms = read(file_path)
            fermi_energy = atoms.calc.get_fermi_level()
            nelec = get_electrons_from_output(file_path)
            muN = fermi_energy * nelec
            energy = atoms.get_potential_energy()
            energy -= muN  # Convert to Grand free energy

            subdir = os.path.basename(root)
            scale_factor = float(subdir.split('_')[-1])  
            data.append((scale_factor, energy))
            print(f"Scale factor: {scale_factor:.2f}, Energy: {energy:.6f} eV, Electrons: {nelec:.2f}, Fermi Energy: {fermi_energy:.2f}")
    
    data.sort(key=lambda x: x[0])
    return np.array(data)

def plot_scale_vs_energy(data):
    scale_factors, energies = data[:, 0], data[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(scale_factors, energies, '-o')
    plt.xlabel('Scale Factor')
    plt.ylabel('Grand Free Energy (eV)')
    plt.title('Grand Free Energy vs Cell Size Scaling')
    plt.grid(True)
    # plt.savefig('grand-energy-vs-scale.png')
    plt.show()

main_directory_path = '/scratch/ba3g18/QE/Grandcanonical/[100]/Charge-density'
data = process_espresso_files(main_directory_path)
plot_scale_vs_energy(data)
