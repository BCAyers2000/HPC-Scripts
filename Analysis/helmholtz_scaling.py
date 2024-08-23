"""
Quantum ESPRESSO Helmholtz Free Energy Analysis

This script processes Quantum ESPRESSO output files, converts Grand free energy
to Helmholtz free energy, and plots Helmholtz free energy versus scale factor.

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

def get_muN_from_output(output_file):
    muN_pattern = r'pot\.stat\. contrib\. \(-muN\) =\s*([-+]?\d*\.?\d+)\s*Ry'
    with open(output_file, 'r') as f:
        for line in f:
            match = re.search(muN_pattern, line)
            if match:
                return float(match.group(1)) * 13.6056980659  # Convert Ry to eV

def process_espresso_files(main_dir):
    data = []
    for root, dirs, files in os.walk(main_dir):
        if 'espresso.pwo' in files:
            file_path = os.path.join(root, 'espresso.pwo')
            atoms = read(file_path)
            nelec = get_electrons_from_output(file_path)
            muN = get_muN_from_output(file_path)
            energy = atoms.get_potential_energy()
            
            energy -= muN  # Convert to Helmholtz free energy

            subdir = os.path.basename(root)
            scale_factor = float(subdir.split('_')[-1])  
            
            data.append((scale_factor, energy))
            
            print(f"Scale factor: {scale_factor:.2f}, Energy: {energy:.6f} eV, Electrons: {nelec:.2f}, -muN: {muN:.6f} eV")
    
    data.sort(key=lambda x: x[0])
    return np.array(data)

def plot_scale_vs_energy(data):
    scale_factors, energies = data[:, 0], data[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(scale_factors, energies, '-o')
    plt.xlabel('Scale Factor')
    plt.ylabel('Helmholtz Free Energy (eV)')
    plt.title('Helmholtz Free Energy vs Cell Size Scaling')
    plt.grid(True)
    # plt.savefig('helmholtz-energy-vs-scale.png')
    plt.show()

main_directory_path = '/scratch/ba3g18/QE/Grandcanonical/Graphene/Grandcanonical'
data = process_espresso_files(main_directory_path)
plot_scale_vs_energy(data)
