"""
Bader Charge Analysis Script

This script extracts Bader charges from an ACF.dat file and assigns them to atoms.
It reads atomic structures from Quantum ESPRESSO input files and visualizes the results.

Usage:
1. Run Bader analysis to generate ACF.dat:
   ./bader water.cube
   or
   ./bader -p all_atom water.cube (for Bader volumes of each atom)

2. Run this script to process the results.

Dependencies:
- NumPy
- ASE (Atomic Simulation Environment)
"""

import numpy as np
from ase.io import read
from ase.visualize import view

def read_acf_charges(filename):
    charges = []
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            if line.strip() and not line.startswith((' -', 'VACUUM', 'NUMBER')):
                parts = line.split()
                if len(parts) >= 5:
                    charges.append(float(parts[4]))
    return np.array(charges)

def calculate_bader_charges(atoms, acf_charges):
    zval = {'O': 6, 'H': 1}
    return np.array([zval[atom.symbol] - charge for atom, charge in zip(atoms, acf_charges)])

atoms = read('/home/ba3g18/Documents/Git-Repos/PhD/Working Directory/DOS/water.in', format='espresso-in')
acf_charges = read_acf_charges('/home/ba3g18/Documents/Git-Repos/PhD/Working Directory/DOS/acf.dat')
bader_charges = calculate_bader_charges(atoms, acf_charges)
atoms.set_initial_charges(bader_charges)

print("Atom  ACF Charge  Bader Charge")
print("----  ----------  ------------")
for atom, acf, bader in zip(atoms, acf_charges, bader_charges):
    print(f"{atom.symbol:4s}  {acf:10.6f}  {bader:12.6f}")
view(atoms)
