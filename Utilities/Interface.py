"""
Li-LiF Interface Analysis Script

This script analyzes the interface between Li and LiF structures for different LiF orientations.
It finds the best matching interfaces with the lowest strain and visualizes the results.

Key features:
1. Iteratively searches for the best interface match with gradually relaxing constraints.
2. Analyzes multiple LiF orientations (100, 110, 111) against a fixed Li (100) orientation.
3. Generates and visualizes the best interface for each orientation.
4. Provides a summary table of LiF orientations and their corresponding lowest strain values.

Usage:
1. Ensure Li.cif and LiF.cif files are in the same directory as the script.
2. Run the script to perform the analysis and generate interface structures.

Dependencies:
- NumPy
- ASE (Atomic Simulation Environment)
- Pymatgen
- tabulate
"""

import numpy as np
from tabulate import tabulate
from ase.visualize import view
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder

def find_best_interface_match(li_structure, lif_structure, li_miller, lif_miller, max_iterations=10):
    overall_best_match, overall_min_strain = None, float('inf')
    for iteration in range(max_iterations):
        area_tol = length_tol = 0.01 * (1 + iteration)
        angle_tol = 0.001 * (1 + iteration)
        max_area = 100 * (1 + iteration)
        substrate_analyzer = SubstrateAnalyzer(
            film_max_miller=1, substrate_max_miller=1,
            max_area_ratio_tol=area_tol, max_area=max_area,
            max_length_tol=length_tol, max_angle_tol=angle_tol,
        )
        matches = substrate_analyzer.calculate(lif_structure, li_structure, film_millers=[lif_miller], substrate_millers=[li_miller])
        iteration_best_match = min(matches, key=lambda x: np.max(x.strain), default=None)
        if iteration_best_match:
            iteration_min_strain = np.max(iteration_best_match.strain)
            if iteration_min_strain < overall_min_strain:
                overall_min_strain = iteration_min_strain
                overall_best_match = iteration_best_match
    return overall_best_match

def process_best_match(best_match, li_structure, lif_structure):
    if not best_match:
        return
    builder = CoherentInterfaceBuilder(
        li_structure, lif_structure,
        best_match.film_miller, best_match.substrate_miller
    )
    available_terminations = builder.terminations
    if not available_terminations:
        return
    chosen_termination = available_terminations[0]
    for interface in builder.get_interfaces(
        termination=chosen_termination,
        gap=2.0, vacuum_over_film=10.0,
        film_thickness=2, substrate_thickness=2
    ):
        atoms = AseAtomsAdaptor.get_atoms(interface)
        view(atoms)
        filename = f"LiF_{best_match.film_miller}_Li_{best_match.substrate_miller}_interface.xyz"
        atoms.write(filename)
        break

def main():
    li_structure = Structure.from_file("Li.cif")
    lif_structure = Structure.from_file("LiF.cif")
    li_miller = (1, 0, 0)
    lif_miller_list = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    results = []
    for lif_miller in lif_miller_list:
        best_match = find_best_interface_match(li_structure, lif_structure, li_miller, lif_miller)
        if best_match:
            strain = np.max(best_match.strain)
            results.append([str(lif_miller), f"{strain:.6f}"])
            process_best_match(best_match, li_structure, lif_structure)
    print("\nSummary of Results:")
    headers = ["LiF Miller Index", "Lowest Strain"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
