"""
Materials Project Surface Analysis and Quantum ESPRESSO Input Generation

This script retrieves surface properties for a specific material from the Materials Project database,
analyzes the data, and generates an input file for Quantum ESPRESSO calculations.

Usage:
1. Replace 'API_KEY' with your actual Materials Project API key.
2. Run the script to retrieve data, analyze surfaces, and generate a Quantum ESPRESSO input file.

Dependencies:
- mp_api
- ASE (Atomic Simulation Environment)
"""

from mp_api.client import MPRester
from ase.io import read, write
from io import StringIO

api_key = "API_KEY"

with MPRester(api_key=api_key) as mpr:
    docs = mpr.materials.surface_properties.search(
        material_ids=["mp-135"],
        fields=["surfaces"]
    )
    
    doc = docs[0]
    print(f"Available surfaces for material ID: mp-135")
    print("Miller Index | Surface Energy (J/m^2) | Work Function (eV)")
    print("-" * 60)
    for surface in doc.surfaces:
        print(f"{surface.miller_index} | {surface.surface_energy:.4f} | {surface.work_function:.4f}")
    
    target_surface = next((s for s in doc.surfaces if list(s.miller_index) == [2, 2, 1]), None)
    print(f"\nSurface information for material ID: mp-135, Miller index: (1, 1, 0)")
    print(f"Surface Energy: {target_surface.surface_energy:.4f} J/m^2")
    print(f"Work Function: {target_surface.work_function:.4f} eV")
    
    cif_file = StringIO(target_surface.structure)
    atoms = read(cif_file, format='cif')
    
    print(f"\nNumber of atoms: {len(atoms)}")
    print(f"Chemical formula: {atoms.get_chemical_formula()}")
    print(f"Cell dimensions: {atoms.cell.lengths()[0]:.4f} x {atoms.cell.lengths()[1]:.4f} x {atoms.cell.lengths()[2]:.4f} Å")
    print(f"Cell angles: {atoms.cell.angles()[0]:.2f}° x {atoms.cell.angles()[1]:.2f}° x {atoms.cell.angles()[2]:.2f}°")
    print(f"Volume: {atoms.get_volume():.2f} Å³")
    
    print("\nComposition:")
    for symbol in set(atoms.get_chemical_symbols()):
        count = atoms.get_chemical_symbols().count(symbol)
        print(f"  {symbol}: {count}")
    
    atoms.center(vacuum=25, axis=2)
    write("test.in", atoms, format="espresso-in", pseudopotentials={'Li': 'li_pbe_v1.4.uspp.F.UPF'})
