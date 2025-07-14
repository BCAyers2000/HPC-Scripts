import os
import numpy as np
import psutil
from pathlib import Path
from ase.io import read, write
from ase.build import bulk
import matplotlib.pyplot as plt
from ase.calculators.vasp import Vasp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator

MILLER_INDEX = [1, 0, 0]
os.environ['ASE_VASP_COMMAND'] = "srun --mpi=pmix /iridisfs/home/ba3g18/Repos/VASP/vasp.6.3.2/bin/vasp_std"
os.environ['VASP_PP_PATH'] = "/home/ba3g18/Repos/Pseudopotentials/POTPAW_VASP"


def simple_vasp_parallel(calc_params):
    params = calc_params.copy()
    ncores = int(os.environ.get('SLURM_NTASKS', psutil.cpu_count(logical=False) or 1))
    
    for ncore in range(int(np.sqrt(ncores)), 0, -1):
        if ncores % ncore == 0:
            params["ncore"] = ncore
            break    
    return params

def get_base_vasp_params(calculation_type="bulk"):
    params = {
        "encut": 500.0,
        "sigma": 0.1,
        "ediff": 1.0e-7,
        "prec": "Accurate",
        "algo": "Fast",
        "lreal": False,
        "lasph": True,
        "lcharg": False,
        "lwave": False,
        "nelm": 100,
        "nelmin": 4,
        "ismear": 1,
        "kspacing": 0.1,
        "kpar":2,
    }
    
    if calculation_type == "bulk":
        params.update({
            "ibrion": 2,
            "isif": 2,
            "nsw": 200,
            "ediffg": -1.0e-6,
        })
    elif calculation_type == "slab":
        params.update({
            "ibrion": 2,
            "isif": 2,
            "nsw": 200,
            "ediffg": -1.0e-2,
            "lvhar": True,
            "lorbit": 11,
        })
    
    return params


def run_calculation(atoms, calc_params, directory):
    Path(directory).mkdir(exist_ok=True)    
    calc_params = simple_vasp_parallel(calc_params)
    
    calc = Vasp(
        xc="PBE",
        setups={'Li': '_sv'},
        directory=directory,
        **calc_params
    )
    
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    
    relaxed_atoms = read(f"{directory}/CONTCAR")
    
    return energy, relaxed_atoms


def prepare_structures(initial_atoms, min_slab_size):
    structure = AseAtomsAdaptor.get_structure(initial_atoms)
    slabgen = SlabGenerator(
        structure,
        miller_index=MILLER_INDEX,
        min_slab_size=min_slab_size,
        min_vacuum_size=20,
        lll_reduce=False,
        center_slab=True,
        primitive=True,
        max_normal_search=5,
        in_unit_planes=True,
    )
    slabs = slabgen.get_slabs()
    ouc = AseAtomsAdaptor.get_atoms(slabs[0].oriented_unit_cell)
    slab = AseAtomsAdaptor.get_atoms(slabs[0])
    return ouc, slab


def calculate_surface_energy(slab_energy, ouc_energy, slab, ouc):
    area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    
    surface_energy = (
        1 / 2 / area * (slab_energy - ((len(slab) / len(ouc)) * ouc_energy))
    )
    
    return surface_energy * 16.02


def main():
    initial_atoms = bulk("Li", a=3.44, cubic=True)
    
    ouc_dir = Path("OUC")
    ouc, _ = prepare_structures(initial_atoms, min_slab_size=3)
    
    ouc_params = get_base_vasp_params("bulk")
    
    print("Starting oriented unit cell calculation...")
    ouc_energy, relaxed_ouc = run_calculation(
        ouc, ouc_params, ouc_dir
    )
    print(f"OUC calculation complete. Energy: {ouc_energy} eV")
    
    slab_thicknesses = [i for i in range(3, 39+1, 3)]
    surface_energies = []
    
    for slab_size in slab_thicknesses:
        slab_dir = Path(f"Surface_slab_{slab_size}")
        print(f"\nCalculating for slab size: {slab_size}")
        
        try:
            _, slab = prepare_structures(initial_atoms, slab_size)
            slab_params = get_base_vasp_params("slab")            
            slab_energy, _ = run_calculation(
                slab, slab_params, slab_dir
            )
            
            surface_energy = calculate_surface_energy(
                slab_energy, ouc_energy, slab, ouc
            )
            
            print(f"Surface energy for slab size {slab_size}: {surface_energy:.6f} J/m²")
            surface_energies.append(surface_energy)
            
        except Exception as e:
            print(f"Error calculating for slab size {slab_size}: {str(e)}")
            surface_energies.append(np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.plot(slab_thicknesses, surface_energies, marker="o")
    plt.xlabel("Slab Thickness")
    plt.ylabel("Surface Energy (J/m²)")
    plt.title(f"Surface Energy vs Slab Thickness for Li {MILLER_INDEX} Surface")
    plt.grid(True)
    plt.savefig("surface_energy_vs_thickness.png")
    plt.close()
    
    print("\nSurface Energies:")
    for thickness, energy in zip(slab_thicknesses, surface_energies):
        print(f"Thickness {thickness}: {energy:.6f} J/m²")


if __name__ == "__main__":
    main()