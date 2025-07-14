import os
import psutil
import numpy as np
from tabulate import tabulate
from typing import Tuple, Optional

from ase.io import read
from ase.build import bulk
from ase.atoms import Atoms
from ase.calculators.vasp import Vasp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator


def generate_kpts(atoms, periodic_3d=False, kspacing=0.10):
    """
    Generate k-points grid based on cell dimensions and kspacing.
    
    Args:
        atoms: ASE Atoms object
        periodic_3d: Whether the structure is periodic in 3D
        kspacing: K-point spacing in Å^-1
        
    Returns:
        List of k-points [kx, ky, kz]
    """
    cell_lengths = atoms.cell.lengths()
    kpts = np.ceil(2 * np.pi / (cell_lengths * kspacing)).astype(int)
    return kpts.tolist() if periodic_3d else [kpts[0], kpts[1], 1]


def simple_vasp_parallel(calc_params):
    """
    optimise VASP parallel execution parameters based on available cores.
    Only determines NCORE, as KPAR is fixed at 2.

    Args:
        calc_params: Dictionary of calculation parameters

    Returns:
        Updated parameters dictionary
    """
    params = calc_params.copy()
    ncores = int(os.environ.get("SLURM_NTASKS", psutil.cpu_count(logical=False) or 1))

    for ncore in range(int(np.sqrt(ncores)), 0, -1):
        if ncores % ncore == 0:
            params["ncore"] = ncore
            break

    return params


def get_vasp_calculator(atoms, directory, use_solvation=False, efermi_ref=0, kspacing=0.05, periodic_3d=True):
    """
    Create a VASP calculator with the specified parameters.

    Args:
        atoms: ASE Atoms object for which to create the calculator
        directory: Directory for calculation
        use_solvation: Whether to use solvation model
        efermi_ref: Reference Fermi energy in eV
        kspacing: K-point spacing in Å^-1
        periodic_3d: Whether the structure is periodic in 3D

    Returns:
        VASP calculator
    """
    calc_params = {
        "algo": "Normal",
        "ediff": 1e-06,
        "ediffg": -0.01,
        "encut": 500.0,
        "ibrion": 2,
        "isif": 2,
        "ismear": 2,
        "ispin": 1,
        "isym": 0,
        "kpar": 2,
        "lasph": True,
        "lcharg": False,
        "lorbit": 11,
        "lreal": False,
        "lvhar": True,
        "lvtot": True,
        "lwave": False,
        "nelm": 500,
        "nelmin": 4,
        "nsw": 500,
        "prec": "Accurate",
        "sigma": 0.2,
        "symprec": 1e-08,
        "weimin": 0.0,
    }
    kpts = generate_kpts(atoms, periodic_3d=periodic_3d, kspacing=kspacing)
    
    if use_solvation:
        solvation_params = {
            "lsol": True,
            "isol": 2,
            "tau": 0.0,
            "nc_k": 0.015,
            "lsol_scf": True,
            "eb_k": 89.9,
            "p_mol": 0.939,
            "epsilon_inf": 2.01,
            "r_solv": 2.0, # averaged radii of EC 
            "r_ion": 2.0,
            "c_molar": 1.0,
            "lnldiel": True,
            "lnlion": True,
        }
        calc_params.update(solvation_params)

    # Add Fermi energy if using grand canonical formalism
    if efermi_ref != 0:
        calc_params["efermi_ref"] = efermi_ref

    calc_params.update(simple_vasp_parallel(calc_params))
    return Vasp(xc="PBE", kpts=kpts, **calc_params, directory=directory, setups="recommended")


def prepare_structures(
    initial_atoms: Atoms,
    miller_index: list[int],
    slab_size: int = 15,
    read_structures: bool = False,
    ouc_path: Optional[str] = None,
    slab_path: Optional[str] = None,
) -> Tuple[Atoms, Atoms]:
    """
    Prepare oriented unit cell and slab structures from initial bulk atoms.

    Args:
        initial_atoms: Bulk ASE Atoms object
        miller_index: Miller indices for the desired surface
        slab_size: Minimum size of the slab in Angstroms
        read_structures: Whether to read pre-calculated structures
        ouc_path: Path to oriented unit cell structure file
        slab_path: Path to slab structure file

    Returns:
        Tuple of (oriented_unit_cell, slab) ASE Atoms objects
    """
    if read_structures and ouc_path and slab_path:
        try:
            return read(ouc_path), read(slab_path)
        except Exception as e:
            print(f"Error reading structures: {e}")
            print("Generating new structures instead.")

    structure = AseAtomsAdaptor.get_structure(initial_atoms)
    slabgen = SlabGenerator(
        structure,
        miller_index=miller_index,
        min_slab_size=slab_size,
        min_vacuum_size=20,
        lll_reduce=False,
        center_slab=True,
        primitive=True,
        max_normal_search=5,
        in_unit_planes=True,
    )
    slabs = slabgen.get_slabs()
    ouc = AseAtomsAdaptor.get_atoms(slabs[0].oriented_unit_cell)
    slab = AseAtomsAdaptor.get_atoms(slabs[0].get_orthogonal_c_slab())
    slab.center(vacuum=20, axis=2)
    return ouc, slab


def calculate_surface_energy(
    miller_index: list[int],
    use_solvation: bool = False,
    read_structures: bool = False,
    ouc_path: Optional[str] = None,
    slab_path: Optional[str] = None,
    slab_size: int = 15,
    efermi_ref: float = 0,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    kspacing: float = 0.10,
) -> dict:
    """
    Calculate surface energy using ASE and VASP.

    Args:
        miller_index: Miller indices for the desired surface
        use_solvation: Whether to use solvation model
        read_structures: Whether to read pre-calculated structures
        ouc_path: Path to oriented unit cell structure file
        slab_path: Path to slab structure file
        slab_size: Minimum size of the slab in Angstroms
        efermi_ref: Reference Fermi energy in eV (if not 0, grand canonical formalism is used)
        supercell: Supercell dimensions (nx, ny, nz)
        kspacing: K-point spacing in Å^-1

    Returns:
        Dictionary with calculation results
    """
    # Set VASP environment variables
    os.environ["VASP_PP_PATH"] = (
        "/iridisfs/home/ba3g18/Repos/Pseudopotentials/POTPAW_VASP"
    )
    os.environ["ASE_VASP_COMMAND"] = (
        "srun --mpi=pmix /iridisfs/home/ba3g18/Repos/VASP/vasp.6.3.2/bin/vasp_std"
    )

    # Create initial bulk lithium
    initial_bulk = bulk("Li", "bcc", cubic=True, a=3.44)
    ouc, slab = prepare_structures(
        initial_atoms=initial_bulk,
        miller_index=miller_index,
        slab_size=slab_size,
        read_structures=read_structures,
        ouc_path=ouc_path,
        slab_path=slab_path,
    )
    if supercell != (1, 1, 1):
        slab = slab * supercell

    # OUC calculation - 3D periodic
    try:
        print("Starting OUC calculation...")
        ouc_calc = get_vasp_calculator(
            atoms=ouc,
            directory="ouc_calc", 
            use_solvation=False, 
            efermi_ref=0,
            kspacing=kspacing,
            periodic_3d=True
        )
        ouc.calc = ouc_calc
        ouc_energy = ouc.get_potential_energy()
        relaxed_ouc = ouc.copy()
        print(f"OUC calculation completed. Energy: {ouc_energy:.6f} eV")
        ouc_kpts = generate_kpts(ouc, periodic_3d=True, kspacing=kspacing)
        print(f"OUC k-points: {ouc_kpts}")
    except Exception as e:
        print(f"Error in OUC calculation: {e}")
        raise

    # Slab calculation - 2D periodic (kz=1)
    try:
        print("Starting slab calculation...")
        slab_calc = get_vasp_calculator(
            atoms=slab,
            directory="slab_calc", 
            use_solvation=use_solvation, 
            efermi_ref=efermi_ref,
            kspacing=kspacing,
            periodic_3d=False
        )
        slab.calc = slab_calc
        slab_energy = slab.get_potential_energy()
        relaxed_slab = slab.copy()
        print(f"Slab calculation completed. Energy: {slab_energy:.6f} eV")
        slab_kpts = generate_kpts(slab, periodic_3d=False, kspacing=kspacing)
        print(f"Slab k-points: {slab_kpts}")
    except Exception as e:
        print(f"Error in slab calculation: {e}")
        raise

    # Calculate surface energy
    area = np.linalg.norm(np.cross(relaxed_slab.cell[0], relaxed_slab.cell[1]))
    n_ouc_in_slab = len(relaxed_slab) / len(relaxed_ouc)
    surface_energy = 1 / 2 / area * (slab_energy - (n_ouc_in_slab * ouc_energy))
    surface_energy_jm2 = surface_energy * 16.021766208

    results = {
        "area": area,
        "ouc_kpts": ouc_kpts,
        "slab_kpts": slab_kpts,
        "kspacing": kspacing,
        "ouc_energy": ouc_energy,
        "slab_energy": slab_energy,
        "relaxed_ouc": relaxed_ouc,
        "relaxed_slab": relaxed_slab,
        "miller_index": miller_index,
        "use_solvation": use_solvation,
        "n_ouc_in_slab": n_ouc_in_slab,
        "surface_energy_eV_A2": surface_energy,
        "surface_energy_J_m2": surface_energy_jm2,

    }

    if efermi_ref != 0:
        results["use_grand_canonical"] = True
        results["efermi_ref"] = efermi_ref

    return results


if __name__ == "__main__":
    SLAB_SIZE = 12
    OUC_PATH = '/'
    SLAB_PATH = '/'
    KSPACING = 0.10
    EFERMI_REF = -1.404
    USE_SOLVATION = True
    SUPERCELL = (2, 2, 1)
    READ_STRUCTURES = False
    MILLER_INDEX = [1, 0, 0]

    try:
        results = calculate_surface_energy(
            ouc_path=OUC_PATH,
            slab_path=SLAB_PATH,
            slab_size=SLAB_SIZE,
            supercell=SUPERCELL,
            kspacing=KSPACING,
            efermi_ref=EFERMI_REF,
            miller_index=MILLER_INDEX,
            use_solvation=USE_SOLVATION,
            read_structures=READ_STRUCTURES,
        )

        table_data = [
            ["Miller Index", str(results["miller_index"])],
            ["Slab Size", f"{SLAB_SIZE} layers"],
            ["Solvation", "Yes" if results["use_solvation"] else "No"],
            ["K-spacing", f"{KSPACING} Å^-1"],
            ["OUC k-points", str(results["ouc_kpts"])],
            ["Slab k-points", str(results["slab_kpts"])],
            ["Supercell", f"{SUPERCELL}"],
            ["OUC Energy", f"{results['ouc_energy']:.6f} eV"],
            ["Slab Energy", f"{results['slab_energy']:.6f} eV"],
            ["Surface Area", f"{results['area']:.6f} Å²"],
            ["OUC Units in Slab", f"{results['n_ouc_in_slab']:.2f}"],
            ["Surface Energy", f"{results['surface_energy_eV_A2']:.8f} eV/Å²"],
            ["Surface Energy", f"{results['surface_energy_J_m2']:.8f} J/m²"],
        ]

        if EFERMI_REF != 0:
            gc_data = [
                ["Grand Canonical", "Yes"],
                ["Reference Fermi Energy", f"{EFERMI_REF:.3f} eV"],
            ]
            table_data.extend(gc_data)

        print("\nSurface Energy Calculation Results:")
        print(tabulate(table_data, tablefmt="grid"))

    except Exception as e:
        print(f"Error during calculation: {e}")
