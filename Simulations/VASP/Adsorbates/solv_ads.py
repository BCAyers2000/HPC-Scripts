"""
Li-F ADSORPTION ENERGY CALCULATOR WITH SOLVATION SUPPORT (VASP VERSION)

This script calculates adsorption energies of fluorine on lithium surfaces with support
for both vacuum and explicit solvation (using VASP's VASPsol++ module).

Parameters:
-----------
slab_file : str
    Path to the file containing the lithium slab structure (POSCAR, CIF, or other ASE-readable format).

site_type : str, default="ontop"
    Type of adsorption site. Options include:
    - "ontop": Adsorption directly above a single Li atom
    - "bridge": Adsorption between two Li atoms
    - "3fold", "hollow", "fcc", "hcp": Adsorption in a three-fold hollow site
    - "4fold": Adsorption in a four-fold hollow site

use_height : bool, default=False
    - If True: Use explicit height above the surface for distance scanning
    - If False: Use Li-F bond distance for scanning

environments : list, default=["vacuum", "solvent"]
    List of environments to run calculations in:
    - ["vacuum"]: Run only vacuum calculations (no solvation)
    - ["solvent"]: Run only solvated calculations (with VASPsol)
    - ["vacuum", "solvent"]: Run both (default)

Usage Examples:
--------------
1. Run bridge site with explicit height in vacuum only:
   main("/path/to/Li.cif", site_type="bridge", use_height=True, environments=["vacuum"])

2. Run ontop site with bond distance in solvent only:
   main("/path/to/Li.cif", site_type="ontop", use_height=False, environments=["solvent"])

3. Run both environments with default settings:
   main("/path/to/Li.cif")

Output:
-------
The script creates separate directories for each environment:
- Li-F_{site_type}_vacuum/
- Li-F_{site_type}_solvent/

Each directory contains:
- Energy curves and adsorption energy plots
- Detailed results in results_{environment}.txt
- Optimised structure in optimised_structure_{environment}.vasp
"""

import os
import psutil
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.atoms import Atoms
import matplotlib.pyplot as plt
from ase.calculators.vasp import Vasp
from acat.build import add_adsorbate
from acat.settings import CustomSurface
from typing import List, Tuple, Dict, Any, Literal
from acat.adsorption_sites import SlabAdsorptionSites, get_layers

# Set environment variables for VASP
os.environ["VASP_PP_PATH"] = "/home/ba3g18/Repos/Pseudopotentials/POTPAW_VASP"
os.environ["ASE_VASP_COMMAND"] = (
    "srun --mpi=pmix /iridisfs/home/ba3g18/Repos/VASP/vasp.6.3.2/bin/vasp_std"
)

# Type definitions
SystemType = Literal["slab", "molecule", "bulk"]
CalculationType = Literal["scf", "relax"]
EnvironmentType = Literal["vacuum", "solvent"]

# Base VASP calculation parameters
BASE_PARAMETERS = {
    "ENCUT": 500.0,
    "SIGMA": 0.10,
    "EDIFF": 1.0e-5,
    "EDIFFG": -5.0e-2,
    "ALGO": "Normal",
    "PREC": "Accurate",
    "IBRION": 2,
    "AMIN": 0.01,
    "ISIF": 2,
    "ISMEAR": 1,
    "LORBIT": 11,
    "NELM": 200,
    "NELMIN": 4,
    "NSW": 200,
    "LASPH": True,
    "LCHARG": False,
    "LWAVE": False,
    "LVTOT": True,
    "LVHAR": True,
    "LREAL": "Auto",
}

# VASPsol++ parameters for solvation
SOLVATION_PARAMETERS = {
    "LSOL": True,
    "ISOL": 2,
    "LSOL_SCF": True,
    "EB_K": 89.9,
    "EPSILON_INF": 1.0,
    "R_ION": 4.0,
    "C_MOLAR": 1.0,
    "LNLDIEL": True,
    "LNLION": True,
    "ZION": 1.0,
}


def optimise_vasp_parallelisation(
    calc_params: Dict[str, Any], kpts=None
) -> Dict[str, Any]:
    """
    Optimise VASP parallelisation settings based on available cores and k-points.
    """
    params = calc_params.copy()
    ncores = int(os.environ.get("SLURM_NTASKS", psutil.cpu_count(logical=False) or 1))
    for ncore in range(int(np.sqrt(ncores)), 0, -1):
        if ncores % ncore == 0:
            params["NCORE"] = ncore
            break
    if kpts and "KPAR" not in params:
        nkpts = np.prod(kpts)
        if nkpts >= 4:
            for kpar in range(2, min(nkpts, ncores // 2) + 1):
                if ncores % kpar == 0:
                    params["KPAR"] = kpar
                    break

    return params


def calculate_site_geometry(
    atoms: Any, site_positions: np.ndarray, site_type: str
) -> Tuple[float, int]:
    """Calculate geometric parameters for adsorption sites."""
    if site_type == "ontop":
        return 0.0, 1
    elif site_type in ["bridge", "longbridge", "shortbridge"]:
        metal_dist = np.linalg.norm(site_positions[0] - site_positions[1])
        return metal_dist / 2, 2
    elif site_type in ["3fold", "hollow", "fcc", "hcp"]:
        positions = site_positions[:3]
        center = np.mean(positions, axis=0)
        radii = [np.linalg.norm(pos - center) for pos in positions]
        return min(radii), 3
    elif site_type == "4fold":
        center = np.mean(site_positions, axis=0)
        radii = [np.linalg.norm(pos - center) for pos in site_positions]
        return max(radii), 4
    else:
        raise ValueError(f"Unknown site type: {site_type}")


def get_site_positions(atoms: Any, sas: Any) -> np.ndarray:
    """Get positions of nearest atoms to adsorption site."""
    temp_atoms = atoms.copy()
    add_adsorbate(
        temp_atoms, adsorbate="F", site="ontop", height=2.0, adsorption_sites=sas
    )
    f_idx = len(temp_atoms) - 1
    distances = temp_atoms.get_distances(f_idx, range(f_idx))
    site_indices = np.argsort(distances)[:4]
    return temp_atoms.positions[site_indices]


def get_bond_length_from_height(
    atoms: Any, sas: Any, height: float, site_type: str
) -> float:
    """Calculate bond length from height for given site type."""
    site_positions = get_site_positions(atoms, sas)
    radius, _ = calculate_site_geometry(atoms, site_positions, site_type)
    return np.sqrt(height**2 + radius**2)


def calculate_height_for_site(
    atoms: Any, sas: Any, desired_bond_length: float, site_type: str
) -> float:
    """Calculate height from desired bond length for given site type."""
    site_positions = get_site_positions(atoms, sas)
    radius, _ = calculate_site_geometry(atoms, site_positions, site_type)
    if site_type == "ontop":
        return desired_bond_length
    return np.sqrt(desired_bond_length**2 - radius**2)


def create_vasp_calculator(
    directory: Path,
    kpts: List[int] = [10, 10, 1],
    calculation_type: CalculationType = "scf",
    system_type: SystemType = "slab",
    use_vaspsol: bool = False,
) -> Vasp:
    """Create VASP calculator with appropriate settings."""
    calc_params = BASE_PARAMETERS.copy()
    if calculation_type == "scf":
        calc_params["IBRION"] = -1
        calc_params["NSW"] = 0
    else:
        calc_params["IBRION"] = 2
        calc_params["NSW"] = 200

    if system_type == "molecule":
        kpts = [1, 1, 1]
    if use_vaspsol:
        calc_params.update(SOLVATION_PARAMETERS)

    calc_params = optimise_vasp_parallelisation(calc_params, kpts)
    calc = Vasp(
        directory=str(directory),
        kpts=kpts,
        xc="PBE",
        setups="recommended",
        **calc_params,
    )

    return calc


def generate_kpts(system_type: SystemType = "slab") -> List[int]:
    """Generate k-points grid based on system type."""
    if system_type == "slab":
        return [10, 10, 1]
    elif system_type == "molecule":
        return [1, 1, 1]
    else:
        return [10, 10, 10]


def load_lithium_slab(filename: str) -> Tuple[Any, SlabAdsorptionSites]:
    """Load and prepare lithium slab structure."""
    slab = read(filename) * (2, 2, 1)
    slab.pbc = True
    slab.set_initial_magnetic_moments([0] * len(slab))
    layers, _ = get_layers(slab, (0, 0, 1), tolerance=0.01)
    n_layers = len(np.unique(layers))
    print(f"\nDetected {n_layers} layers in the structure")
    custom_surface = CustomSurface(slab, n_layers=n_layers)
    sas = SlabAdsorptionSites(
        slab, surface=custom_surface, surrogate_metal="Li", both_sides=False
    )
    return slab, sas


def create_f2_molecule() -> Atoms:
    """Create F2 molecule in a vacuum cell."""
    f2 = Atoms("F2", positions=[[0, 0, 0], [0, 0, 1.42]], cell=[20, 20, 20])
    f2.center()
    f2.set_initial_magnetic_moments([0] * len(f2))
    f2.pbc = True
    return f2


def generate_distances(use_height: bool = False) -> np.ndarray:
    """Generate array of distances for scanning."""
    if use_height:
        return np.arange(1.5, 1.8, 0.05)
    return np.arange(1.4, 1.80, 0.05)


def run_calculation(
    atoms: Atoms,
    directory: Path,
    calculation_type: CalculationType = "scf",
    system_type: SystemType = "slab",
    use_vaspsol: bool = False,
) -> Tuple[float, Atoms]:
    """Run VASP calculation."""
    directory.mkdir(exist_ok=True, parents=True)
    kpts = generate_kpts(system_type)
    calc = create_vasp_calculator(
        directory,
        kpts=kpts,
        calculation_type=calculation_type,
        system_type=system_type,
        use_vaspsol=use_vaspsol,
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    if calculation_type == "relax":
        relaxed_atoms = atoms.copy()
        return energy, relaxed_atoms

    return energy, atoms


def create_structure_at_height(
    base_atoms: Any,
    sas: SlabAdsorptionSites,
    distance: float,
    site_type: str = "ontop",
    use_height: bool = False,
) -> Any:
    """Create structure with adsorbate at specified height/distance."""
    atoms = base_atoms.copy()
    height = (
        distance
        if use_height
        else calculate_height_for_site(atoms, sas, distance, site_type)
    )
    add_adsorbate(
        atoms, adsorbate="F", site=site_type, height=height, adsorption_sites=sas
    )
    atoms.set_initial_magnetic_moments([0] * len(atoms))
    return atoms


def plot_energy_curve(
    distances: np.ndarray,
    energies: List[float],
    adsorption_energies: List[float],
    output_dir: Path,
    environment: str = "vacuum",
) -> float:
    """Plot energy curves and return optimal distance."""
    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, "bo-")
    plt.xlabel("Li-F Distance (Å)")
    plt.ylabel("Total Energy (eV)")
    plt.title(f"Total Energy vs Li-F Bond Length ({environment})")
    plt.grid(True)
    plt.savefig(output_dir / f"total_energy_curve_{environment}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(distances, adsorption_energies, "ro-")
    plt.xlabel("Li-F Distance (Å)")
    plt.ylabel("Adsorption Energy (eV)")
    plt.title(f"Adsorption Energy vs Li-F Bond Length ({environment})")
    plt.grid(True)
    plt.savefig(output_dir / f"adsorption_energy_curve_{environment}.png")
    plt.close()

    min_idx = np.argmin(adsorption_energies)
    return distances[min_idx]


def calculate_adsorption_energy(
    total_energy: float, clean_li_energy: float, f2_energy: float
) -> float:
    """Calculate adsorption energy."""
    return total_energy - clean_li_energy - 0.5 * f2_energy


def write_formatted_results(
    output_file: Path,
    clean_energy: float,
    f2_energy: float,
    f2_bond_length: float,
    distances: np.ndarray,
    energies: List[float],
    adsorption_energies: List[float],
    optimal_distance: float,
    optimised_structure: Any,
    optimised_energy: float,
    optimised_ads_energy: float,
    site_type: str,
    base_atoms: Any,
    sas: Any,
    use_height: bool = False,
    environment: str = "vacuum",
) -> None:
    """Write analysis results to file."""
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(
            f"FLUORINE ADSORPTION ON Li SURFACE - {site_type.upper()} SITE ANALYSIS IN {environment.upper()}\n"
        )
        f.write("=" * 80 + "\n\n")
        f.write("REFERENCE ENERGIES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Clean Li slab energy:         {clean_energy:.3f} eV\n")
        f.write(f"F2 molecule energy:           {f2_energy:.3f} eV\n")
        f.write(f"F2 bond length:               {f2_bond_length:.3f} Å\n\n")
        f.write("DISTANCE SCAN RESULTS\n")
        f.write("-" * 80 + "\n")

        if use_height:
            f.write(
                f"{'Height (Å)':>12} {'Li-F (Å)':>18} {'Total Energy (eV)':>18} {'Adsorption Energy (eV)':>22}\n"
            )
        else:
            f.write(
                f"{'Li-F (Å)':>12} {'Height (Å)':>18} {'Total Energy (eV)':>18} {'Adsorption Energy (eV)':>22}\n"
            )

        f.write("-" * 80 + "\n")
        for d, e, ae in zip(distances, energies, adsorption_energies):
            if use_height:
                bond_length = get_bond_length_from_height(base_atoms, sas, d, site_type)
                f.write(f"{d:12.3f} {bond_length:18.3f} {e:18.3f} {ae:22.3f}\n")
            else:
                height = calculate_height_for_site(base_atoms, sas, d, site_type)
                f.write(f"{d:12.3f} {height:18.3f} {e:18.3f} {ae:22.3f}\n")

        f.write("\nOPTIMISATION RESULTS\n")
        f.write("-" * 80 + "\n")
        if use_height:
            f.write(f"Optimal height:              {optimal_distance:.3f} Å\n")
            opt_bond_length = get_bond_length_from_height(
                base_atoms, sas, optimal_distance, site_type
            )
            f.write(f"Corresponding Li-F distance: {opt_bond_length:.3f} Å\n")
        else:
            f.write(f"Optimal Li-F distance:        {optimal_distance:.3f} Å\n")
            opt_height = calculate_height_for_site(
                base_atoms, sas, optimal_distance, site_type
            )
            f.write(f"Optimal ACAT height:          {opt_height:.3f} Å\n")

        final_li_f = optimised_structure.get_distances(
            -1, range(len(optimised_structure) - 1)
        )[0]
        final_height = optimised_structure.positions[-1][2] - np.mean(
            optimised_structure.positions[:-1][:, 2]
        )
        f.write(f"Final optimised Li-F:         {final_li_f:.3f} Å\n")
        f.write(f"Final height:                 {final_height:.3f} Å\n")
        f.write(f"Final optimised energy:       {optimised_energy:.3f} eV\n")
        f.write(f"Final adsorption energy:      {optimised_ads_energy:.3f} eV\n")
        f.write("\n" + "=" * 80 + "\n")


def main(
    slab_file: str,
    site_type: str = "ontop",
    use_height: bool = False,
    environments: List[str] = ["vacuum", "solvent"],
):
    """Main function to run adsorption analysis."""

    for environment in environments:
        use_vaspsol = environment != "vacuum"

        base_dir = Path(f"Li-F_{site_type}_{environment}")
        base_dir.mkdir(exist_ok=True)
        base_atoms, sas = load_lithium_slab(slab_file)

        print(f"\n{'=' * 50}")
        print(f"STARTING CALCULATIONS IN {environment.upper()} ENVIRONMENT")
        print(f"{'=' * 50}")

        print("Optimising clean Li slab...")
        clean_dir = base_dir / "clean_li"
        clean_dir.mkdir(exist_ok=True)
        clean_energy, optimised_li = run_calculation(
            base_atoms,
            clean_dir,
            calculation_type="relax",
            system_type="slab",
            use_vaspsol=use_vaspsol,
        )
        print(f"Clean Li slab optimised energy: {clean_energy:.3f} eV")

        print("\nOptimising F2 molecule...")
        f2_dir = base_dir / "f2"
        f2_dir.mkdir(exist_ok=True)
        f2_molecule = create_f2_molecule()
        f2_energy, optimised_f2 = run_calculation(
            f2_molecule,
            f2_dir,
            calculation_type="relax",
            system_type="molecule",
            use_vaspsol=use_vaspsol,
        )
        print(f"F2 molecule optimised energy: {f2_energy:.3f} eV")
        print(
            f"F2 bond length after optimisation: {optimised_f2.get_distance(0, 1):.3f} Å"
        )

        distances = generate_distances(use_height)
        energies = []
        adsorption_energies = []

        print("\nStarting distance calculations...")
        for i, distance in enumerate(distances):
            calc_dir = base_dir / f"{site_type}_{distance:.2f}"
            calc_dir.mkdir(exist_ok=True)

            atoms = create_structure_at_height(
                optimised_li, sas, distance, site_type=site_type, use_height=use_height
            )
            energy, _ = run_calculation(
                atoms,
                calc_dir,
                calculation_type="scf",
                system_type="slab",
                use_vaspsol=use_vaspsol,
            )
            ads_energy = calculate_adsorption_energy(energy, clean_energy, f2_energy)

            energies.append(energy)
            adsorption_energies.append(ads_energy)

            if use_height:
                bond_length = get_bond_length_from_height(
                    atoms, sas, distance, site_type
                )
                print(
                    f"Completed {i + 1}/{len(distances)}: "
                    f"height={distance:.3f} Å, Li-F={bond_length:.3f} Å, "
                    f"E={energy:.3f} eV, E_ads={ads_energy:.3f} eV"
                )
            else:
                height = calculate_height_for_site(atoms, sas, distance, site_type)
                print(
                    f"Completed {i + 1}/{len(distances)}: "
                    f"Li-F={distance:.3f} Å, height={height:.3f} Å, "
                    f"E={energy:.3f} eV, E_ads={ads_energy:.3f} eV"
                )

        optimal_distance = plot_energy_curve(
            distances, energies, adsorption_energies, base_dir, environment
        )

        print("\nPerforming final optimisation at optimal distance...")
        optimise_dir = base_dir / "optimisation"
        optimise_dir.mkdir(exist_ok=True)

        final_structure = create_structure_at_height(
            optimised_li,
            sas,
            optimal_distance,
            site_type=site_type,
            use_height=use_height,
        )
        optimised_energy, optimised_structure = run_calculation(
            final_structure,
            optimise_dir,
            calculation_type="relax",
            system_type="slab",
            use_vaspsol=use_vaspsol,
        )
        optimised_ads_energy = calculate_adsorption_energy(
            optimised_energy, clean_energy, f2_energy
        )

        write_formatted_results(
            base_dir / f"results_{environment}.txt",
            clean_energy,
            f2_energy,
            optimised_f2.get_distance(0, 1),
            distances,
            energies,
            adsorption_energies,
            optimal_distance,
            optimised_structure,
            optimised_energy,
            optimised_ads_energy,
            site_type,
            optimised_li,
            sas,
            use_height,
            environment,
        )

        print(
            f"\nAnalysis in {environment} environment complete. Results saved to results_{environment}.txt"
        )
        write(
            base_dir / f"optimised_structure_{environment}.vasp",
            optimised_structure,
            format="vasp",
        )

    print("\nAll calculations complete!")


if __name__ == "__main__":
    main(
        "/scratch/ba3g18/VASP/Lithium/Convergence/[100]/Surface_slab_24/vasprun.xml",
        site_type="ontop",
        use_height=True,
        environments=["solvent"], 
    )