import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.atoms import Atoms
import matplotlib.pyplot as plt
from acat.build import add_adsorbate
from acat.settings import CustomSurface
from typing import List, Tuple, Dict, Any, Literal
from ase.calculators.aims import Aims, AimsProfile
from acat.adsorption_sites import SlabAdsorptionSites, get_layers

KSPACING = 0.1
SystemType = Literal["slab", "molecule", "bulk"]
CalculationType = Literal["scf", "relax"]

BASE_PARAMETERS = {
    "xc": "pbe",
    "relativistic": "atomic_zora scalar",
    "occupation_type": "cold 0.1",
    "sc_accuracy_rho": 1e-4,
    "sc_iter_limit": 300,
    "mixer": "pulay",
    "n_max_pulay": 14,
    "charge_mix_param": 0.01,
    "output_level": "MD_light",
    "override_error_charge_integration": True,
}


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


def create_aims_profile() -> AimsProfile:
    """Create FHI-aims profile with appropriate paths."""
    return AimsProfile(
        command="srun --mpi=pmix /home/ba3g18/Repos/FHIaims/build/aims.250131.scalapack.mpi.x",
        default_species_directory="/home/ba3g18/Repos/FHIaims/species_defaults/defaults_2020/intermediate/",
    )


def generate_kpts(atoms: Atoms, system_type: SystemType = "slab") -> list:
    """Generate k-points grid based on system type."""
    cell_lengths = atoms.cell.lengths()
    kpts = np.ceil(2 * np.pi / (cell_lengths * KSPACING)).astype(int)

    if system_type == "slab":
        kpts[2] = 1

    return kpts.tolist()


def load_lithium_slab(filename: str) -> Tuple[Any, SlabAdsorptionSites]:
    """Load and prepare lithium slab structure."""
    slab = read(filename)
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
    return f2


def get_input_parameters(calculation_type: CalculationType = "scf") -> Dict[str, Any]:
    """Get FHI-aims calculation parameters."""
    params = BASE_PARAMETERS.copy()
    if calculation_type == "relax":
        params.update({"relax_geometry": "bfgs 1e-2", "relax_unit_cell": "none"})
    return params


def generate_distances(use_height: bool = False) -> np.ndarray:
    """Generate array of distances for scanning."""
    if use_height:
        return np.arange(-1.0, 1.0, 0.05)
    return np.arange(1.8, 2.1, 0.05)


def run_calculation(
    atoms: Atoms,
    directory: Path,
    profile: AimsProfile,
    calculation_type: CalculationType = "scf",
    system_type: SystemType = "slab",
) -> Tuple[float, Atoms]:
    """Run FHI-aims calculation."""
    params = get_input_parameters(calculation_type)
    if system_type != "molecule":
        params["k_grid"] = generate_kpts(atoms, system_type)

    calc = Aims(profile=profile, directory=str(directory), **params)
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    if calculation_type == "relax":
        relaxed_atoms = read(directory / "aims.out")
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
    return atoms


def plot_energy_curve(
    distances: np.ndarray,
    energies: List[float],
    adsorption_energies: List[float],
    output_dir: Path,
) -> float:
    """Plot energy curves and return optimal distance."""
    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, "bo-")
    plt.xlabel("Li-F Distance (Å)")
    plt.ylabel("Total Energy (eV)")
    plt.title("Total Energy vs Li-F Bond Length")
    plt.grid(True)
    plt.savefig(output_dir / "total_energy_curve.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(distances, adsorption_energies, "ro-")
    plt.xlabel("Li-F Distance (Å)")
    plt.ylabel("Adsorption Energy (eV)")
    plt.title("Adsorption Energy vs Li-F Bond Length")
    plt.grid(True)
    plt.savefig(output_dir / "adsorption_energy_curve.png")
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
) -> None:
    """Write analysis results to file."""
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(
            f"FLUORINE ADSORPTION ON Li SURFACE - {site_type.upper()} SITE ANALYSIS\n"
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


def main(slab_file: str, site_type: str = "ontop", use_height: bool = False):
    """Main function to run adsorption analysis."""
    base_dir = Path(f"Li-F_{site_type}")
    base_dir.mkdir(exist_ok=True)
    base_atoms, sas = load_lithium_slab(slab_file)
    profile = create_aims_profile()

    print("Optimising clean Li slab...")
    clean_dir = base_dir / "clean_li"
    clean_dir.mkdir(exist_ok=True)
    clean_energy, optimised_li = run_calculation(
        base_atoms, clean_dir, profile, "relax", system_type="slab"
    )
    print(f"Clean Li slab optimised energy: {clean_energy:.3f} eV")

    print("\nOptimising F2 molecule...")
    f2_dir = base_dir / "f2"
    f2_dir.mkdir(exist_ok=True)
    f2_molecule = create_f2_molecule()
    f2_energy, optimised_f2 = run_calculation(
        f2_molecule, f2_dir, profile, "relax", system_type="molecule"
    )
    print(f"F2 molecule optimised energy: {f2_energy:.3f} eV")
    print(f"F2 bond length after optimisation: {optimised_f2.get_distance(0, 1):.3f} Å")

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
        energy, _ = run_calculation(atoms, calc_dir, profile, "scf", system_type="slab")
        ads_energy = calculate_adsorption_energy(energy, clean_energy, f2_energy)

        energies.append(energy)
        adsorption_energies.append(ads_energy)

        if use_height:
            bond_length = get_bond_length_from_height(atoms, sas, distance, site_type)
            print(
                f"Completed {i+1}/{len(distances)}: "
                f"height={distance:.3f} Å, Li-F={bond_length:.3f} Å, "
                f"E={energy:.3f} eV, E_ads={ads_energy:.3f} eV"
            )
        else:
            height = calculate_height_for_site(atoms, sas, distance, site_type)
            print(
                f"Completed {i+1}/{len(distances)}: "
                f"Li-F={distance:.3f} Å, height={height:.3f} Å, "
                f"E={energy:.3f} eV, E_ads={ads_energy:.3f} eV"
            )

    optimal_distance = plot_energy_curve(
        distances, energies, adsorption_energies, base_dir
    )

    print("\nPerforming final optimisation at optimal distance...")
    optimise_dir = base_dir / "optimisation"
    optimise_dir.mkdir(exist_ok=True)

    final_structure = create_structure_at_height(
        optimised_li, sas, optimal_distance, site_type=site_type, use_height=use_height
    )
    optimised_energy, optimised_structure = run_calculation(
        final_structure, optimise_dir, profile, "relax", system_type="slab"
    )
    optimised_ads_energy = calculate_adsorption_energy(
        optimised_energy, clean_energy, f2_energy
    )

    write_formatted_results(
        base_dir / "results.txt",
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
    )

    print("\nAnalysis complete. Results saved to results.txt")


if __name__ == "__main__":
    main(
        "/scratch/ba3g18/FHI-aims/Lithium/Adsorption/3_FOLD/Li.xyz",
        site_type="3fold",
        use_height=True,
    )
