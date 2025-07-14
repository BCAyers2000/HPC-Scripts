import numpy as np
from pathlib import Path
from ase.atoms import Atoms
import matplotlib.pyplot as plt
from ase.cluster import Octahedron
from acat.build import add_adsorbate
from typing import List, Tuple, Dict, Any
from acat.adsorption_sites import ClusterAdsorptionSites
from ase.calculators.espresso import Espresso, EspressoProfile


def calculate_height_for_site(atoms: Any, cas: Any, desired_bond_length: float, site_type: str, surface: str) -> float:
    """Calculate ACAT height parameter for any site type given desired metal-adsorbate bond length
    
    Args:
        atoms: ASE Atoms object of the cluster
        cas: ClusterAdsorptionSites object
        desired_bond_length: Target metal-adsorbate bond length
        site_type: Type of adsorption site (e.g., 'bridge', 'ontop', 'hollow')
        surface: Surface type (e.g., 'edge', 'face')
        
    Returns:
        float: Calculated height parameter for ACAT
    """
    temp_atoms = atoms.copy()
    add_adsorbate(
        temp_atoms,
        adsorbate="O",
        site=site_type,
        surface=surface,
        height=2.0,  # Temporary height
        adsorption_sites=cas,
    )
    
    o_idx = len(temp_atoms) - 1
    distances = temp_atoms.get_distances(o_idx, range(o_idx))
    site_indices = np.argsort(distances)[:4]  # Get up to 4 nearest neighbors
    site_positions = temp_atoms.positions[site_indices]

    if site_type == "ontop":
        return desired_bond_length
    
    elif site_type in ["bridge", "longbridge", "shortbridge"]:
        metal_dist = np.linalg.norm(site_positions[0] - site_positions[1])
        return np.sqrt(desired_bond_length**2 - (metal_dist/2)**2)
    
    elif site_type in ["3fold", "hollow", "fcc", "hcp"]:
        # Use first 3 positions for 3-fold sites
        positions = site_positions[:3]
        center = np.mean(positions, axis=0)
        radii = [np.linalg.norm(pos - center) for pos in positions]
        max_radius = min(radii)  # Use minimum radius for stability
        return np.sqrt(desired_bond_length**2 - max_radius**2)
    
    elif site_type == "4fold":
        center = np.mean(site_positions, axis=0)
        radius = max(np.linalg.norm(pos - center) for pos in site_positions)
        return np.sqrt(desired_bond_length**2 - radius**2)
    
    else:
        raise ValueError(f"Unknown site type: {site_type}")


def create_espresso_profile() -> EspressoProfile:
    """Creates EspressoProfile"""
    return EspressoProfile(
        command="srun --mpi=pmix /iridisfs/home/ba3g18/Repos/q-e/bin/pw.x",
        pseudo_dir="/home/ba3g18/Repos/Pseudopotentials/SSSP_1.3.0_PBE_efficiency",
    )


def create_base_structure(metal="Pt") -> Tuple[Any, ClusterAdsorptionSites]:
    """Creates base cluster structure with adsorption sites"""
    atoms = Octahedron(metal, 3, 1)
    atoms.set_cell([20, 20, 20])
    atoms.center()
    magmoms = np.zeros(len(atoms))
    magmoms[:] = 2.0
    atoms.set_initial_magnetic_moments(magmoms)
    cas = ClusterAdsorptionSites(
        atoms,
        allow_6fold=False,
        composition_effect=False,
        label_sites=True,
        surrogate_metal=metal,
    )
    return atoms, cas


def create_o2_molecule() -> Atoms:
    """Creates O2 molecule centred in box with magnetic moments"""
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]], cell=[20, 20, 20])
    o2.center()
    magmoms = np.zeros(2)
    magmoms[:] = 1.0  # Set O2 magnetic moments to 1.0
    o2.set_initial_magnetic_moments(magmoms)
    return o2


def get_input_data(
    calculation_type: str = "scf", 
    includes_oxygen: bool = False,
    oxygen_molecule: bool = False
) -> Dict[str, Any]:
    """Returns QE input parameters for given calculation type
    
    Args:
        calculation_type: Type of calculation ('scf' or 'relax')
        includes_oxygen: Whether system includes oxygen atoms
        oxygen_molecule: Whether system is specifically O2 molecule
    """
    input_data = {
        "control": {
            "calculation": calculation_type,
            "verbosity": "high",
            "restart_mode": "from_scratch",
            "nstep": 999,
            "tstress": False,
            "tprnfor": True,
            "etot_conv_thr": 1e-05,
            "forc_conv_thr": 4e-04,
        },
        "system": {
            "ibrav": 0,
            "ecutwfc": 50.0,
            "ecutrho": 800.0,
            "occupations": "smearing",
            "degauss": 0.000735 if oxygen_molecule else 0.00735,
            "assume_isolated": "mt",
            "smearing": "gaussian",
            "input_dft": "pbe",
            "ntyp": 2 if includes_oxygen else 1,
        },
        "electrons": {
            "electron_maxstep": 999,
            "scf_must_converge": True,
            "conv_thr": 1e-10,
            "mixing_mode": "local-TF",
            "mixing_beta": 0.15,
            "diagonalization": "rmm-davidson",
            "startingwfc": "random",
        },
    }

    if includes_oxygen:
        input_data["system"]["nspin"] = 2

    if calculation_type == "relax":
        input_data["ions"] = {"ion_dynamics": "bfgs", "upscale": 1e2}

    return input_data


def generate_distances() -> np.ndarray:
    """Generates array of Pt-O distances"""
    return np.arange(2.0, 2.4, 0.05)


def calculate_energy(
    atoms: Any,
    directory: Path,
    pseudopotentials: Dict[str, str],
    includes_oxygen: bool = False,
) -> float:
    """Runs QE calculation and returns energy"""
    profile = create_espresso_profile()
    calc = Espresso(
        input_data=get_input_data("scf", includes_oxygen, oxygen_molecule=False),
        pseudopotentials=pseudopotentials,
        profile=profile,
        directory=str(directory),
        kpts=None,
    )
    atoms.calc = calc
    return atoms.get_potential_energy()


def optimise_structure(
    atoms: Any,
    directory: Path,
    pseudopotentials: Dict[str, str],
    includes_oxygen: bool = False,
    oxygen_molecule: bool = False,
) -> Tuple[Any, float]:
    """Optimises structure using QE and returns optimised structure and energy"""
    profile = create_espresso_profile()
    calc = Espresso(
        input_data=get_input_data("relax", includes_oxygen, oxygen_molecule=oxygen_molecule),
        pseudopotentials=pseudopotentials,
        profile=profile,
        directory=str(directory),
        kpts=None,
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    return atoms, energy


def create_structure_at_height(
    base_atoms: Any, cas: ClusterAdsorptionSites, metal_o_distance: float,
    site_type: str = "bridge", surface: str = "edge"
) -> Any:
    """Creates structure with O at specified metal-O distance at given site
    
    Args:
        base_atoms: Base cluster structure
        cas: ClusterAdsorptionSites object
        metal_o_distance: Desired metal-oxygen bond length
        site_type: Type of adsorption site
        surface: Surface type
    """
    atoms = base_atoms.copy()
    height = calculate_height_for_site(atoms, cas, metal_o_distance, site_type, surface)
    
    add_adsorbate(
        atoms,
        adsorbate="O",
        site=site_type,
        surface=surface,
        height=height,
        adsorption_sites=cas,
    )
    magmoms = np.zeros(len(atoms))
    magmoms[:-1] = 2.0
    magmoms[-1] = 2.0
    atoms.set_initial_magnetic_moments(magmoms)
    return atoms


def plot_energy_curve(
    distances: np.ndarray,
    energies: List[float],
    adsorption_energies: List[float],
    output_dir: Path,
) -> float:
    """Plots energy curves vs Pt-O distance and returns minimum distance"""
    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, "bo-")
    plt.xlabel("Pt-O Distance (Å)")
    plt.ylabel("Total Energy (eV)")
    plt.title("Total Energy vs Pt-O Bond Length")
    plt.grid(True)
    plt.savefig(output_dir / "total_energy_curve.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(distances, adsorption_energies, "ro-")
    plt.xlabel("Pt-O Distance (Å)")
    plt.ylabel("Adsorption Energy (eV)")
    plt.title("Adsorption Energy vs Pt-O Bond Length")
    plt.grid(True)
    plt.savefig(output_dir / "adsorption_energy_curve.png")
    plt.close()

    min_idx = np.argmin(adsorption_energies)
    return distances[min_idx]


def calculate_adsorption_energy(
    total_energy: float, clean_pt_energy: float, o2_energy: float
) -> float:
    """Calculates adsorption energy using E_ads = E_total - E_clean - 1/2 E_O2"""
    return total_energy - clean_pt_energy - 0.5 * o2_energy


def write_formatted_results(
    output_file: Path,
    clean_energy: float,
    o2_energy: float,
    o2_bond_length: float,
    pt_o_distances: np.ndarray,
    energies: List[float],
    adsorption_energies: List[float],
    optimal_distance: float,
    optimised_structure: Any,
    optimised_energy: float,
    optimised_ads_energy: float,
    site_type: str,
    surface: str,
    base_atoms: Any,  # Added base atoms parameter
    cas: Any,         # Added ClusterAdsorptionSites parameter
) -> None:
    """Writes all results to a nicely formatted text file"""
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"OXYGEN ADSORPTION ON Pt13 CLUSTER - {site_type.upper()} SITE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("REFERENCE ENERGIES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Clean Pt13 cluster energy:      {clean_energy:.3f} eV\n")
        f.write(f"O2 molecule energy:             {o2_energy:.3f} eV\n")
        f.write(f"O2 bond length:                 {o2_bond_length:.3f} Å\n\n")
        f.write("DISTANCE SCAN RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'M-O (Å)':>12} {'ACAT height (Å)':>18} {'Total Energy (eV)':>18} {'Adsorption Energy (eV)':>22}\n"
        )
        f.write("-" * 80 + "\n")
        for d, e, ae in zip(pt_o_distances, energies, adsorption_energies):
            height = calculate_height_for_site(base_atoms, cas, d, site_type, surface)
            f.write(f"{d:12.3f} {height:18.3f} {e:18.3f} {ae:22.3f}\n")
        f.write("\n")
        f.write("OPTIMISATION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Optimal M-O distance:          {optimal_distance:.3f} Å\n")
        opt_height = calculate_height_for_site(base_atoms, cas, optimal_distance, site_type, surface)
        f.write(f"Optimal ACAT height:            {opt_height:.3f} Å\n")
        final_m_o = optimised_structure.get_distances(-1, 0)[0]
        final_height = calculate_height_for_site(base_atoms, cas, final_m_o, site_type, surface)
        f.write(f"Final optimised M-O:           {final_m_o:.3f} Å\n")
        f.write(f"Final ACAT height:              {final_height:.3f} Å\n")
        f.write(f"Final optimised energy:         {optimised_energy:.3f} eV\n")
        f.write(f"Final adsorption energy:        {optimised_ads_energy:.3f} eV\n")
        f.write("\n" + "=" * 80 + "\n")


def main(site_type: str = "4fold", surface: str = "fcc100"):
    """Main function to run O adsorption analysis
    
    Args:
        site_type: Type of adsorption site to analyze
        surface: Surface type for adsorption
    """
    base_dir = Path(f"Pt-O_{site_type}_{surface}")
    base_dir.mkdir(exist_ok=True)
    base_atoms, cas = create_base_structure()

    pseudopotentials = {
        "Pt": "pt_pbe_v1.4.uspp.F.UPF",
        "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
    }

    print("Optimising clean Pt13 cluster...")
    clean_dir = base_dir / "clean_pt"
    clean_dir.mkdir(exist_ok=True)
    optimised_pt, clean_energy = optimise_structure(
        base_atoms, clean_dir, pseudopotentials, includes_oxygen=False
    )
    print(f"Clean Pt13 cluster optimised energy: {clean_energy:.3f} eV")

    print("Optimising O2 molecule...")
    o2_dir = base_dir / "o2"
    o2_dir.mkdir(exist_ok=True)
    o2_molecule = create_o2_molecule()
    optimised_o2, o2_energy = optimise_structure(
        o2_molecule, o2_dir, pseudopotentials, includes_oxygen=True, oxygen_molecule=True
    )
    print(f"O2 molecule optimised energy: {o2_energy:.3f} eV")
    print(f"O2 bond length after optimisation: {optimised_o2.get_distance(0, 1):.3f} Å")

    pt_o_distances = generate_distances()
    energies = []
    adsorption_energies = []

    print("Starting distance calculations...")
    for i, distance in enumerate(pt_o_distances):
        calc_dir = base_dir / f"{site_type}_{distance:.2f}"
        calc_dir.mkdir(exist_ok=True)

        atoms = create_structure_at_height(
            optimised_pt, cas, distance, site_type=site_type, surface=surface
        )
        energy = calculate_energy(
            atoms, calc_dir, pseudopotentials, includes_oxygen=True
        )
        ads_energy = calculate_adsorption_energy(energy, clean_energy, o2_energy)

        energies.append(energy)
        adsorption_energies.append(ads_energy)
        height = calculate_height_for_site(atoms, cas, distance, site_type, surface)
        print(
            f"Completed {i+1}/{len(pt_o_distances)}: "
            f"M-O={distance:.3f} Å, height={height:.3f} Å, "
            f"E={energy:.3f} eV, E_ads={ads_energy:.3f} eV"
        )

    optimal_distance = plot_energy_curve(
        pt_o_distances, energies, adsorption_energies, base_dir
    )

    print("\nPerforming final optimisation at optimal distance...")
    optimise_dir = base_dir / "optimisation"
    optimise_dir.mkdir(exist_ok=True)

    final_structure = create_structure_at_height(
        optimised_pt, cas, optimal_distance, site_type=site_type, surface=surface
    )
    optimised_structure, optimised_energy = optimise_structure(
        final_structure, optimise_dir, pseudopotentials, includes_oxygen=True
    )
    optimised_ads_energy = calculate_adsorption_energy(
        optimised_energy, clean_energy, o2_energy
    )

    write_formatted_results(
        base_dir / "results.txt",
        clean_energy,
        o2_energy,
        optimised_o2.get_distance(0, 1),
        pt_o_distances,
        energies,
        adsorption_energies,
        optimal_distance,
        optimised_structure,
        optimised_energy,
        optimised_ads_energy,
        site_type,
        surface,
        optimised_pt, 
        cas           
    )

    print("\nAnalysis complete. Results saved to results.txt")


if __name__ == "__main__":
    main()