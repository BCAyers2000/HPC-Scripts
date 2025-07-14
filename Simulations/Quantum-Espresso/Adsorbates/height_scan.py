"""
GENERAL ADSORPTION ENERGY CALCULATOR WITH SOLVATION, CONSTRAINT, AND EXTERNAL OPTIMISER FUNCTIONALITIES
"""

import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.atoms import Atoms
from ase.build import molecule
import matplotlib.pyplot as plt
from ase.constraints import FixAtoms
from ase.optimize import GoodOldQuasiNewton
from acat.build import add_adsorbate, add_adsorbate_to_site
from acat.settings import CustomSurface
from typing import List, Tuple, Dict, Any, Literal
from ase.io.espresso import write_fortran_namelist
from ase.calculators.espresso import Espresso, EspressoProfile
from acat.adsorption_sites import SlabAdsorptionSites, get_layers

KSPACING = 0.1
SystemType = Literal["slab", "molecule", "bulk"]
CalculationType = Literal["scf", "relax"]

BASE_PARAMETERS = {
    "control": {
        "restart_mode": "from_scratch",
        "nstep": 999,
        "tprnfor": True,
        "disk_io": "none",
        "etot_conv_thr": 1e-04,
        "forc_conv_thr": 4e-04,
    },
    "system": {
        "ecutwfc": 80.0,
        "ecutrho": 800.0,
        "degauss": 0.01,
        "smearing": "cold",
        "occupations": "smearing",
    },
    "electrons": {
        "conv_thr": 1e-8,
        "mixing_beta": 0.10,
        "mixing_mode": "plain",
        "electron_maxstep": 250,
        "diagonalization": "davidson",
        "startingwfc": "atomic+random",
    },
    "ions": {"ion_dynamics": "bfgs", "bfgs_ndim": 10, "upscale": 1e3},
}


def get_environ_parameters():
    """Standard input data for environ calculations."""
    return {
            "environ": {
                "verbose": 0,
                "cion(1)": 1.0,
                "cion(2)": 1.0,
                "zion(1)": 1.0,
                "zion(2)": -1.0,
                "cionmax": 10.0,
                "system_dim": 2,
                "system_axis": 3,
                "environ_thr": 1.0,
                "env_pressure": 0.0,
                "temperature": 300.0,
                "environ_type": "input",
                "env_electrostatic": True,
                "env_electrolyte_ntyp": 2,
                "env_surface_tension": 37.3,
                "electrolyte_entropy": "full",
                "env_static_permittivity": 89.9,
            },
            "boundary": {
                "alpha": 1.32,
                "radius_mode": "bondi",
                "solvent_mode": "ionic",
                "electrolyte_mode": "ionic",
            },
            "electrostatic": {
                "pbc_dim": 2,
                "pbc_axis": 3,
                "tol": 1.0e-15,
                "inner_tol": 1.0e-20,
                "pbc_correction": "parabolic",
            },
        }


def find_closest_site_to_centre(sites, slab_centre, site_type=None):
    """Find the site of given type closest to the centre of the slab."""
    closest_site = None
    min_distance = float("inf")

    for site in sites:
        if site_type and site["site"] != site_type:
            continue

        position = site["position"]
        distance = np.sqrt(
            (position[0] - slab_centre[0]) ** 2 + (position[1] - slab_centre[1]) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            closest_site = site

    return closest_site, min_distance


def get_site_statistics(all_sites):
    """Get count of each site type in the list of sites."""
    site_counts = {}
    for site in all_sites:
        site_type = site["site"]
        if site_type not in site_counts:
            site_counts[site_type] = 0
        site_counts[site_type] += 1
    return site_counts


def create_espresso_profile(use_environ: bool = False) -> EspressoProfile:
    """Create QE profile with appropriate paths."""
    command = "srun -v --distribution=block:block --hint=nomultithread --cpu-freq=2250000 /mnt/lustre/a2fs-work3/work/e89/e89/ba3g18/Repos/q-e/bin/pw.x"
    if use_environ:
        command += " --environ"

    return EspressoProfile(
        command=command,
        pseudo_dir="/work/e89/e89/ba3g18/Repos/Pseudopotentials/Pslibrary"
    )


def generate_kpts(atoms: Atoms, system_type: SystemType = "slab") -> list:
    """Generate k-points grid based on system type."""
    cell_lengths = atoms.cell.lengths()
    kpts = np.ceil(2 * np.pi / (cell_lengths * KSPACING)).astype(int)
    if system_type == "slab":
        kpts[2] = 1
    return kpts.tolist()


def load_slab(filename: str, surrogate_metal: str = "Li") -> Tuple[Any, SlabAdsorptionSites, Dict, List]:
    """Load and prepare slab structure."""
    slab = read(filename)
    slab.pbc = [True, True, False]
    slab.set_initial_magnetic_moments([0] * len(slab))
    
    layers, _ = get_layers(slab, (0, 0, 1), tolerance=0.01)
    n_layers = len(np.unique(layers))
    print(f"\nDetected {n_layers} layers in the structure")
    
    custom_surface = CustomSurface(slab, n_layers=n_layers)
    sas = SlabAdsorptionSites(
        slab, surface=custom_surface, surrogate_metal=surrogate_metal, both_sides=False
    )

    all_sites = sas.get_sites()
    site_counts = get_site_statistics(all_sites)
    print(f"Found {len(all_sites)} total adsorption sites")
    print("\nSite types available:")
    for st, count in sorted(site_counts.items()):
        print(f"  {st}: {count} sites")

    return slab, sas, site_counts, all_sites


def create_molecule_reference(molecule_name: str) -> Atoms:
    """Create reference molecule in a vacuum cell."""
    mol = molecule(molecule_name)
    mol.set_cell([20.0, 20.0, 20.0])
    mol.center()
    mol.set_initial_magnetic_moments([0] * len(mol))
    return mol


def get_input_parameters(
    calculation_type: CalculationType = "scf",
    includes_adsorbate: bool = False,
    is_molecule: bool = False,
    use_external: bool = False,
) -> Dict[str, Any]:
    """Get QE calculation parameters."""
    params = BASE_PARAMETERS.copy()
    params["control"]["calculation"] = "scf" if use_external else calculation_type
    params["system"]["ntyp"] = 2 if includes_adsorbate else 1
    params["electrons"]["conv_thr"] = 1e-8 if calculation_type == "scf" else 1e-6

    if is_molecule:
        params["system"]["ibrav"] = 0
    else:
        params["system"]["ibrav"] = 12
    if calculation_type == "relax" and not use_external:
        params["ions"] = {"ion_dynamics": "bfgs", "upscale": 1e6, "bfgs_ndim": 1}
    return params


def generate_heights() -> np.ndarray:
    """Generate array of heights for scanning."""
    return np.arange(-1.0, -0.0, 0.1)


def write_environ_file(directory: Path):
    """Write environ input file to directory."""
    with (directory / "environ.in").open("w") as f:
        write_fortran_namelist(f, get_environ_parameters())


def run_calculation(
    atoms: Atoms,
    directory: Path,
    profile: EspressoProfile,
    pseudopotentials: Dict[str, str],
    calculation_type: CalculationType = "scf",
    system_type: SystemType = "slab",
    includes_adsorbate: bool = False,
    is_molecule: bool = False,
    use_environ: bool = False,
    use_external: bool = False,
) -> Tuple[float, Atoms]:
    """Run QE calculation."""
    params = get_input_parameters(calculation_type, includes_adsorbate, is_molecule, use_external)
    
    if system_type != "molecule":
        kpts = generate_kpts(atoms, system_type)
    else:
        kpts = [1, 1, 1]

    if use_environ:
        write_environ_file(directory)

    calc = Espresso(
        input_data=params,
        pseudopotentials=pseudopotentials,
        profile=profile,
        directory=str(directory),
        kpts=kpts,
    )
    
    atoms.calc = calc
    
    if use_external and calculation_type == "relax":
        trajectory_file = directory / "ase_trajectory.traj"
        log_file = directory / "ase_optimisation.log"
        
        print(f"Using ASE external OPTIMISER (GoodOldQuasiNewton) with trajectory: {trajectory_file}")
        opt = GoodOldQuasiNewton(atoms, trajectory=str(trajectory_file), logfile=str(log_file))
        opt.run(fmax=0.05, steps=100)  
        energy = atoms.get_potential_energy()
        return energy, atoms.copy()
    else:
        energy = atoms.get_potential_energy()
        if calculation_type == "relax":
            try:
                relaxed_atoms = read(directory / "espresso.pwo")
                return energy, relaxed_atoms
            except Exception as e:
                print(f"Warning: Could not read optimised structure: {e}")
                return energy, atoms
        return energy, atoms


def create_structure_at_height(
    base_atoms: Any,
    sas: SlabAdsorptionSites,
    all_sites: List,
    height: float,
    adsorbate_atom: str,
    site_type: str = "ontop",
    constrained: bool = False,
    centre_adsorbate: bool = False,
) -> Any:
    """Create structure with adsorbate at specified height."""
    atoms = base_atoms.copy()

    if centre_adsorbate:
        slab_centre = np.mean(atoms.positions, axis=0)
        print(f"\nSlab centre: {slab_centre}")

        selected_site, min_distance = find_closest_site_to_centre(
            all_sites, slab_centre, site_type
        )
        print(f"\nSelected {site_type} site closest to centre:")
        print(f"  Position: {selected_site['position']}")
        print(f"  Distance from centre (xy-plane): {min_distance:.3f} Å")

        add_adsorbate_to_site(atoms, adsorbate=adsorbate_atom, site=selected_site, height=height)
    else:
        add_adsorbate(
            atoms, adsorbate=adsorbate_atom, site=site_type, height=height, adsorption_sites=sas
        )

    atoms.set_initial_magnetic_moments([0] * len(atoms))

    if constrained:
        constraint_mask = [True] * (len(atoms) - 1) + [False]
        constraint = FixAtoms(mask=constraint_mask)
        atoms.set_constraint(constraint)
        print("Applied constraint: Fixed substrate atoms, only adsorbate is allowed to move")

    return atoms


def plot_energy_curve(
    heights: np.ndarray,
    energies: List[float],
    adsorption_energies: List[float],
    output_dir: Path,
    environment: str = "vacuum",
) -> float:
    """Plot energy curves and return optimal height."""
    plt.figure(figsize=(10, 6))
    plt.plot(heights, energies, "bo-")
    plt.xlabel("Height (Å)")
    plt.ylabel("Total Energy (eV)")
    plt.title(f"Total Energy vs Height ({environment})")
    plt.grid(True)
    plt.savefig(output_dir / f"total_energy_curve_{environment}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(heights, adsorption_energies, "ro-")
    plt.xlabel("Height (Å)")
    plt.ylabel("Adsorption Energy (eV)")
    plt.title(f"Adsorption Energy vs Height ({environment})")
    plt.grid(True)
    plt.savefig(output_dir / f"adsorption_energy_curve_{environment}.png")
    plt.close()

    min_idx = np.argmin(adsorption_energies)
    return heights[min_idx]


def calculate_adsorption_energy(
    total_energy: float, 
    clean_slab_energy: float, 
    molecule_energy: float,
    stoichiometry: float = 0.5
) -> float:
    """Calculate adsorption energy."""
    return total_energy - clean_slab_energy - stoichiometry * molecule_energy


def write_formatted_results(
    output_file: Path,
    clean_energy: float,
    molecule_energy: float,
    molecule_bond_length: float,
    heights: np.ndarray,
    energies: List[float],
    adsorption_energies: List[float],
    optimal_height: float,
    optimised_structure: Any,
    optimised_energy: float,
    optimised_ads_energy: float,
    site_type: str,
    environment: str = "vacuum",
    constrained: bool = False,
    centre_adsorbate: bool = False,
    adsorbate_atom: str = "F",
    molecule_name: str = "F2",
    use_external: bool = False,
) -> None:
    """Write analysis results to file."""
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(
            f"{adsorbate_atom.upper()} ADSORPTION ON SURFACE - {site_type.upper()} SITE ANALYSIS IN {environment.upper()}\n"
        )
        if constrained:
            f.write("WITH CONSTRAINED SUBSTRATE ATOMS\n")
        if centre_adsorbate:
            f.write("WITH ADSORBATE PLACED AT CENTRE OF SLAB\n")
        if use_external:
            f.write("WITH EXTERNAL ASE OPTIMISER (GoodOldQuasiNewton)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("REFERENCE ENERGIES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Clean slab energy:            {clean_energy:.3f} eV\n")
        f.write(f"{molecule_name} molecule energy:        {molecule_energy:.3f} eV\n")
        f.write(f"{molecule_name} bond length:            {molecule_bond_length:.3f} Å\n\n")
        
        f.write("HEIGHT SCAN RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Height (Å)':>12} {'Total Energy (eV)':>18} {'Adsorption Energy (eV)':>22}\n")
        f.write("-" * 80 + "\n")
        
        for h, e, ae in zip(heights, energies, adsorption_energies):
            f.write(f"{h:12.3f} {e:18.3f} {ae:22.3f}\n")

        f.write("\nOPTIMISATION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Optimal height:               {optimal_height:.3f} Å\n")
        
        final_height = optimised_structure.positions[-1][2] - np.mean(
            optimised_structure.positions[:-1][:, 2]
        )
        f.write(f"Final optimised height:       {final_height:.3f} Å\n")
        f.write(f"Final optimised energy:       {optimised_energy:.3f} eV\n")
        f.write(f"Final adsorption energy:      {optimised_ads_energy:.3f} eV\n")
        f.write("\n" + "=" * 80 + "\n")


def main(
    slab_file: str,
    molecule_name: str = "F2",
    adsorbate_atom: str = "F",
    site_type: str = "ontop",
    environments: List[str] = ["vacuum", "solvent"],
    constrained: bool = False,
    centre_adsorbate: bool = False,
    surrogate_metal: str = "Li",
    stoichiometry: float = 0.5,
    use_external: bool = False,
):
    """Main function to run adsorption analysis."""
    
    pseudopotentials = {
        "Li": "Li.pbe-sl-kjpaw_psl.1.0.0.UPF",
        "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF ",
    }

    for environment in environments:
        use_environ = environment != "vacuum"

        constraint_suffix = "_constrained" if constrained else ""
        centre_suffix = "_centred" if centre_adsorbate else ""
        external_suffix = "_external" if use_external else ""
        base_dir = Path(
            f"{adsorbate_atom}-adsorption_{site_type}_{environment}{constraint_suffix}{centre_suffix}{external_suffix}"
        )
        base_dir.mkdir(exist_ok=True)
        
        base_atoms, sas, site_counts, all_sites = load_slab(slab_file, surrogate_metal)
        profile = create_espresso_profile(use_environ=use_environ)

        print(f"\n{'=' * 50}")
        print(f"STARTING CALCULATIONS IN {environment.upper()} ENVIRONMENT")
        if constrained:
            print("WITH CONSTRAINED SUBSTRATE ATOMS")
        if centre_adsorbate:
            print("WITH ADSORBATE PLACED AT CENTRE OF SLAB")
        if use_external:
            print("WITH EXTERNAL ASE OPTIMISER (GoodOldQuasiNewton)")
        print(f"{'=' * 50}")

        print("Optimising clean slab...")
        clean_dir = base_dir / "clean_slab"
        clean_dir.mkdir(exist_ok=True)
        clean_energy, optimised_slab = run_calculation(
            base_atoms,
            clean_dir,
            profile,
            pseudopotentials,
            "relax",
            system_type="slab",
            use_environ=use_environ,
            use_external=use_external,
        )
        print(f"Clean slab optimised energy: {clean_energy:.3f} eV")

        print(f"\nOptimising {molecule_name} molecule...")
        mol_dir = base_dir / "molecule"
        mol_dir.mkdir(exist_ok=True)
        ref_molecule = create_molecule_reference(molecule_name)
        molecule_energy, optimised_molecule = run_calculation(
            ref_molecule,
            mol_dir,
            profile,
            pseudopotentials,
            "relax",
            system_type="molecule",
            includes_adsorbate=True,
            is_molecule=True,
            use_environ=use_environ,
            use_external=use_external,
        )
        print(f"{molecule_name} molecule optimised energy: {molecule_energy:.3f} eV")
        
        if len(optimised_molecule) == 2:
            bond_length = optimised_molecule.get_distance(0, 1)
            print(f"{molecule_name} bond length after optimisation: {bond_length:.3f} Å")
        else:
            bond_length = 0.0

        heights = generate_heights()
        energies = []
        adsorption_energies = []

        print("\nStarting height scan calculations...")
        for i, height in enumerate(heights):
            calc_dir = base_dir / f"{site_type}_{height:.2f}"
            calc_dir.mkdir(exist_ok=True)

            atoms = create_structure_at_height(
                optimised_slab,
                sas,
                all_sites,
                height,
                adsorbate_atom,
                site_type=site_type,
                constrained=constrained,
                centre_adsorbate=centre_adsorbate,
            )
            
            energy, _ = run_calculation(
                atoms,
                calc_dir,
                profile,
                pseudopotentials,
                "scf",
                system_type="slab",
                includes_adsorbate=True,
                use_environ=use_environ,
                use_external=False, 
            )
            
            ads_energy = calculate_adsorption_energy(
                energy, clean_energy, molecule_energy, stoichiometry
            )

            energies.append(energy)
            adsorption_energies.append(ads_energy)

            print(
                f"Completed {i + 1}/{len(heights)}: "
                f"height={height:.2f} Å, E={energy:.3f} eV, E_ads={ads_energy:.3f} eV"
            )

        optimal_height = plot_energy_curve(
            heights, energies, adsorption_energies, base_dir, environment
        )

        print("\nPerforming final optimisation at optimal height...")
        optimise_dir = base_dir / "optimisation"
        optimise_dir.mkdir(exist_ok=True)

        final_structure = create_structure_at_height(
            optimised_slab,
            sas,
            all_sites,
            optimal_height,
            adsorbate_atom,
            site_type=site_type,
            constrained=constrained,
            centre_adsorbate=centre_adsorbate,
        )
        
        optimised_energy, optimised_structure = run_calculation(
            final_structure,
            optimise_dir,
            profile,
            pseudopotentials,
            "relax",
            system_type="slab",
            includes_adsorbate=True,
            use_environ=use_environ,
            use_external=use_external,
        )
        
        optimised_ads_energy = calculate_adsorption_energy(
            optimised_energy, clean_energy, molecule_energy, stoichiometry
        )

        write_formatted_results(
            base_dir / f"results_{environment}.txt",
            clean_energy,
            molecule_energy,
            bond_length,
            heights,
            energies,
            adsorption_energies,
            optimal_height,
            optimised_structure,
            optimised_energy,
            optimised_ads_energy,
            site_type,
            environment,
            constrained,
            centre_adsorbate,
            adsorbate_atom,
            molecule_name,
            use_external,
        )

        print(f"\nAnalysis in {environment} environment complete. Results saved to results_{environment}.txt")
        write(
            base_dir / f"optimised_structure_{environment}.xyz",
            optimised_structure,
            format="extxyz",
        )

    print("\nAll calculations complete!")


if __name__ == "__main__":
    main(
        "/mnt/lustre/a2fs-nvme/work/e89/e89/ba3g18/QE/Lithium/Adsorption/Oxygen/Neutral/Solvent/3_FOLD/311.pwo",
        molecule_name="O2",
        adsorbate_atom="O",
        site_type="3fold",
        environments=["solvent"],
        constrained=True,
        centre_adsorbate=True,
        surrogate_metal="Li",
        stoichiometry=0.5,
        use_external=True,
    )