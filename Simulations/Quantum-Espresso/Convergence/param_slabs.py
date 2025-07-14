import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from ase.build import bulk
from ase.atoms import Atoms
from ase.io.espresso import write_fortran_namelist
from ase.calculators.espresso import Espresso, EspressoProfile
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator


ELEMENT = "Li"
USE_ENVIRON = True
MILLER_INDEX = [1, 0, 0]
BASE_CMD = "/home/ba3g18/Repos/q-e/bin/pw.x"
PSEUDOPOTENTIALS = {"Li": "li_pbe_v1.4.uspp.F.UPF"}
PSEUDO_DIR = "/home/ba3g18/Repos/Pseudopotentials/SSSP_1.3.0_PBE_efficiency"


def create_espresso_profile(use_environ: bool = False) -> EspressoProfile:
    """Create an EspressoProfile with the correct command string."""
    command = f"srun --mpi=pmix {BASE_CMD}"
    if use_environ:
        command += " --environ"
    return EspressoProfile(command=command, pseudo_dir=PSEUDO_DIR)


def get_inputs(ecutwfc: float = 40, ecutrho: float = 480) -> Dict[str, Any]:
    """Generate input parameters for Quantum ESPRESSO calculation."""
    return {
        "control": {
            "calculation": "scf",
            "disk_io": "none",
        },
        "system": {
            "ibrav": 0,
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "occupations": "smearing",
            "degauss": 0.01,
            "smearing": "cold",
            "input_dft": "pbe",
        },
        "electrons": {
            "conv_thr": 1.0e-6,
            "mixing_beta": 0.2,
            "mixing_mode": "local-TF",
            "startingwfc": "atomic+random",
            "diagonalization": "rmm-davidson",

        },
    }


def get_environ_parameters() -> Dict[str, Any]:
    """Get environ calculation parameters."""
    return {
        "environ": {
            "verbose": 0,
            "cion(1)": 1.0,
            "cion(2)": 1.0,
            "zion(1)": 1.0,
            "zion(2)": -1.0,
            "cionmax": 4.0,
            "system_dim": 2,
            "system_axis": 3,
            "environ_thr": 0.1,
            "env_pressure": 0.0,
            "temperature": 298.15,
            "environ_type": "input",
            "env_electrostatic": True,
            "env_electrolyte_ntyp": 2,
            "env_surface_tension": 37.3,
            "electrolyte_entropy": "full",
            "env_static_permittivity": 89.9,
            "electrolyte_linearized": False,
        },
        "boundary": {
            "alpha": 1.22,
            "radius_mode": "bondi",
            "solvent_mode": "ionic",
        },
        "electrostatic": {
            "pbc_dim": 2,
            "pbc_axis": 3,
            "tol": 1.0e-15,
            "inner_tol": 1.0e-18,
            "pbc_correction": "parabolic",
        },
    }


def generate_slab(element: str, miller_index: List[int], bulk_a: float) -> Atoms:
    """Generate a slab structure for the given element and Miller index."""
    initial_bulk = bulk(element, "bcc", cubic=True, a=bulk_a)
    structure = AseAtomsAdaptor.get_structure(initial_bulk)

    slabgen = SlabGenerator(
        structure,
        miller_index=miller_index,
        min_slab_size=9.0,
        min_vacuum_size=15.0,
        lll_reduce=False,
        center_slab=True,
        primitive=True,
        max_normal_search=5,
        in_unit_planes=True,
    )

    slabs = slabgen.get_slabs()
    slab = AseAtomsAdaptor.get_atoms(slabs[0].get_orthogonal_c_slab()) * (2, 2, 1)
    slab.center(vacuum=15.0, axis=2)
    slab.pbc = [True, True, False]

    return slab


def run_calc(
    atoms: Atoms,
    inputs: Dict[str, Any],
    profile: EspressoProfile,
    kpts: List[int],
    directory: Path,
    use_environ: bool = False,
) -> float:
    """Run a calculation with Quantum ESPRESSO."""
    calc = Espresso(
        input_data=inputs,
        pseudopotentials=PSEUDOPOTENTIALS,
        profile=profile,
        kpts=kpts,
        directory=directory,
    )
    atoms.calc = calc

    if use_environ:
        with (directory / "environ.in").open("w") as f:
            write_fortran_namelist(f, get_environ_parameters())

    return atoms.get_potential_energy()


def convergence_test(
    atoms: Atoms,
    param_ranges: Dict[str, List[float]],
    profile: EspressoProfile,
    directory: Path,
) -> Dict[str, List[float]]:
    """Run convergence tests for various parameters."""
    energies = {}

    for param_name, param_range in param_ranges.items():
        energies[param_name] = []
        param_dir = directory / param_name
        param_dir.mkdir(exist_ok=True)

        for param in param_range:
            if param_name == "kpoints":
                kpts = [param, param, 1]
                run_dir = param_dir / f"[{param},{param},1]"
                inputs = get_inputs()
            elif param_name == "ecutwfc":
                kpts = [8, 8, 1]
                run_dir = param_dir / f"{param}"
                inputs = get_inputs(ecutwfc=param)
            else:  # ecutrho
                kpts = [8, 8, 1]
                run_dir = param_dir / f"{param}"
                inputs = get_inputs(ecutrho=param)

            run_dir.mkdir(exist_ok=True)

            energy = run_calc(atoms, inputs, profile, kpts, run_dir, USE_ENVIRON)
            energies[param_name].append(energy)
            print(f"{param_name}: {param}, Energy: {energy:.6f} eV")

    return energies


def plot_convergence(
    param_ranges: Dict[str, List[float]],
    energies: Dict[str, List[float]],
    directory: Path,
):
    """Plot convergence results for all parameters."""
    env_type = "with_environ" if USE_ENVIRON else "vacuum"

    fig, axs = plt.subplots(1, len(param_ranges), figsize=(6 * len(param_ranges), 6))

    if len(param_ranges) == 1:
        axs = [axs]

    for ax, (param, values) in zip(axs, param_ranges.items()):
        ax.plot(values, energies[param], "o-")
        ax.set_xlabel(param)
        ax.set_ylabel("Total Energy (eV)")
        ax.set_title(f"Slab Convergence: {param}")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.6f"))
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(directory / f"slab_{env_type}_convergence.png")
    plt.close()


def find_optimal_params(
    param_ranges: Dict[str, List[float]], energies: Dict[str, List[float]]
) -> Dict[str, float]:
    """Find optimal parameters based on energy convergence."""
    return {
        "kpoints": param_ranges["kpoints"][
            np.argmin(np.abs(np.diff(energies["kpoints"])) < 1e-8) + 1
        ],
        "ecutwfc": param_ranges["ecutwfc"][
            np.argmin(np.abs(np.diff(energies["ecutwfc"])) < 1e-8) + 1
        ],
        "ecutrho": param_ranges["ecutrho"][
            np.argmin(np.abs(np.diff(energies["ecutrho"])) < 1e-6) + 1
        ],
    }


def optimize_lattice(profile: EspressoProfile, directory: Path) -> Tuple[float, float]:
    """Optimize lattice constant for slab structure."""
    lattice_dir = directory / "lattice_opt"
    lattice_dir.mkdir(exist_ok=True)

    lattice_constants = np.arange(3.25, 3.75, 0.005)
    energies = []

    for a in lattice_constants:
        run_dir = lattice_dir / f"a_{a:.4f}"
        run_dir.mkdir(exist_ok=True)

        # Generate new slab with this lattice constant
        slab = generate_slab(ELEMENT, MILLER_INDEX, a)
        energy = run_calc(slab, get_inputs(), profile, [8, 8, 1], run_dir, USE_ENVIRON)
        energies.append(energy)
        print(f"a = {a:.4f} Å, E = {energy:.6f} eV")

    optimal_idx = np.argmin(energies)

    plt.figure(figsize=(10, 6))
    plt.plot(lattice_constants, energies, "-o")
    plt.xlabel("Lattice Constant (Å)")
    plt.ylabel("Energy (eV)")
    plt.grid(True)
    plt.savefig(directory / "slab_lattice_optimization.png")
    plt.close()

    return lattice_constants[optimal_idx], energies[optimal_idx]


def main():
    """Main function to run slab convergence tests."""
    base_dir = Path("Li_Slab_Convergence")
    base_dir.mkdir(exist_ok=True)

    profile = create_espresso_profile(use_environ=USE_ENVIRON)
    param_ranges = {
        "kpoints": list(range(2, 20, 1)),
        "ecutwfc": list(range(20, 120, 10)),
        "ecutrho": list(range(200, 1200, 100)),
    }

    # First find optimal lattice constant
    print("\n=== Starting Lattice Constant Optimization ===")
    a_opt, e_min = optimize_lattice(profile, base_dir)
    print(f"\nOptimal lattice constant: {a_opt:.4f} Å")
    print(f"Minimum energy: {e_min:.6f} eV")
    slab = generate_slab(
        element=ELEMENT,
        miller_index=MILLER_INDEX,
        bulk_a=a_opt,
    )

    # Run convergence tests
    print(
        f"\n=== Starting Slab Convergence Tests ({'with ENVIRON' if USE_ENVIRON else 'vacuum'}) ==="
    )
    energies = convergence_test(slab, param_ranges, profile, base_dir)
    plot_convergence(param_ranges, energies, base_dir)

    # Find optimal parameters
    optimal_params = find_optimal_params(param_ranges, energies)
    print("\nOptimal parameters for slab:")
    for param, value in optimal_params.items():
        print(f"{param}: {value}")

    # Save optimal parameters
    with open(base_dir / "optimal_params.txt", "w") as f:
        f.write(
            f"Optimal parameters for slab calculation ({'with ENVIRON' if USE_ENVIRON else 'vacuum'}):\n"
        )
        f.write(f"Lattice constant: {a_opt:.4f} Å\n")
        for param, value in optimal_params.items():
            f.write(f"{param}: {value}\n")


if __name__ == "__main__":
    main()
