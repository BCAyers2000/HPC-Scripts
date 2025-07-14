import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from ase.build import bulk
from ase.calculators.espresso import Espresso, EspressoProfile

PSEUDO_DIR = "/home/ba3g18/Repos/Pseudopotentials/SSSP_1.3.0_PBE_efficiency"
COMMAND = "srun --mpi=pmix /home/ba3g18/Repos/q-e/bin/pw.x"

def get_inputs(ecutwfc: float = 40, ecutrho: float = 400) -> Dict[str, Any]:
    return {
        "control": {
            "calculation": "scf",
            "verbosity": "high",
            "prefix": "Li",
            "nstep": 999,
            "tstress": False,
            "tprnfor": True,
            "disk_io": "low",
            "outdir": "./Lithium/",
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
            "conv_thr": 1.0e-20,
            "mixing_beta": 0.7,
        },
    }


def run_calc(
    atoms, inputs: Dict[str, Any], profile: EspressoProfile, kpts: List[int]
) -> float:
    calc = Espresso(
        input_data=inputs,
        pseudopotentials={"Li": "li_pbe_v1.4.uspp.F.UPF"},
        profile=profile,
        kpts=kpts,
    )
    atoms.calc = calc
    return atoms.get_potential_energy()


def convergence_test(
    atoms, param_ranges: Dict[str, List[float]], profile: EspressoProfile
) -> Dict[str, List[float]]:
    energies = {}
    for param_name, param_range in param_ranges.items():
        energies[param_name] = []
        for param in param_range:
            if param_name == "kpoints":
                kpts = [param] * 3
                inputs = get_inputs()
            elif param_name == "ecutwfc":
                kpts = [30] * 3
                inputs = get_inputs(ecutwfc=param)
            else:  # ecutrho
                kpts = [30] * 3
                inputs = get_inputs(ecutrho=param)

            energy = run_calc(atoms, inputs, profile, kpts)
            energies[param_name].append(energy)
            print(f"{param_name}: {param}, Energy: {energy:.6f} eV")

    return energies


def plot_convergence(
    param_ranges: Dict[str, List[float]], energies: Dict[str, List[float]]
):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (param, values) in zip(axs, param_ranges.items()):
        ax.plot(values, energies[param], "o-")
        ax.set_xlabel(param)
        ax.set_ylabel("Total Energy (eV)")
        ax.set_title(f"Convergence: {param}")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.6f"))
        ax.grid(True)
    plt.tight_layout()
    plt.savefig("convergence.png")
    plt.close()


def find_optimal_params(
    param_ranges: Dict[str, List[float]], energies: Dict[str, List[float]]
) -> Dict[str, float]:
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


def optimize_lattice(atoms, profile: EspressoProfile) -> tuple[float, float]:
    lattice_constants = np.arange(3.25, 3.75, 0.05)
    energies = []

    for a in lattice_constants:
        atoms = bulk("Li", "bcc", a=a, cubic=True)
        energy = run_calc(atoms, get_inputs(), profile, [30] * 3)
        energies.append(energy)
        print(f"a = {a:.4f} Å, E = {energy:.6f} eV")

    optimal_idx = np.argmin(energies)

    plt.figure(figsize=(10, 6))
    plt.plot(lattice_constants, energies, "-o")
    plt.xlabel("Lattice Constant (Å)")
    plt.ylabel("Energy (eV)")
    plt.grid(True)
    plt.savefig("lattice_optimisation.png")
    plt.close()

    return lattice_constants[optimal_idx], energies[optimal_idx]


def main():
    Path("Lithium").mkdir(exist_ok=True)
    profile = EspressoProfile(command=COMMAND, pseudo_dir=PSEUDO_DIR)
    atoms = bulk("Li", "bcc", a=3.44, cubic=True)

    param_ranges = {
        "kpoints": list(range(2, 30, 1)),
        "ecutwfc": list(range(20, 120, 10)),
        "ecutrho": list(range(200, 1200, 100)),
    }

    energies = convergence_test(atoms, param_ranges, profile)
    plot_convergence(param_ranges, energies)

    optimal_params = find_optimal_params(param_ranges, energies)
    print("\nOptimal parameters:")
    for param, value in optimal_params.items():
        print(f"{param}: {value}")

    a_opt, e_min = optimize_lattice(atoms, profile)
    print(f"\nOptimal lattice constant: {a_opt:.4f} Å")
    print(f"Minimum energy: {e_min:.6f} eV")


if __name__ == "__main__":
    main()
