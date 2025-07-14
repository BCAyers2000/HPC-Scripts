import os
import numpy as np
from typing import Dict, Any
from ase.io import write, read
from ase.optimize import GoodOldQuasiNewton
from ase.calculators.espresso import Espresso, EspressoProfile


# Constants
BASE_CMD = "/home/ba3g18/Repos/q-e/bin/pw.x"
PSEUDO_DIR = "/home/ba3g18/Repos/Pseudopotentials/SSSP_1.3.0_PBE_efficiency"
PSEUDOPOTENTIALS = {"Pt": "pt_pbe_v1.4.uspp.F.UPF", "O": "O.pbe-n-kjpaw_psl.0.1.UPF"}

def get_qe_parameters(use_external: bool = False) -> Dict[str, Any]:
    """Get Quantum ESPRESSO calculation parameters."""
    params = {
        "control": {
            "calculation": "scf" if use_external else "relax",
            "verbosity": "high",
            "nstep": 200,
            "prefix": "Pt-O",
            "outdir": "./Pt/",
            "etot_conv_thr": 1.0e-4,
            "forc_conv_thr": 1.0e-3,
            "pseudo_dir": PSEUDO_DIR,
        },
        "system": {
            "ibrav": 0,
            "ecutwfc": 60.0,
            "ecutrho": 600.0,
            "degauss": 0.01,
            "smearing": "mv",
            "input_dft": "rpbe",
            "occupations": "smearing",
        },
        "electrons": {
            "conv_thr": 1.0e-10 if use_external else 1.0e-6,
            "mixing_beta": 0.20,
            "electron_maxstep": 250,
            "mixing_mode": "local-TF",
            "diagonalization": "david",
            "startingwfc": "atomic+random",
        },
        "ions": {"ion_dynamics": "bfgs", "upscale": 1e6, "bfgs_ndim": 1},
    }
    return params

def run_calculation(
    input_file: str, use_external: bool = False, directory: str = "./Pt_O/"
) -> tuple[float, Any]:
    """
    Run Quantum ESPRESSO calculation with optional external optimiser.

    Args:
        input_file: Path to input XYZ file
        use_external: Whether to use external optimiser (GoodOldQuasiNewton)
        directory: Output directory for calculation

    Returns:
        Tuple of (potential energy, relaxed atoms)
    """
    os.makedirs(directory, exist_ok=True)
    nano = read(input_file)

    magmoms = np.zeros(len(nano))
    magmoms[-1] = 1.0
    nano.set_initial_magnetic_moments(magmoms)

    nano_profile = EspressoProfile(
        command=f"srun --mpi=pmix {BASE_CMD}", pseudo_dir=PSEUDO_DIR
    )
    nano_params = get_qe_parameters(use_external=use_external)
    calc = Espresso(
        input_data=nano_params,
        pseudopotentials=PSEUDOPOTENTIALS,
        profile=nano_profile,
        directory=directory,
        kpts=None,
    )
    nano.calc = calc

    if use_external:
        opt = GoodOldQuasiNewton(
            nano, trajectory=f"{directory}/QE.traj", logfile=f"{directory}/QE.log"
        )
        opt.run(fmax=0.05, steps=1000)
        energy = nano.get_potential_energy()
        relaxed_nano = nano.copy()
    else:
        energy = nano.get_potential_energy()
        try:
            relaxed_nano = read(f"{directory}/espresso.pwo")
        except Exception as e:
            raise RuntimeError(f"Failed to read relaxed structure: {e}")
    write("relaxed_nano.xyz", relaxed_nano, format="extxyz")

    return energy, relaxed_nano

if __name__ == "__main__":
    input_file = "/scratch/ba3g18/QE/Platinum/Pt55_O/pt_o.xyz"
    energy, relaxed_structure = run_calculation(input_file, use_external=True)