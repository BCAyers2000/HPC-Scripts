import os
import shutil
import psutil
import numpy as np
from pathlib import Path
from ase.build import bulk
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from ase.calculators.vasp import Vasp


os.environ['ASE_VASP_COMMAND'] = "srun --mpi=pmix /iridisfs/home/ba3g18/Repos/VASP/vasp.6.3.2/bin/vasp_std"
os.environ['VASP_PP_PATH'] = "/home/ba3g18/Repos/Pseudopotentials/POTPAW_VASP"

def simple_vasp_parallel(calc_params, kpts=None):
    params = calc_params.copy()
    ncores = int(os.environ.get('SLURM_NTASKS', psutil.cpu_count(logical=False) or 1))
    for ncore in range(int(np.sqrt(ncores)), 0, -1):
        if ncores % ncore == 0:
            params["ncore"] = ncore
            break
    if kpts and "kpar" not in params:
        nkpts = np.prod(kpts)
        if nkpts >= 4:
            for kpar in range(2, min(nkpts, ncores//2) + 1):
                if ncores % kpar == 0:
                    params["kpar"] = kpar
                    break
    return params

def get_base_calc_params(scf_only=True) -> Dict[str, Any]:
    params = {
        "xc": "PBE",
        "encut": 520.0,
        "sigma": 0.10,
        "ismear": 0,
        "ediff": 1.0e-6,
        "symprec": 1.0e-8,
        "algo": "Fast",
        "prec": "Accurate",
        "ibrion": -1 if scf_only else 2,
        "isif": 2,
        "isym": 0,
        "lorbit": 11,
        "nelm": 100,
        "nelmin": 4,
        "lasph": True,
        "lcharg": False,
        "lwave": False,
        "lreal": False,
    }
    
    if not scf_only:
        params["ediffg"] = -1.0e-2
        params["nsw"] = 100
    
    return params

def run_calc(atoms, calc_params: Dict[str, Any], directory: str) -> float:
    if Path(directory).exists():
        shutil.rmtree(directory)
    Path(directory).mkdir(parents=True, exist_ok=True)
    kpts = calc_params.get("kpts")
    calc_params = simple_vasp_parallel(calc_params, kpts)
    calc = Vasp(directory=directory, **calc_params, setups="recommended")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    return energy

def convergence_test(
    atoms, param_ranges: Dict[str, List[float]]
) -> Dict[str, List[float]]:
    energies = {}
    
    kpt_dir = Path("kpoints")
    kpt_dir.mkdir(exist_ok=True)
    
    kpt_range = param_ranges["kpoints"]
    energies["kpoints"] = []
    for kpt in kpt_range:
        calc_params = get_base_calc_params(scf_only=True)
        calc_params["kpts"] = [kpt, kpt, kpt]
        
        kpt_subdir = kpt_dir / f"[{kpt},{kpt},{kpt}]"
        energy = run_calc(atoms, calc_params, str(kpt_subdir))
        energies["kpoints"].append(energy)
        print(f"kpoints: [{kpt},{kpt},{kpt}], Energy: {energy:.6f} eV")
    
    encut_dir = Path("ecutwfc")
    encut_dir.mkdir(exist_ok=True)
    
    encut_range = param_ranges["encut"]
    energies["encut"] = []
    for encut in encut_range:
        calc_params = get_base_calc_params(scf_only=True)
        calc_params["encut"] = encut
        calc_params["kpts"] = [30, 30, 30]
        
        encut_subdir = encut_dir / f"{encut}"
        energy = run_calc(atoms, calc_params, str(encut_subdir))
        energies["encut"].append(energy)
        print(f"encut: {encut}, Energy: {energy:.6f} eV")
    
    return energies

def plot_convergence(
    param_ranges: Dict[str, List[float]], energies: Dict[str, List[float]]
):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    ax = axs[0]
    ax.plot(param_ranges["kpoints"], energies["kpoints"], "o-")
    ax.set_xlabel("K-points mesh (n*n*n)")
    ax.set_ylabel("Total Energy (eV)")
    ax.set_title("K-points Convergence")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.6f"))
    ax.grid(True)
    
    ax = axs[1]
    ax.plot(param_ranges["encut"], energies["encut"], "o-")
    ax.set_xlabel("ENCUT (eV)")
    ax.set_ylabel("Total Energy (eV)")
    ax.set_title("Plane-wave Cutoff Convergence")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.6f"))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("vasp_convergence.png", dpi=300)
    plt.close()

def find_optimal_params(
    param_ranges: Dict[str, List[float]], energies: Dict[str, List[float]]
) -> Dict[str, float]:
    optimal = {}
    
    kpt_diffs = np.abs(np.diff(energies["kpoints"]))
    for i, diff in enumerate(kpt_diffs):
        if diff < 1e-4:
            optimal["kpoints"] = param_ranges["kpoints"][i+1]
            break
    else:
        optimal["kpoints"] = param_ranges["kpoints"][-1]
    
    encut_diffs = np.abs(np.diff(energies["encut"]))
    for i, diff in enumerate(encut_diffs):
        if diff < 1e-4:
            optimal["encut"] = param_ranges["encut"][i+1]
            break
    else:
        optimal["encut"] = param_ranges["encut"][-1]
    
    return optimal

def optimise_lattice(atoms, optimal_params: Dict[str, float]) -> tuple[float, float]:
    lattice_dir = Path("lattice_opt")
    lattice_dir.mkdir(exist_ok=True)
    
    lattice_constants = np.arange(3.95, 4.25, 0.01)
    energies = []
    
    for a in lattice_constants:
        Li = bulk("Li", "bcc", a=a, cubic=True)
        
        calc_params = get_base_calc_params(scf_only=True)
        calc_params["encut"] = optimal_params["encut"]
        calc_params["kpts"] = [optimal_params["kpoints"]] * 3
        
        a_subdir = lattice_dir / f"a_{a:.4f}"
        energy = run_calc(Li, calc_params, str(a_subdir))
        energies.append(energy)
        print(f"a = {a:.4f} Å, E = {energy:.6f} eV")
    
    optimal_idx = np.argmin(energies)
    
    plt.figure(figsize=(10, 6))
    plt.plot(lattice_constants, energies, "-o")
    plt.xlabel("Lattice Constant (Å)")
    plt.ylabel("Energy (eV)")
    plt.grid(True)
    plt.savefig("vasp_lattice_optimisation.png", dpi=300)
    plt.close()
    
    return lattice_constants[optimal_idx], energies[optimal_idx]

def final_calculation(a_opt: float, optimal_params: Dict[str, float]) -> float:
    Li = bulk("Li", "bcc", a=a_opt, cubic=True)
    
    calc_params = get_base_calc_params(scf_only=False)
    calc_params["encut"] = optimal_params["encut"]
    calc_params["kpts"] = [optimal_params["kpoints"]] * 3
    calc_params["isif"] = 3
    calc_params["nsw"] = 100
    
    directory = "final_relax"
    energy = run_calc(Li, calc_params, directory)
    
    return energy

def main():
    atoms = bulk("Li", "bcc", a=3.44, cubic=True)
    
    param_ranges = {
        "kpoints": list(range(2, 25, 2)),
        "encut": list(range(100, 725, 25)),
    }
    
    print("Running convergence tests...")
    energies = convergence_test(atoms, param_ranges)
    
    plot_convergence(param_ranges, energies)
    
    optimal_params = find_optimal_params(param_ranges, energies)
    print("\nOptimal parameters:")
    for param, value in optimal_params.items():
        print(f"{param}: {value}")
    
    print("\nOptimizing lattice constant...")
    a_opt, e_min = optimise_lattice(atoms, optimal_params)
    print(f"\nOptimal lattice constant: {a_opt:.4f} Å")
    print(f"Minimum energy: {e_min:.6f} eV")
    
    print("\nRunning final calculation with optimised parameters...")
    final_energy = final_calculation(a_opt, optimal_params)
    print(f"Final energy: {final_energy:.6f} eV")

if __name__ == "__main__":
    main()