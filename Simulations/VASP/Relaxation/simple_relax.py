import os
import numpy as np
import psutil
from ase.io import read
from ase.build import bulk
from ase.calculators.vasp import Vasp

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

os.environ['ASE_VASP_COMMAND'] = "srun --mpi=pmix /iridisfs/home/ba3g18/Repos/VASP/vasp.6.3.2/bin/vasp_std "
os.environ['VASP_PP_PATH'] = "/home/ba3g18/Repos/Pseudopotentials/POTPAW_VASP"

calc_params = {
    "ENCUT": 500.0, "SIGMA": 0.10, "EDIFF": 1.0e-6, "EDIFFG": -1.0e-2,
    "ALGO": "fast", "PREC": "accurate", "IDIPOL" : 3, "LDIPOL" : True,
    "IBRION": 2, "ISIF": 2, "ISMEAR": 1, "LORBIT": 11, "NELM": 100,
    "NELMIN": 4, "NSW": 200, "IVDW": 0, "LASPH": True, "LCHARG": False, "LWAVE": False, "LREAL": False,
}

kpts = [18, 18, 1]
calc_params.update(simple_vasp_parallel(calc_params, kpts))

li = read("/scratch/ba3g18/VASP/Lithium/Convergence/[100]/Surface_slab_24/OUTCAR")
calc = Vasp(
    xc='PBE',
    kpts=kpts,
    **calc_params,
    directory='Li_DIPOL_TEST'
)
li.calc = calc
energy = li.get_potential_energy()
print(f"Final energy: {energy} eV")