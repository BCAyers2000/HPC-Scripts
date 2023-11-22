from ase.build import molecule
from os import environ
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import read
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry
from ase.build import bulk
import re
from shutil import copyfile
import matplotlib.pyplot as plt

pseudo_path = Path(environ['ESPRESSO_PSEUDO'])
print(f'pseudo_path: {pseudo_path}')
label = 'pwscf'
pbc = True
outdir = Path('pw.dir')
atoms = bulk('Li', 'bcc', cubic=True)
original_cell = atoms.get_cell()
atoms.set_pbc(pbc)
kpts = np.floor(2 * np.pi / (atoms.cell.lengths() * 0.25)).astype(int)
kpts[kpts < 1] = 1
if np.all(kpts == 1):
    kpts = None

input_data = {
'control': {
'calculation': 'vc-relax',
'verbosity': 'high',
'restart_mode': 'from_scratch',
'nstep': 999,
'tstress': False,
'tprnfor': True,
'outdir': str(outdir),
'prefix': label,
'max_seconds': 86100,
'etot_conv_thr': 1.0e-8,
'forc_conv_thr': 1.0e-7,
'disk_io': 'low',
'pseudo_dir': str(pseudo_path),
'trism': False,
},
'system': {
'ibrav': 0,
'tot_charge': 0.0,
'tot_magnetization': -10000,
'ecutwfc': 40.0,
'ecutrho': 40*8,
'occupations': 'smearing',
'degauss': 0.01,
'smearing': 'cold',
'input_dft': 'pbe',
'nspin': 2,
'assume_isolated': 'none',
'esm_bc': 'pbc',
'vdw_corr': 'none',
},
'electrons': {
'electron_maxstep': 999,
'scf_must_converge': True,
'conv_thr': 1.0e-12,
'mixing_mode': 'plain',
'mixing_beta': 0.8,
'mixing_ndim': 8,
'diagonalization': 'david',
'diago_david_ndim': 2,
'diago_rmm_ndim': 4,
'diago_rmm_conv': False,
'diago_full_acc': False,
},
'ions': {
'ion_dynamics': 'bfgs',
'upscale': 100,
'bfgs_ndim': 6,
},
'cell': {
'cell_dynamics': 'bfgs',
'press_conv_thr': 0.1,
'cell_dofree': 'all',
}
}

espresso_command = [
'srun',
'--cpu-freq=2250000',
'--hint=nomultithread',
'--distribution=block:block',
'/work/e89/e89/ba3g18/Repos/q-e-qe-7.2/bin/pw.x'
]

u_elements = np.unique(atoms.get_chemical_symbols())

pseudopotentials = {}
pot_files = pseudo_path.glob('*')

for pot_file in pot_files:
    el_name = re.split(r'[._\s]+', pot_file.name)[0].title()
    if el_name in u_elements:
        pseudopotentials[el_name] = pot_file.name

calc_profile = EspressoProfile(
argv=espresso_command
)

calc = Espresso(
pseudo_path=pseudo_path,
pseudopotentials=pseudopotentials,
input_data=input_data,
kpts=kpts,
profile=calc_profile
)

atoms.calc = calc
for x in np.arange(0.75, 1.30, 0.05):
    x = np.eye(3) * x
    atoms.set_cell(x @ original_cell, scale_atoms=True)
    atoms.get_potential_energy()
    copyfile('espresso.pwo', f'li_{x[0, 0]:.2f}.pwo')