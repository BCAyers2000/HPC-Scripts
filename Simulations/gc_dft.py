import re
import copy
import numpy as np
from ase import Atoms
from os import environ
from ase.io import read
from pathlib import Path
from ase.build.surface import bcc100
from ase.calculators.espresso import Espresso, EspressoProfile

input_data = {
    'control': {
        'calculation': 'relax',
        'verbosity': 'high',
        'restart_mode': 'from_scratch',
        'nstep': 999,
        'tstress': False,
        'tprnfor': True,
        'outdir': 'pw.dir',
        'prefix': 'can-dft',
        'max_seconds': 86100,
        'etot_conv_thr': 1.0e-8,
        'forc_conv_thr': 1.0e-5,
        'disk_io': 'none',
        'pseudo_dir': environ['ESPRESSO_PSEUDO'] ,
        'trism': False,
    },
    'system': {
        'tot_charge': 0.0,
        'tot_magnetization': -10000,
        'ecutwfc': 40.0,
        'ecutrho': 40*8,
        'occupations': 'smearing',
        'degauss': 0.01,
        'smearing': 'cold',
        'nspin': 1,
        'assume_isolated': 'esm',
        'esm_bc': 'bc1',
        'vdw_corr': 'none',
        'ibrav': 0,
    },
    'electrons': {
        'electron_maxstep': 999,
        'scf_must_converge': True,
        'conv_thr': 1.0e-12,
        'mixing_mode': 'local-TF',
        'mixing_beta': 0.15,
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
        'bfgs_ndim': 1,
    },
    'cell': {
        'cell_dynamics': 'bfgs',
        'press_conv_thr': 0.1,
        'cell_dofree': '2Dxy',
    }
}

espresso_command = [
    'srun',
    '--cpu-freq=2250000',
    '--hint=nomultithread',
    '--distribution=block:block',
    '/work/e89/e89/ba3g18/Repos/q-e-qe-7.2/bin/pw.x'
]

calc_profile = EspressoProfile(argv=espresso_command)

def create_directory(directory_name):
    directory = Path(directory_name)
    directory.mkdir(exist_ok=True)
    return directory

def get_element_pseudopotentials(pot_files, u_elements):
    pseudopotentials = {}
    for pot_file in pot_files:
        el_name = re.split(r'[._\s]+', pot_file.name)[0].title()
        if el_name in u_elements:
            pseudopotentials[el_name] = pot_file.name
    return pseudopotentials

results = {}
for layer in range(1,15+1,2):
    atoms = bcc100('Li', (2, 2, layer), vacuum=7.5, orthogonal=True)
    atoms.center(about=0)

    can_dir = create_directory(f'{layer}-thick-can')
    kpts = np.ceil(2 * np.pi / (atoms.cell.lengths() * 0.20)).astype(int)
    kpts[kpts < 1] = 1
    if np.all(kpts == 1):
        kpts = None
    kpts[-1] = 1    

    u_elements = np.unique(atoms.get_chemical_symbols())
    pseudo_path = Path(environ['ESPRESSO_PSEUDO'])
    pseudopotentials_can = get_element_pseudopotentials(pseudo_path.glob('*'), u_elements)
    
    calc = Espresso(
        pseudo_path=pseudo_path,
        kpts=kpts,
        directory=can_dir,
        input_data=input_data,
        pseudopotentials=pseudopotentials_can,
        profile=calc_profile)
    atoms.calc = calc

    run_opt = True
    if Path('espresso.pwo').exists():
        try:
            atoms = read('espresso.pwo')
            check_forces = atoms.get_forces()
            if np.any(np.abs(check_forces) < 1e-3):
                run_opt = False
        except Exception:
            pass
    if run_opt:
        atoms.get_potential_energy()
        Fermi_energy = atoms.calc.get_fermi_level()
        results[layer] = atoms
    
    gc_dir = create_directory(f'{layer}-thick-gc')
    input_data_gc = copy.deepcopy(input_data)
    input_data_gc['control'].update({
        'calculation': 'scf', 'prefix': 'gc_dft'
    })
    input_data_gc['system'].update({
        'lgcscf': True, 'gcscf_mu': Fermi_energy, 'gcscf_conv_thr': 1E-4,
        'gcscf_beta': 0.05, 'assume_isolated': 'esm', 'esm_bc': 'bc3'
    })
    input_data_gc['electrons'].update({
        'diago_rmm_conv': True,
        'diago_full_acc': True
    })
    
    atoms = Atoms(results[layer].get_chemical_symbols(),
                  results[layer].get_positions(),
                  cell=results[layer].get_cell())
    pseudopotentials_gc = pseudopotentials_can
    calc = Espresso(
    pseudo_path=pseudo_path,
    pseudopotentials=pseudopotentials_gc,
    input_data=input_data_gc,
    kpts=kpts,
    directory = gc_dir,
    profile=calc_profile )
    atoms.calc = calc
    atoms.get_potential_energy()
