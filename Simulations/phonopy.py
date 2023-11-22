import re
from os import environ
from pathlib import Path

import csv
import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import read
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry
from ase.build import bulk
import re
from ase.build.surface import bcc100

label = 'pwscf'
supercell = np.diag([4,4,1]) 
atoms = bcc100('Li', (2, 2, 1), vacuum=7.5, orthogonal=True)

input_data = {
    'control': {
        'calculation': 'vc-relax',
        'verbosity': 'high',
        'restart_mode': 'from_scratch',
        'nstep': 999,
        'tstress': False,
        'tprnfor': True,
        'outdir': 'pw.dir',
        'prefix': label,
        'max_seconds': 86100,
        'etot_conv_thr': 1.0e-8,
        'forc_conv_thr': 1.0e-6,
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
        # 'input_dft': 'rpbe',
        'nspin': 1,
        'assume_isolated': 'none',
        'esm_bc': 'pbc',
        'vdw_corr': 'none',
        'ibrav': 8,
    },
    'electrons': {
        'electron_maxstep': 999,
        'scf_must_converge': True,
        'conv_thr': 1.0e-12,
        'mixing_mode': 'local-TF',
        'mixing_beta': 0.4,
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

u_elements = np.unique(atoms.get_chemical_symbols())
pseudo_path = Path(input_data['control']['pseudo_dir']) 
pseudopotentials = {}
pot_files = pseudo_path.glob('*')
for pot_file in pot_files:
    el_name = re.split(r'[._\s]+', pot_file.name)[0].title()
    if el_name in u_elements:
        pseudopotentials[el_name] = pot_file.name


kpts = np.ceil(2 * np.pi / (atoms.cell.lengths() * 0.25)).astype(int)
kpts[kpts < 1] = 1
if np.all(kpts == 1):
    kpts = None


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

# Optimise the structure

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
    vc_energy = atoms.get_potential_energy()

input_data['control']['calculation'] = 'scf'
phonopy_atoms = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                             cell=atoms.get_cell(),
                             scaled_positions=atoms.get_scaled_positions()
                             )

phonons = Phonopy(phonopy_atoms,
                  supercell_matrix=supercell,
                  primitive_matrix=np.eye(3),
                  is_symmetry=True
                  )

symmetry = Symmetry(phonopy_atoms)
pointgroup = symmetry.get_pointgroup()
tol = symmetry.get_symmetry_tolerance()
print('symmetry detected:', pointgroup, 'with tolerance', tol, 'Å')

phonons.generate_displacements(distance=0.005)
displacements = phonons.get_supercells_with_displacements()

forces = []

for i, disp in enumerate(displacements):
    print(f'calculating displacement {i}', flush=True)
    # Create a directory for each displacement
    disp_dir = Path(f'disp_{i}')
    disp_dir.mkdir(exist_ok=True)
    # We build the ASE atoms object back from phonopyAtoms
    atoms = Atoms(disp.get_chemical_symbols(),
                  disp.get_positions(),
                  cell=disp.get_cell())
    # If supercell we have to recalculate the kpoints
    kpts = np.ceil(2 * np.pi / (atoms.cell.lengths() * 0.25)).astype(int)
    kpts[kpts < 1] = 1
    if np.all(kpts == 1):
        kpts = None

    calc = Espresso(
        pseudo_path=pseudo_path,
        kpts=kpts,
        directory=disp_dir,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        profile=calc_profile)
    atoms.calc = calc
    # If the calculation has already been done, we read the forces
    if (disp_dir / 'espresso.pwo').exists():
        try:
            forces.append(read(disp_dir / 'espresso.pwo').get_forces())
        except Exception as e:
            forces.append(atoms.get_forces())
    else:
        forces.append(atoms.get_forces())

phonons.forces = forces
phonons.produce_force_constants()
phonons.save(f'{label}_fc.yaml', settings={'force_constants': True})

phonons.run_mesh([50, 50, 50])

phonons.run_total_dos()
pdos = np.stack(phonons.get_total_DOS())
pdos_reshaped = pdos.T.reshape(-1, 2)
pdos_reshaped[:, 0] *= 33.356
np.savetxt('phonopy_dos.csv', pdos_reshaped, delimiter=',', header='frequency (cm^-1), population')

temperatures = np.arange(0, 1000+1, 10)
temperatures = np.append(temperatures, [273.15, 293.15, 298.15])
temperatures = np.sort(temperatures)

phonons.run_thermal_properties(temperatures=temperatures)

thermo_dict = phonons.get_thermal_properties_dict()
entropy = thermo_dict['entropy']
free_energy = thermo_dict['free_energy']
temperatures = thermo_dict['temperatures']
heat_capacity = thermo_dict['heat_capacity']

freq = phonons.get_frequencies_with_eigenvectors((0, 0, 0))[0]*33.356
Kjmol_to_eV = 0.01036410
jmol_to_eV = Kjmol_to_eV / 1000 

with open(f'{label}_thermo.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['frequencies at Gamma (cm^-1)'] + [f'{f:.8f}' for f in freq])
    csv_writer.writerow([''])
    csv_writer.writerow(['T (K)', 'F_vib (eV)', 'S_vib (eV/K)', 'Cv (eV/K)', 'G_tot (eV)'])
    
    for t, F, S, cv in zip(temperatures, free_energy, entropy, heat_capacity):
        G_tot = vc_energy + F * Kjmol_to_eV - t * S * jmol_to_eV
        csv_writer.writerow([t, F * Kjmol_to_eV, S * jmol_to_eV, cv * jmol_to_eV, G_tot])




