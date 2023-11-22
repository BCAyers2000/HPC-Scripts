from pathlib import Path
from ase.io import write
from ase import Atoms
from ase.io.onetep import get_onetep_keywords
from ase.optimize.sciopt import SciPyFminBFGS
from ase.build.surface import bcc100, add_adsorbate
from ase.calculators.onetep import Onetep

main_dir = Path("/home/mmm1182/Scratch/New-Works/Scipy-BFGS") 
keywords = get_onetep_keywords('/home/mmm1182/Scratch/New-Works/Scipy-BFGS/Keywords.dat')
s1 = bcc100('Li', (3, 3, 10), a = 3.44, vacuum=25)
s1.set_tags([0]*len(s1))
molecule = Atoms('2N', positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.1)])
molecule.center()
add_adsorbate(s1, molecule, 1.8, 'bridge')
local_dir = main_dir/f"Scipy_0_volts"
local_dir.mkdir(exist_ok=True)
filename = local_dir/f'Li.dat'
write(filename, s1, format='onetep-in', keywords=keywords)
calc = Onetep(
    label = filename.stem,
    edft = True,
    ngwf_radius = 9.5,
    directory = local_dir,
    ngwf_count = {'Li': -1, 'N': -1},
    pseudo_path='/home/mmm1182/Scratch/New-Works/Scipy-BFGS',
    pseudo_suffix='.usp',
    keywords=keywords,
    autorestart=True,
    append=True)
s1.set_calculator(calc)
opt = SciPyFminBFGS(s1, trajectory=str(local_dir / 'Li.traj'), force_consistent=True)
opt.run(fmax=0.05)
