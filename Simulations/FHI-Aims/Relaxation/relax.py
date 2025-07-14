import numpy as np
from ase.io import read, write
from pathlib import Path
from ase.build import bulk
from ase.atoms import Atoms
from typing import Dict, Any, Tuple
from ase.calculators.aims import Aims, AimsProfile
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator

# Constants
KSPACING = 0.1
MILLER_INDEX = (1, 0, 0)
MIN_SLAB_SIZE = 15
MIN_VACUUM_SIZE = 20

PARAMETERS = {
    "xc": "pbe",
    "relativistic": "atomic_zora scalar",
    "occupation_type": "gaussian 0.1",
    "sc_accuracy_stress": 1e-3,
    "sc_accuracy_rho": 1e-6,
    "sc_accuracy_etot": 1e-6,
    "sc_iter_limit": 300,
    "mixer": "pulay",
    "n_max_pulay": 14,
    "charge_mix_param": 0.02,
    "relax_geometry": "bfgs 1e-4",
    "relax_unit_cell": "full",
    "output_level": "MD_light",
}


def generate_kpts(atoms: Atoms, periodic_3d: bool = True) -> list:
    cell_lengths = atoms.cell.lengths()
    kpts = np.ceil(2 * np.pi / (cell_lengths * KSPACING)).astype(int)
    if not periodic_3d:
        kpts[2] = 1
    return kpts.tolist()


def create_aims_profile() -> AimsProfile:
    return AimsProfile(
        command="srun --mpi=pmix /home/ba3g18/Repos/FHIaims/build/aims.250131.scalapack.mpi.x",
        default_species_directory="/home/ba3g18/Repos/FHIaims/species_defaults/defaults_2020/intermediate/",
    )


def run_calculation(
    atoms: Atoms,
    parameters: Dict[str, Any],
    profile: AimsProfile,
    directory: Path,
    periodic_3d: bool = True,
) -> Tuple[float, Atoms]:
    parameters["k_grid"] = generate_kpts(atoms, periodic_3d=periodic_3d)
    calc = Aims(profile=profile, directory=directory, **parameters)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    relaxed_atoms = read(f"{directory}/aims.out")
    return energy, relaxed_atoms


def prepare_structures(initial_atoms: Atoms) -> Tuple[Atoms, Atoms]:
    structure = AseAtomsAdaptor.get_structure(initial_atoms)
    slabgen = SlabGenerator(
        structure,
        miller_index=MILLER_INDEX,
        min_slab_size=MIN_SLAB_SIZE,
        min_vacuum_size=MIN_VACUUM_SIZE,
        lll_reduce=False,
        center_slab=True,
        primitive=True,
        max_normal_search=5,
        in_unit_planes=True,
    )

    slabs = slabgen.get_slabs()
    ouc = AseAtomsAdaptor.get_atoms(slabs[0].oriented_unit_cell)
    slab = AseAtomsAdaptor.get_atoms(slabs[0].get_orthogonal_c_slab()) * (2, 2, 1)
    slab.center(vacuum=20, axis=2)
    return ouc, slab


def main() -> None:
    profile = create_aims_profile()
    initial_bulk = bulk("Li", "bcc", cubic=True, a=3.44)
    ouc, slab = prepare_structures(initial_bulk)

    ouc_dir = Path("OUC")
    ouc_dir.mkdir(exist_ok=True)
    ouc_energy, relaxed_ouc = run_calculation(
        ouc, PARAMETERS, profile, ouc_dir, periodic_3d=True
    )

    slab_dir = Path("Surface")
    slab_dir.mkdir(exist_ok=True)
    slab_parameters = PARAMETERS.copy()
    slab_parameters["relax_unit_cell"] = "none"
    slab_energy, relaxed_slab = run_calculation(
        slab, slab_parameters, profile, slab_dir, periodic_3d=False
    )
    write("relaxed_slab.xyz", relaxed_slab, format="extxyz")

    area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    surface_energy = (
        1 / 2 / area * (slab_energy - ((len(slab) / len(ouc)) * ouc_energy))
    )
    print(f"Surface energy: {surface_energy * 16.02:.8f} J/m^2")


if __name__ == "__main__":
    main()
