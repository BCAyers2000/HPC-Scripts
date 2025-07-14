import numpy as np
from ase.io import read
from pathlib import Path
from ase.build import bulk
import matplotlib.pyplot as plt
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator
from ase.calculators.espresso import Espresso, EspressoProfile

KSPACING, MILLER_INDEX = 0.10, [1,0,0]
PSEUDOPOTENTIALS = {"Li": "li_pbe_v1.4.uspp.F.UPF"}


def generate_kpts(atoms, periodic_3d=False):
    cell_lengths = atoms.cell.lengths()
    kpts = np.ceil(2 * np.pi / (cell_lengths * KSPACING)).astype(int)
    return kpts.tolist() if periodic_3d else [kpts[0], kpts[1], 1]


def create_espresso_profile():
    return EspressoProfile(
        command="srun --mpi=pmix /home/ba3g18/Repos/q-e/bin/pw.x",
        pseudo_dir="/home/ba3g18/Repos/Pseudopotentials/SSSP_1.3.0_PBE_efficiency",
    )


def get_input_data(calculation_type="vc-relax"):
    return {
        "control": {
            "calculation": calculation_type,
            "verbosity": "high",
            "prefix": "Li",
            "nstep": 999,
            "tstress": False,
            "tprnfor": True,
            "disk_io": "low",
            "outdir": "./Lithium/",
            "etot_conv_thr": 1.0e-5,
            "forc_conv_thr": 1.0e-5,
        },
        "system": {
            "ibrav": 0,
            "tot_charge": 0.0,
            "ecutwfc": 40.0,
            "ecutrho": 600,
            "occupations": "smearing",
            "degauss": 0.01,
            "smearing": "cold",
            "input_dft": "pbe",
            "nspin": 1,
        },
        "electrons": {
            "electron_maxstep": 999,
            "scf_must_converge": True,
            "conv_thr": 1.0e-12,
            "mixing_mode": "plain",
            "mixing_beta": 0.80,
            "startingwfc": "random",
            "diagonalization": "david",
        },
        "ions": {"ion_dynamics": "bfgs", "upscale": 1e8, "bfgs_ndim": 6},
        "cell": {"press_conv_thr": 0.1, "cell_dofree": "all"},
    }


def run_calculation(atoms, input_data, profile, directory, periodic_3d=True):
    calc = Espresso(
        input_data=input_data,
        pseudopotentials=PSEUDOPOTENTIALS,
        profile=profile,
        directory=directory,
        kpts=generate_kpts(atoms, periodic_3d),
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    relaxed_atoms = read(f"{directory}/espresso.pwo")
    return energy, relaxed_atoms


def prepare_structures(initial_atoms, min_slab_size):
    structure = AseAtomsAdaptor.get_structure(initial_atoms)
    slabgen = SlabGenerator(
        structure,
        miller_index=MILLER_INDEX,
        min_slab_size=min_slab_size,
        min_vacuum_size=15,
        lll_reduce=False,
        center_slab=True,
        primitive=True,
        max_normal_search=5,
        in_unit_planes=True,
    )
    slabs = slabgen.get_slabs()
    ouc = AseAtomsAdaptor.get_atoms(slabs[0].oriented_unit_cell)
    slab = AseAtomsAdaptor.get_atoms(slabs[0])
    return ouc, slab


def calculate_surface_energy(slab_energy, ouc_energy, slab, ouc):
    area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    surface_energy = (
        1 / 2 / area * (slab_energy - ((len(slab) / len(ouc)) * ouc_energy))
    )
    return surface_energy * 16.02


def main():
    initial_atoms = bulk("Li", a=3.435, cubic=True)
    profile = create_espresso_profile()
    ouc_dir = Path("OUC")
    ouc_dir.mkdir(exist_ok=True)

    ouc, _ = prepare_structures(initial_atoms, min_slab_size=3)
    ouc_input_data = get_input_data("relax")
    ouc_energy, relaxed_ouc = run_calculation(
        ouc, ouc_input_data, profile, ouc_dir, periodic_3d=True
    )

    slab_thicknesses = [i for i in range(3, 30+1, 3)]
    surface_energies = []

    for slab_size in slab_thicknesses:
        slab_dir = Path(f"Surface_slab_{slab_size}")
        slab_dir.mkdir(exist_ok=True)
        print(f"\nCalculating for slab size: {slab_size}")

        try:
            _, slab = prepare_structures(initial_atoms, slab_size)
            slab_input_data = get_input_data("relax")
            slab_input_data["control"].update(
                {
                    "etot_conv_thr": 1.0e-4,
                    "forc_conv_thr": 4.0e-4,
                }
            )
            slab_input_data["system"]["nbnd"] = int((3 * len(slab) / 2) * 1.4)
            slab_input_data["electrons"].update(
                {
                    "mixing_beta": 0.2,
                    "conv_thr": 1.0e-6,
                    "mixing_mode": "local-TF",
                    "diagonalization": "david",
                }
            )

            slab_energy, _ = run_calculation(
                slab, slab_input_data, profile, slab_dir, periodic_3d=False
            )

            surface_energy = calculate_surface_energy(
                slab_energy, ouc_energy, slab, ouc
            )

            print(
                f"Surface energy for slab size {slab_size}: {surface_energy:.6f} J/m^2"
            )
            surface_energies.append(surface_energy)

        except Exception as e:
            print(f"Error calculating for slab size {slab_size}: {str(e)}")
            surface_energies.append(np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(slab_thicknesses, surface_energies, marker="o")
    plt.xlabel("Slab Thickness")
    plt.ylabel("Surface Energy (J/m²)")
    plt.title(f"Surface Energy vs Slab Thickness for Li {MILLER_INDEX} Surface")
    plt.grid(True)
    plt.savefig("surface_energy_vs_thickness.png")
    plt.close()

    print("\nSurface Energies:")
    for thickness, energy in zip(slab_thicknesses, surface_energies):
        print(f"Thickness {thickness}: {energy:.6f} J/m²")


if __name__ == "__main__":
    main()
