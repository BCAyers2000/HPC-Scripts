import numpy as np
from pathlib import Path
from ase.cluster import Octahedron
from acat.build import add_adsorbate
from acat.adsorption_sites import ClusterAdsorptionSites
from ase.calculators.aims import Aims, AimsProfile

def create_aims_profile():
    return AimsProfile(
        command="srun --mpi=pmix /iridisfs/home/ba3g18/Repos/FHIaims/build/aims.240717.scalapack.mpi.x",
        default_species_directory="/iridisfs/home/ba3g18/Repos/FHIaims/species_defaults/defaults_2020/intermediate/",
    )


def main():
    atoms = Octahedron("Pt", 3, 1)
    atoms.center()
    atoms.pbc = False
    atoms.set_initial_magnetic_moments([2.0]*len(atoms))
    cas = ClusterAdsorptionSites(atoms, 
                                allow_6fold=False,
                                composition_effect=False,
                                label_sites=True,
                                surrogate_metal='Pt')

    add_adsorbate(
            atoms,
            adsorbate="O",
            site="hcp",
            surface="fcc111",
            height=2.0,
            adsorption_sites=cas,
    )

    base_dir = Path("rundir")
    base_dir.mkdir(exist_ok=True)

    calc = Aims(
        directory=base_dir,
        profile=create_aims_profile(),
        xc="pbe",
        relativistic="atomic_zora scalar",
        occupation_type="gaussian 0.1",    
        sc_iter_limit=1000,
        sc_accuracy_rho= 1e-5,       
        sc_accuracy_etot= 1e-5,    
        mixer="pulay",
        n_max_pulay=14, 
        charge_mix_param=0.02,
        spin="collinear",     
        output_level="MD_light",
        relax_geometry="bfgs 1e-3"
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy:.3f} eV")

if __name__ == "__main__":
    main()