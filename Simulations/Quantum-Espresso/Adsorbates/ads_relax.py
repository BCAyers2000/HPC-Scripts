"""
ADSORPTION ENERGY CALCULATOR FOR F, O, AND CO3
Supports solvation, constraints, and external optimisation
"""

import copy
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Literal, Optional

from ase.io import read, write
from ase.atoms import Atoms
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.optimize import GoodOldQuasiNewton
from ase.io.espresso import write_fortran_namelist
from ase.calculators.espresso import Espresso, EspressoProfile

from acat.build import add_adsorbate, add_adsorbate_to_site
from acat.settings import CustomSurface
from acat.adsorption_sites import SlabAdsorptionSites, get_layers


# Type definitions
SystemType = Literal["slab", "molecule", "bulk"]
CalculationType = Literal["scf", "relax"]
EnvironmentType = Literal["vacuum", "solvent"]
AdsorbateType = Literal["F", "O", "CO3"]


@dataclass
class AdsorbateConfig:
    """Configuration for adsorbates"""
    name: str
    initial_height: float
    reference_molecule: Optional[str] = None
    stoichiometry: float = 1.0
    gcscf_mu: Optional[float] = None
    

# Adsorbate configurations
ADSORBATES = {
    "F": AdsorbateConfig(
        name="F",
        initial_height=-0.75,
        reference_molecule="F2",
        stoichiometry=0.5,
    ),
    "O": AdsorbateConfig(
        name="O", 
        initial_height=1.7,
        reference_molecule="O2",
        stoichiometry=0.5,
        gcscf_mu=-2.904,
    ),
    "CO3": AdsorbateConfig(
        name="CO3",
        initial_height=0.4,
        reference_molecule=None,  
        stoichiometry=1.0,
    ),
}


class QECalculator:
    """Handles Quantum ESPRESSO calculations"""
    
    def __init__(
        self, 
        pseudo_dir: str = "/work/e89/e89/ba3g18/Repos/Pseudopotentials/Pslibrary",
        qe_command: str = "srun -v --distribution=block:block --hint=nomultithread --cpu-freq=2250000 /mnt/lustre/a2fs-work3/work/e89/e89/ba3g18/Repos/q-e/bin/pw.x",
        kspacing: float = 0.1
    ):
        self.pseudo_dir = pseudo_dir
        self.qe_command = qe_command
        self.kspacing = kspacing
        
    def get_base_parameters(self) -> Dict:
        """Base QE input parameters"""
        return {
            "control": {
                "restart_mode": "from_scratch",
                "nstep": 999,
                "tprnfor": True,
                "disk_io": "none",
                "etot_conv_thr": 1e-04,
                "forc_conv_thr": 1e-03,
            },
            "system": {
                "ecutwfc": 80.0,
                "ecutrho": 800.0,
                "degauss": 0.01,
                "smearing": "cold",
                "occupations": "smearing",
            },
            "electrons": {
                "conv_thr": 1e-6,
                "mixing_beta": 0.20,
                "mixing_mode": "local-TF",
                "electron_maxstep": 250,
                "diagonalization": "paro",
                "startingwfc": "random",
                "scf_must_converge": False,
            },
            "ions": {
                "ion_dynamics": "bfgs",
                "bfgs_ndim": 1,
                "upscale": 1e5,
            },
        }
    
    def get_environ_parameters(self) -> Dict:
        """Environ-specific parameters for solvation"""
        return {
            "environ": {
                "verbose": 0,
                "cion(1)": 1.0,
                "cion(2)": 1.0,
                "zion(1)": 1.0,
                "zion(2)": -1.0,
                "cionmax":  5.0,
                "system_dim": 2,
                "system_axis": 3,
                "environ_thr": 1.0,
                "env_pressure": 0.0,
                "temperature": 300.0,
                "environ_type": "input",
                "env_electrostatic": True,
                "env_electrolyte_ntyp": 2,
                "env_surface_tension": 37.3,
                "electrolyte_entropy": "full",
                "env_static_permittivity": 89.9,
            },
            "boundary": {
                "alpha": 1.32,
                "radius_mode": "bondi",
                "solvent_mode": "ionic",
                "electrolyte_mode": "ionic",
            },
            "electrostatic": {
                "pbc_dim": 2,
                "pbc_axis": 3,
                "tol": 1.0e-15,
                "inner_tol": 1.0e-20,
                "pbc_correction": "parabolic",
            },
        }
    
    def create_profile(self, use_environ: bool = False) -> EspressoProfile:
        """Create QE execution profile"""
        command = self.qe_command
        if use_environ:
            command += " --environ"
        return EspressoProfile(command=command, pseudo_dir=self.pseudo_dir)
    
    def generate_kpoints(self, atoms: Atoms, system_type: SystemType = "slab") -> List[int]:
        """Generate k-points based on cell size"""
        cell_lengths = atoms.cell.lengths()
        kpts = np.ceil(2 * np.pi / (cell_lengths * self.kspacing)).astype(int)
        if system_type == "slab":
            kpts[2] = 1
        return kpts.tolist()
    
    def prepare_input_data(
        self,
        calculation_type: CalculationType,
        system_type: SystemType,
        includes_adsorbate: bool = False,
        use_gcscf: bool = False,
        gcscf_mu: Optional[float] = None,
        use_external_optimiser: bool = False,
    ) -> Dict:
        """Prepare QE input parameters"""
        params = copy.deepcopy(self.get_base_parameters())
        params["control"]["calculation"] = "scf" if use_external_optimiser else calculation_type
        
        # Set number of atomic types
        if includes_adsorbate:
            params["system"]["ntyp"] = 2 
        
        # Set lattice
        if system_type == "molecule":
            params["system"]["ibrav"] = 0
        else:
            params["system"]["ibrav"] = 8
            params["system"]["celldm(1)"] = 12.983008999907588
            params["system"]["celldm(2)"] = 1.8027760000196391
            params["system"]["celldm(3)"] = 7.575360234766154

        # GCSCF settings
        if use_gcscf and includes_adsorbate and gcscf_mu is not None:
            params["system"]["lgcscf"] = True
            params["system"]["gcscf_mu"] = gcscf_mu
            params["system"]["gcscf_beta"] = 0.15
            params["system"]["gcscf_conv_thr"] = 0.001
            params["electrons"]["conv_thr"] = 1e-9
            params["electrons"]["mixing_beta"] = 0.10
            params["electrons"]["mixing_ndim"] = 14
            
        # Relaxation settings
        if calculation_type == "relax" and not use_external_optimiser:
            params["ions"] = {
                "ion_dynamics": "bfgs",
                "upscale": 1e6,
                "bfgs_ndim": 1
            }
            
        return params
    
    def run_calculation(
        self,
        atoms: Atoms,
        directory: Path,
        pseudopotentials: Dict[str, str],
        calculation_type: CalculationType = "scf",
        system_type: SystemType = "slab",
        includes_adsorbate: bool = False,
        use_environ: bool = False,
        use_gcscf: bool = False,
        gcscf_mu: Optional[float] = None,
        use_external_optimiser: bool = False,
    ) -> Tuple[float, Atoms]:
        """Execute QE calculation"""
        directory.mkdir(exist_ok=True, parents=True)
        
        # Prepare input
        params = self.prepare_input_data(
            calculation_type,
            system_type,
            includes_adsorbate,
            use_gcscf,
            gcscf_mu,
            use_external_optimiser
        )
        
        # Set k-points
        if system_type == "molecule":
            kpts = [1, 1, 1]
        else:
            kpts = self.generate_kpoints(atoms, system_type)
            
        # Write environ file if needed
        if use_environ:
            with (directory / "environ.in").open("w") as f:
                write_fortran_namelist(f, self.get_environ_parameters())
                
        # Create calculator
        profile = self.create_profile(use_environ)
        calc = Espresso(
            input_data=params,
            pseudopotentials=pseudopotentials,
            profile=profile,
            directory=str(directory),
            kpts=kpts,
        )
        atoms.calc = calc
        
        # Run calculation
        if use_external_optimiser and calculation_type == "relax":
            trajectory_file = directory / "QE.traj"
            log_file = directory / "QE.log"
            
            print(f"Using ASE optimiser (GoodOldQuasiNewton) - trajectory: {trajectory_file}")
            opt = GoodOldQuasiNewton(atoms, trajectory=str(trajectory_file), logfile=str(log_file))
            opt.run(fmax=0.05, steps=100)
            
            energy = atoms.get_potential_energy()
            return energy, atoms.copy()
        else:
            energy = atoms.get_potential_energy()
            
            if calculation_type == "relax":
                try:
                    relaxed_atoms = read(directory / "espresso.pwo")
                    return energy, relaxed_atoms
                except Exception as e:
                    print(f"Warning: Could not read optimised structure: {e}")
                    return energy, atoms
            
            return energy, atoms


def create_co3_molecule() -> Atoms:
    """Create CO3 molecule with proper geometry"""
    bond_length = 1.29  
    
    positions = [
        [0.0, 0.0, 0.0],  # C at origin
        [bond_length, 0.0, 0.0],  # O1
        [-bond_length/2, bond_length*np.sqrt(3)/2, 0.0],  # O2
        [-bond_length/2, -bond_length*np.sqrt(3)/2, 0.0]  # O3
    ]
    
    co3 = Atoms('CO3', positions=positions)
    return co3


class SlabHandler:
    """Handles slab structures and adsorption sites"""
    
    @staticmethod
    def load_slab(
        filename: str,
        surrogate_metal: str = "Li"
    ) -> Tuple[Atoms, SlabAdsorptionSites, CustomSurface, Dict[str, int], List[Dict]]:
        """Load and analyze slab structure"""
        slab = read(filename)
        slab.pbc = [True, True, False]
        slab.set_initial_magnetic_moments([0] * len(slab))
        
        # Detect layers
        layers, _ = get_layers(slab, (0, 0, 1), tolerance=0.01)
        n_layers = len(np.unique(layers))
        print(f"\nDetected {n_layers} layers in the structure")
        
        # Create custom surface
        custom_surface = CustomSurface(slab, n_layers=n_layers)
        
        # Create adsorption site finder
        sas = SlabAdsorptionSites(
            slab, 
            surface=custom_surface, 
            surrogate_metal=surrogate_metal,
            both_sides=False
        )
        
        # Get all sites
        all_sites = sas.get_sites()
        
        # Count site types
        site_counts = {}
        for site in all_sites:
            site_type = site["site"]
            site_counts[site_type] = site_counts.get(site_type, 0) + 1
            
        print(f"Found {len(all_sites)} total adsorption sites")
        print("Site types available:")
        for st, count in sorted(site_counts.items()):
            print(f"  {st}: {count} sites")
            
        return slab, sas, custom_surface, site_counts, all_sites
    
    @staticmethod
    def find_center_site(
        sites: List[Dict],
        slab: Atoms,
        site_type: str
    ) -> Tuple[Dict, float]:
        """Find site of given type closest to slab center"""
        slab_center = np.mean(slab.positions, axis=0)
        
        closest_site = None
        min_distance = float("inf")
        
        for site in sites:
            if site["site"] != site_type:
                continue
                
            position = site["position"]
            # Distance in xy-plane only
            distance = np.sqrt(
                (position[0] - slab_center[0])**2 + 
                (position[1] - slab_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_site = site
                
        return closest_site, min_distance
    
    @staticmethod
    def add_adsorbate_to_slab(
        slab: Atoms,
        adsorbate_type: AdsorbateType,
        site: Dict,
        height: float,
        constrain_slab: bool = False
    ) -> Atoms:
        """Add adsorbate to slab at specified site using ACAT"""
        atoms = slab.copy()
        
        # Create adsorbate
        if adsorbate_type == "CO3":
            adsorbate = create_co3_molecule()
        else:
            # Single atoms (F, O) - ACAT expects string for single atoms
            adsorbate = adsorbate_type
        add_adsorbate_to_site(atoms, adsorbate=adsorbate, site=site, height=height)
        atoms.set_initial_magnetic_moments([0] * len(atoms))
        
        # Apply constraints if requested
        if constrain_slab:
            n_slab_atoms = len(slab)
            constraint_mask = [True] * n_slab_atoms + [False] * (len(atoms) - n_slab_atoms)
            atoms.set_constraint(FixAtoms(mask=constraint_mask))
            print("Applied constraint: Fixed slab atoms, only adsorbate can move")
            
        return atoms


def create_reference_molecule(molecule_name: str) -> Atoms:
    """Create reference molecule for adsorption energy calculation"""
    if molecule_name == "CO3":
        mol = create_co3_molecule()
    else:
        mol = molecule(molecule_name)
        
    mol.set_cell([20.0, 20.0, 20.0])
    mol.center()
    mol.set_initial_magnetic_moments([0] * len(mol))
    return mol


def calculate_adsorption_energy(
    total_energy: float,
    clean_slab_energy: float,
    reference_energy: float,
    stoichiometry: float = 1.0
) -> float:
    """Calculate adsorption energy"""
    return total_energy - clean_slab_energy - stoichiometry * reference_energy


def write_results(
    output_file: Path,
    adsorbate_type: AdsorbateType,
    site_type: str,
    environment: str,
    clean_energy: float,
    reference_energy: float,
    optimised_energy: float,
    adsorption_energy: float,
    optimised_structure: Atoms,
    reference_bond_length: Optional[float] = None,
    constrained: bool = False,
    centered: bool = False,
    use_gcscf: bool = False,
    use_external: bool = False,
):
    """Write formatted results to file"""
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{adsorbate_type} ADSORPTION ON SURFACE - {site_type.upper()} SITE IN {environment.upper()}\n")
        
        if constrained:
            f.write("WITH CONSTRAINED SLAB ATOMS\n")
        if centered:
            f.write("WITH ADSORBATE AT SLAB CENTER\n")
        if use_gcscf:
            f.write("WITH GCSCF ENABLED\n")
        if use_external:
            f.write("WITH EXTERNAL ASE optimiseR\n")
            
        f.write("=" * 80 + "\n\n")
        
        f.write("REFERENCE ENERGIES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Clean slab energy:              {clean_energy:.6f} eV\n")
        f.write(f"Reference molecule energy:      {reference_energy:.6f} eV\n")
        
        if reference_bond_length is not None:
            f.write(f"Reference bond length:          {reference_bond_length:.3f} Å\n")
            
        f.write("\nOPTIMISATION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Final optimised energy:         {optimised_energy:.6f} eV\n")
        f.write(f"Adsorption energy:              {adsorption_energy:.6f} eV\n")
        
        # Calculate final height
        slab_z_max = max([atom.position[2] for atom in optimised_structure[:-len(ADSORBATES[adsorbate_type].name)]])
        adsorbate_z_min = min([atom.position[2] for atom in optimised_structure[-len(ADSORBATES[adsorbate_type].name):]])
        final_height = adsorbate_z_min - slab_z_max
        
        f.write(f"Final height above surface:     {final_height:.3f} Å\n")
        f.write("\n" + "=" * 80 + "\n")


def main(
    slab_file: str,
    adsorbate_type: AdsorbateType,
    site_type: str = "ontop",
    environments: List[EnvironmentType] = ["vacuum", "solvent"],
    constrained: bool = False,
    center_adsorbate: bool = False,
    use_gcscf: bool = False,
    use_external_optimiser: bool = False,
    surrogate_metal: str = "Li",
    pseudopotentials: Optional[Dict[str, str]] = None,
):
    """Main function to run adsorption calculations"""
    
    # Default pseudopotentials
    if pseudopotentials is None:
        pseudopotentials = {
            "Li": "Li.pbe-sl-kjpaw_psl.1.0.0.UPF",
            "F": "F.pbe-n-kjpaw_psl.1.0.0.UPF",
            "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
            "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
        }
    # Get adsorbate configuration
    adsorbate_config = ADSORBATES[adsorbate_type]
    
    # Initialize calculator
    qe_calc = QECalculator()
    
    # Process each environment
    for environment in environments:
        use_environ = (environment == "solvent")
        dir_name_parts = [
            f"{adsorbate_type}-{site_type}",
            environment,
        ]
        if constrained:
            dir_name_parts.append("constrained")
        if center_adsorbate:
            dir_name_parts.append("centered")
        if use_gcscf:
            dir_name_parts.append("gcscf")
        if use_external_optimiser:
            dir_name_parts.append("external")
            
        base_dir = Path("_".join(dir_name_parts))
        base_dir.mkdir(exist_ok=True)
        
        print(f"\n{'=' * 60}")
        print(f"STARTING {adsorbate_type} ADSORPTION IN {environment.upper()}")
        print(f"Site type: {site_type}")
        if constrained:
            print("Constraints: ENABLED (slab atoms fixed)")
        if center_adsorbate:
            print("Position: CENTER of slab")
        if use_gcscf:
            print("GCSCF: ENABLED")
        if use_external_optimiser:
            print("optimiser: EXTERNAL (ASE)")
        print(f"{'=' * 60}\n")
        
        # Load slab
        slab, sas, custom_surface, site_counts, all_sites = SlabHandler.load_slab(slab_file, surrogate_metal)
        
        # Step 1: optimise clean slab
        print("Step 1: optimising clean slab...")
        clean_dir = base_dir / "01_clean_slab"
        clean_energy, optimised_slab = qe_calc.run_calculation(
            slab,
            clean_dir,
            pseudopotentials,
            calculation_type="relax",
            system_type="slab",
            use_environ=use_environ,
        )
        print(f"Clean slab energy: {clean_energy:.6f} eV")
        
        # Step 2: Calculate reference molecule energy
        reference_bond_length = None
        if adsorbate_config.reference_molecule:
            print(f"\nStep 2: optimising {adsorbate_config.reference_molecule} reference molecule...")
            mol_dir = base_dir / "02_reference_molecule"
            ref_mol = create_reference_molecule(adsorbate_config.reference_molecule)
            reference_energy, optimised_ref = qe_calc.run_calculation(
                ref_mol,
                mol_dir,
                pseudopotentials,
                calculation_type="relax",
                system_type="molecule",
                includes_adsorbate=True,
                use_environ=use_environ,
            )
            print(f"{adsorbate_config.reference_molecule} energy: {reference_energy:.6f} eV")
            
            # Get bond length if diatomic
            if len(optimised_ref) == 2:
                reference_bond_length = optimised_ref.get_distance(0, 1)
                print(f"{adsorbate_config.reference_molecule} bond length: {reference_bond_length:.3f} Å")
        else:
            print(f"\nStep 2: optimising {adsorbate_type} reference molecule...")
            mol_dir = base_dir / "02_reference_molecule"
            ref_mol = create_reference_molecule(adsorbate_type)
            reference_energy, optimised_ref = qe_calc.run_calculation(
                ref_mol,
                mol_dir,
                pseudopotentials,
                calculation_type="relax",
                system_type="molecule",
                includes_adsorbate=True,
                use_environ=use_environ,
            )
            print(f"{adsorbate_type} energy: {reference_energy:.6f} eV")
        
        # Step 3: Add adsorbate and optimise
        print(f"\nStep 3: optimising {adsorbate_type} on {site_type} site...")
        
        # Find adsorption site
        if center_adsorbate:
            site, distance = SlabHandler.find_center_site(all_sites, optimised_slab, site_type)
            print(f"Selected {site_type} site at center (distance: {distance:.3f} Å)")
        else:
            site = next(s for s in all_sites if s["site"] == site_type)
            
        # Create structure with adsorbate
        adsorbate_slab = SlabHandler.add_adsorbate_to_slab(
            optimised_slab,
            adsorbate_type,
            site,
            adsorbate_config.initial_height,
            constrain_slab=constrained
        )
        
        # Save initial structure for debugging
        write(base_dir / "initial_structure.xyz", adsorbate_slab, format="extxyz")
        print(f"Saved initial structure with {len(adsorbate_slab)} atoms")
        
        # optimise with adsorbate
        opt_dir = base_dir / "03_optimisation"
        optimised_energy, optimised_structure = qe_calc.run_calculation(
            adsorbate_slab,
            opt_dir,
            pseudopotentials,
            calculation_type="relax",
            system_type="slab",
            includes_adsorbate=True,
            use_environ=use_environ,
            use_gcscf=use_gcscf,
            gcscf_mu=adsorbate_config.gcscf_mu,
            use_external_optimiser=use_external_optimiser,
        )
        
        # Calculate adsorption energy
        adsorption_energy = calculate_adsorption_energy(
            optimised_energy,
            clean_energy,
            reference_energy,
            adsorbate_config.stoichiometry
        )
        
        print(f"\nFinal Results:")
        print(f"optimised total energy: {optimised_energy:.6f} eV")
        print(f"Adsorption energy: {adsorption_energy:.6f} eV")
        
        # Write results
        write_results(
            base_dir / f"results_{environment}.txt",
            adsorbate_type,
            site_type,
            environment,
            clean_energy,
            reference_energy,
            optimised_energy,
            adsorption_energy,
            optimised_structure,
            reference_bond_length,
            constrained,
            center_adsorbate,
            use_gcscf,
            use_external_optimiser,
        )
        write(
            base_dir / f"optimised_structure_{environment}.xyz",
            optimised_structure,
            format="extxyz"
        )
        
        print(f"\nCompleted {environment} calculations. Results saved to {base_dir}")
    
    print("\n" + "=" * 60)
    print("ALL CALCULATIONS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main(
        slab_file="/work/e89/e89/ba3g18/Working_dir/QE/Lithium/Adsorption/Oxygen/Voltage/[0.0V]/4FOLD/320.xyz",
        adsorbate_type="O",  # Can be "F", "O", or "CO3"
        site_type="ontop",
        environments=["solvent"],
        constrained=True,
        center_adsorbate=True,
        use_gcscf=True,
        use_external_optimiser=True,
        surrogate_metal="Li",
    )