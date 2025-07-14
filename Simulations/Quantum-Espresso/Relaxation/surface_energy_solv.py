"""
Quantum ESPRESSO Calculator
"""

import shutil
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional

from ase.build import bulk
from ase.atoms import Atoms
from ase.io import read, write
from ase.optimize import GoodOldQuasiNewton
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import SlabGenerator
from ase.io.espresso import write_fortran_namelist
from ase.calculators.espresso import Espresso, EspressoProfile


@dataclass
class QEConfig:
    """Configuration for Quantum ESPRESSO calculations."""
    ibrav: int = 6
    element: str = "Li"
    kspacing: float = 0.10
    min_slab_size: int = 15
    min_vacuum_size: int = 20
    lattice_scaling: Tuple[int, int, int] = (2, 2, 1)
    miller_index: List[int] = field(default_factory=lambda: [1, 0, 0])

    restart: bool = True
    use_environ: bool = True
    use_external: bool = False
    read_structures: bool = True
    gcscf_mu: Optional[float] = None
    qe_cmd: str = "/home/ba3g18/Repos/q-e/bin/pw.x"
    pseudo_dir: str = "/home/ba3g18/Repos/Pseudopotentials/SSSP_1.3.0_PBE_efficiency"
    pseudopotentials: Dict[str, str] = field(default_factory=lambda: {"Li": "li_pbe_v1.4.uspp.F.UPF"})


class QESurfaceCalculator:
    """Implementation for calculating surface energies using Quantum ESPRESSO."""

    def __init__(self, base_dir: str, config: Optional[QEConfig] = None):
        """Initialise the Surface Energy Calculator with the given configuration."""
        self.config = config or QEConfig()
        self.base_dir = Path(base_dir)
        self.lgcscf = self.config.gcscf_mu is not None

        self.ouc_dir = Path("OUC")
        self.slab_dir = Path("Surface")
        self.restart_path = self.base_dir / "Surface/Li"
        self.ouc_path = self.base_dir / "OUC/espresso.pwo"
        self.slab_path = self.base_dir / "Surface/espresso.pwo"
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure calculation directories exist."""
        self.ouc_dir.mkdir(exist_ok=True)
        self.slab_dir.mkdir(exist_ok=True)
        
    """
    K-point 
    """
    
    def generate_kpts(self, atoms: Atoms) -> List[int]:
        """Generate k-points grid based on structure size."""
        kpts = np.ceil(2 * np.pi / (atoms.cell.lengths() * self.config.kspacing)).astype(int)
        if not atoms.pbc[2]:
            kpts[2] = 1
        return kpts.tolist()
    

    """
    Quantum ESPRESSO Parameters and Profiles
    """
    
    def create_espresso_profile(self, use_environ: bool = False) -> EspressoProfile:
        """Create an EspressoProfile for running QE calculations."""
        command = f"srun  {self.config.qe_cmd}"
        if use_environ:
            command += " --environ"
        return EspressoProfile(command=command, pseudo_dir=self.config.pseudo_dir)
    
    def get_qe_parameters(self, calc_type: str, for_slab: bool = False) -> Dict[str, Any]:
        """Get Quantum ESPRESSO calculation parameters."""
        restart_mode = "restart" if self.config.restart else "from_scratch"
        params = {
            "control": {
                "calculation": calc_type,
                "restart_mode": restart_mode,
                "nstep": 200,
                "outdir": "./Li/",
                "prefix": "Lithium",
                "etot_conv_thr": 1.0e-6,
                "forc_conv_thr": 1.0e-6,
                "pseudo_dir": self.config.pseudo_dir,
                "disk_io": "low",
            },
            "system": {
                "ibrav": 0,
                "ecutwfc": 40.0,
                "ecutrho": 600,
                "occupations": "smearing",
                "degauss": 0.01,
                "smearing": "mv",
                "input_dft": "pbe",
            },
            "electrons": {
                "conv_thr": 1.0e-10,
                "mixing_beta": 0.80,
                "mixing_mode": "plain",
                "electron_maxstep": 250,
                "scf_must_converge": False,
                "startingpot": "atomic" if not self.config.restart else "file",
                "startingwfc": "atomic+random" if not self.config.restart else "file",
                "diagonalization": "rmm-davidson",
            },
            "ions": {"ion_dynamics": "bfgs", "upscale": 1e6, "bfgs_ndim": 6},
            "cell": {"press_conv_thr": 0.1, "cell_dofree": "all"},
        }

        if for_slab:
            params["system"]["ibrav"] = self.config.ibrav
            params["control"].update({"etot_conv_thr": 1.0e-5, "forc_conv_thr": 3.88e-4})
            params["electrons"].update({
                "conv_thr": 1.0e-6,
                "mixing_beta": 0.10,
                "diago_thr_init": 1.0e-8,
                "mixing_mode": "local-TF",
                "diagonalization": "paro",
            })

            if self.config.use_external:
                params["control"].update({"calculation": "scf", "tprnfor": True})
                params["electrons"].update({"conv_thr": 1.0e-11})
                
            if self.lgcscf:
                params["system"].update({"lgcscf": True, "gcscf_mu": self.config.gcscf_mu, "gcscf_beta": 0.15, "gcscf_conv_thr": 1e-03})

        return params
    
    def get_environ_parameters(self) -> Dict[str, Any]:
        """Get Environ solvation module parameters."""
        return  {
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

    """
    Structure Generation and Preparation
    """
    
    def copy_restart_files(self, source_path: Path, destination_dir: Path) -> None:
        """Copy restart files from previous calculation."""
        destination_outdir = destination_dir / "Li"

        if destination_outdir.exists():
            shutil.rmtree(destination_outdir)

        destination_outdir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"Copying restart files from {source_path} to {destination_outdir}")
            for item in source_path.glob("*"):
                if item.is_file():
                    shutil.copy2(item, destination_outdir)
                elif item.is_dir():
                    shutil.copytree(
                        item, destination_outdir / item.name, dirs_exist_ok=True
                    )
            print("Restart files copied successfully")
        except Exception as e:
            raise RuntimeError(f"Error copying restart files: {e}")
    
    def prepare_structures(self) -> Tuple[Atoms, Atoms]:
        """Prepare oriented unit cell and slab structures."""
        if self.config.read_structures:
            try:
                print(f"Reading structures from: {self.ouc_path} and {self.slab_path}")
                return read(self.ouc_path, index=-1), read(self.slab_path, index=-1)
            except Exception as e:
                raise RuntimeError(f"Failed to read structures: {e}")

        print(f"Creating {self.config.element} bulk structure")
        initial_bulk = bulk(self.config.element, "bcc", cubic=True, a=3.435)

        print(f"Generating slab with Miller index {self.config.miller_index}")
        structure = AseAtomsAdaptor.get_structure(initial_bulk)
        slabgen = SlabGenerator(
            structure,
            miller_index=self.config.miller_index,
            min_slab_size=self.config.min_slab_size,
            min_vacuum_size=self.config.min_vacuum_size,
            lll_reduce=False,
            center_slab=True,
            primitive=True,
            max_normal_search=5,
            in_unit_planes=True,
        )

        slabs = slabgen.get_slabs()
        if not slabs:
            raise ValueError(f"No slabs generated with Miller index {self.config.miller_index}")

        ouc = AseAtomsAdaptor.get_atoms(slabs[0].oriented_unit_cell)
        slab = AseAtomsAdaptor.get_atoms(slabs[0].get_orthogonal_c_slab()) * self.config.lattice_scaling
        slab.center(vacuum=self.config.min_vacuum_size, axis=2)

        return ouc, slab

    """
    Calculation Execution Methods
    """
    
    def run_calculation(
        self,
        atoms: Atoms,
        params: Dict[str, Any],
        profile: EspressoProfile,
        directory: Path,
        is_slab: bool = False,
    ) -> Tuple[float, Atoms]:
        """Run a Quantum ESPRESSO calculation."""
        if is_slab and self.config.restart and Path(self.restart_path).exists():
            self.copy_restart_files(Path(self.restart_path), directory)
            
        calc = Espresso(
            input_data=params,
            pseudopotentials=self.config.pseudopotentials,
            profile=profile,
            directory=directory,
            kpts=self.generate_kpts(atoms),
        )
        atoms.calc = calc

        if self.config.use_environ and not atoms.pbc[2]:
            with (directory / "environ.in").open("w") as f:
                write_fortran_namelist(f, self.get_environ_parameters())

        print(f"Running calculation in {directory}")

        if self.config.use_external:
            opt = GoodOldQuasiNewton(
                atoms, trajectory=f"{directory}/QE.traj", logfile=f"{directory}/QE.log"
            )
            opt.run(fmax=0.01, steps=1000)
            energy = atoms.get_potential_energy()
            return energy, atoms.copy()
        else:
            energy = atoms.get_potential_energy()
            try:
                relaxed_atoms = read(f"{directory}/espresso.pwo")
                return energy, relaxed_atoms
            except Exception as e:
                raise RuntimeError(f"Failed to read relaxed structure: {e}")
    
    def calculate_surface_energy(
        self, ouc_energy: float, slab_energy: float, slab: Atoms, ouc: Atoms
    ) -> float:
        """Calculate surface energy from bulk and slab energies."""
        area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
        bulk_energy_per_atom = ouc_energy / len(ouc)
        slab_excess_energy = slab_energy - (len(slab) * bulk_energy_per_atom)
        return (slab_excess_energy / (2 * area)) * 16.02

    """
    Main Calculation Workflow
    """
    
    def run(self) -> float:
        """Run full surface energy calculation workflow."""
        print(f"Starting surface energy calculation for {self.config.element} "
              f"{self.config.miller_index} surface")

        """
        Oriented Unit Cell (OUC) Calculation
        """
        ouc, slab = self.prepare_structures()

        ouc_profile = self.create_espresso_profile(use_environ=False)
        ouc_params = self.get_qe_parameters("vc-relax")
        print(f"Running OUC calculation with standard parameters")
        ouc_energy, _ = self.run_calculation(ouc, ouc_params, ouc_profile, self.ouc_dir)
        
        """
        Slab Calculation
        """
        slab.pbc = [True, True, False]
        slab.center(vacuum=self.config.min_vacuum_size, axis=2)
        slab_params = self.get_qe_parameters("relax", for_slab=True)
        
        nbnd = int(sum(slab.numbers) / 2 * 1.5)
        slab_params["system"]["nbnd"] = nbnd
        slab_profile = self.create_espresso_profile(
            use_environ=self.config.use_environ
        )
        
        print(f"Running slab calculation")
        slab_energy, relaxed_slab = self.run_calculation(
            slab, slab_params, slab_profile, self.slab_dir, is_slab=True
        )

        """
        Surface Energy Calculation and Output
        """
        write("relaxed_slab.xyz", relaxed_slab, format="extxyz")
        surface_energy = self.calculate_surface_energy(
            ouc_energy, slab_energy, slab, ouc
        )
        return surface_energy



def main():
    """Run Li surface energy calculation."""
    base_dir = "/scratch/ba3g18/QE/Lithium/Production/Voltage/[0.5V]/[100]"

    print("=" * 50)
    print("Surface Energy Calculator")
    print("=" * 50)

    config = QEConfig(
        element="Li",
        ibrav=6,
        miller_index=[1, 0, 0],
        lattice_scaling=(2, 2, 1),
        min_slab_size=9,
        min_vacuum_size=20,
        restart=False,
        use_environ=True,
        use_external=True,
        read_structures=True,
        #gcscf_mu=-4.404, # Uncomment to enable grand-canonical SCF
    )

    calculator = QESurfaceCalculator(base_dir=base_dir, config=config)

    try:
        surface_energy = calculator.run()
        print("=" * 50)
        print(f"Surface energy: {surface_energy:.8f} J/m²")
        print("=" * 50)
        return surface_energy
    except Exception as e:
        print("=" * 50)
        print(f"Error: {e}")
        print("=" * 50)
        return None


if __name__ == "__main__": 
    main()