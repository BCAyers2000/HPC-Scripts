from ase.build import bulk
from quacc.recipes.aims.core import static_job
import os

# Simple bulk calculation
atoms = bulk("Si", "diamond", a=5.431)
results = static_job(
    atoms, 
    species_defaults="intermediate",
    kspacing=0.01,
)