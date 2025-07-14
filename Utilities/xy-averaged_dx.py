"""
XY-Averaged Electrolyte Concentration Analysis Script

This script reads concentration data for Li+ and PF6- ions from DX files,
performs xy-plane averaging, and plots the results along the z-axis.

Usage:
1. Ensure you have DX files with concentration data for Li+ and PF6- ions.
2. Run this script to generate the xy-averaged concentration plot.

Note:
this will also work for any DX files i.e. electrostaic potential

Dependencies:
- NumPy
- Matplotlib
- GridData
"""

import numpy as np
import matplotlib.pyplot as plt
from gridData import Grid
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

Li_conc_data = Grid("Li_out_bion_conc_species_1.dx")
pf6_conc_data = Grid("Li_out_bion_conc_species_2.dx")

av_li_conc = np.mean(Li_conc_data.grid, axis=(0, 1))
av_pf6_conc = np.mean(pf6_conc_data.grid, axis=(0, 1))

z_axis = np.arange(Li_conc_data.grid.shape[2]) * Li_conc_data.delta[2]

plt.figure(figsize=(10, 6))
plt.ylabel('xy-averaged electrolyte concentration / M', fontsize=15)
plt.xlabel('z / Å', fontsize=15)
plt.plot(z_axis, av_li_conc, label=r'Li+')
plt.plot(z_axis, av_pf6_conc, label=r'PF6−')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("average_concentration_plot.png", dpi=2000)
plt.show()
