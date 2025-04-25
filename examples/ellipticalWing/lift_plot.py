import numpy as np
import matplotlib.pyplot as plt
import os

# Global settings for consistent and high-quality plot appearance using rcParams
scale = 1.0
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.titlesize'] = 10 * scale
plt.rcParams['axes.labelsize'] = 10 * scale
plt.rcParams['lines.linewidth'] = 0.8
plt.rcParams['lines.markersize'] = 0.8
plt.rcParams['xtick.labelsize'] = 8 * scale
plt.rcParams['ytick.labelsize'] = 8 * scale
plt.rcParams['legend.fontsize'] = 8 * scale
plt.rcParams['figure.dpi'] = 300  
plt.rcParams['savefig.dpi'] = 300  

# Data loading and processing
ellipticalCase = True



CL_vortex_elliptical = np.genfromtxt('./outputs/liftDistribution_elliptical.dat')
blade_centers_elliptical_delta = CL_vortex_elliptical[:,0]
lift_elliptical_delta = CL_vortex_elliptical[:,1]
Reference_data = 2 * np.pi / (1 + 2 / 6) * 5 * np.pi / 180 * np.ones(len(blade_centers_elliptical_delta))

# Plot setup
fig, ax = plt.subplots(figsize=(4, 2.9))  
ax.plot()
ax.plot(blade_centers_elliptical_delta, lift_elliptical_delta, 'g--', label='SVEN')
ax.plot(blade_centers_elliptical_delta, Reference_data, color='black', linestyle='-', label='Theoretical value')


ax.set_xlabel('Blade center positions [m]')
ax.set_ylabel('$C_L$ [-]')
ax.set_ylim(top=0.5)

ax.grid(True)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.56, 1.0), frameon=True)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to leave space for the legend
base_directory = os.getcwd()
plt.savefig(os.path.join(base_directory, "elliptical_case_plot.eps"), format='eps', dpi=300)
plt.show()
