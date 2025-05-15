import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# Style graphique compact
scale = 0.8
matplotlib.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'axes.titlesize': 16 * scale,
    'axes.labelsize': 16 * scale,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'xtick.labelsize': 12 * scale,
    'ytick.labelsize': 12 * scale,
    'legend.fontsize': 14 * scale,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.fancybox': True,
    'legend.borderpad': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# Données
CL_vortex_elliptical = np.genfromtxt('./outputs/liftDistribution_elliptical.dat')
blade_centers_elliptical_delta = CL_vortex_elliptical[:, 0]
lift_elliptical_delta = CL_vortex_elliptical[:, 1]
Reference_data = 2 * np.pi / (1 + 2 / 6) * 5 * np.pi / 180 * np.ones_like(blade_centers_elliptical_delta)

# Figure plus compacte
fig, ax = plt.subplots(figsize=(5, 3.2))

line_sven, = ax.plot(blade_centers_elliptical_delta, lift_elliptical_delta, 'g--', label='SVEN')
line_theory, = ax.plot(blade_centers_elliptical_delta, Reference_data, 'k-', label='Theoretical value')

ax.set_xlabel('Blade centers [m]')
ax.set_ylabel('$C_L$ [-]')
ax.set_ylim(top=0.5)
ax.grid(True)

# Légende globale au-dessus
fig.legend([line_sven, line_theory], ['SVEN', 'Theoretical value'],
           loc='upper center', ncol=2, frameon=True, bbox_to_anchor=(0.55, 1.02))

plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Laisse de la place pour la légende

# Sauvegarde PNG
plt.savefig(os.path.join(os.getcwd(), "elliptical_case_plot.png"), format='png')
plt.show()
