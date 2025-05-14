import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import os

# -------------------------
# Choose a wind speed
# -------------------------
wind_speed = 24  # choose between 10, 15, ou 24
case_str = str(wind_speed)

scale = 0.8
matplotlib.rcParams.update({
    'axes.titlesize': 24 * scale,
    'axes.labelsize': 24 * scale,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'xtick.labelsize': 20 * scale,
    'ytick.labelsize': 20 * scale,
    'legend.fontsize': 24 * scale,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.fancybox': True,
    'legend.borderpad': 0.8
})


col_map = {
    '10': (0, 1, 2),
    '15': (0, 3, 4),
    '24': (0, 5, 6)
}
x_col, fn_col, ft_col = col_map[case_str]

data_castor = np.genfromtxt('CASTOR_data.dat', skip_header=3)
x_castor = data_castor[:, x_col]
fn_castor = data_castor[:, fn_col]
ft_castor = data_castor[:, ft_col]

f, ax = plt.subplots(ncols=2, figsize=(14, 6))

# CASTOR
line_fn_castor, = ax[0].plot(x_castor, fn_castor, 'r:', label='CASTOR')
line_ft_castor, = ax[1].plot(x_castor, ft_castor, 'r:', label='CASTOR')

# SVEN 
line_fn_sven, = ax[0].plot([], [], 'g--', label='SVEN')
line_ft_sven, = ax[1].plot([], [], 'g--', label='SVEN')

# Format axes
for a in ax:
    a.set_xlim([0., 2.25])
    a.grid()
    a.relim()
    a.autoscale_view()

ax[0].set_xlabel('Blade centers [m]')
ax[0].set_ylabel('Fn [N/m]')

ax[1].set_xlabel('Blade centers [m]')
ax[1].set_ylabel('Ft [N/m]')

# Légende globale
f.legend(['CASTOR', 'SVEN'], loc='upper center', ncol=2, frameon=True, bbox_to_anchor=(0.55, 1.01))
plt.tight_layout()
plt.subplots_adjust(top=0.85)

# -------------------------
# Fonction de mise à jour
# -------------------------
def update_sven_plot(event=None):
    try:
        files = glob.glob(f'./outputs/bladeForces_case_{case_str}_*.dat')
        if not files:
            print(f"Aucun fichier trouvé pour le cas {case_str} m/s.")
            return
        latest_file = max(files, key=os.path.getmtime)
        data_sven = np.genfromtxt(latest_file)
        if data_sven.shape[1] < 3:
            print("Fichier incomplet :", latest_file)
            return

        # Mise à jour des courbes SVEN
        line_fn_sven.set_data(data_sven[:, 0], data_sven[:, 1])
        line_ft_sven.set_data(data_sven[:, 0], data_sven[:, 2])

        # Réajustement des axes
        for a in ax:
            a.relim()
            a.autoscale_view()

        f.canvas.draw()
        print("Données mises à jour depuis :", latest_file)
    except Exception as e:
        print("Erreur lors de la mise à jour :", e)

# -------------------------
# Connexion événement clavier
# -------------------------
f.canvas.mpl_connect('key_press_event', lambda event: update_sven_plot(event) if event.key == 'u' else None)

# -------------------------
# Mise à jour initiale
# -------------------------
update_sven_plot()

plt.show()
