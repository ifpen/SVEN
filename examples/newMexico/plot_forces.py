import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Update default matplotlib parameters (font size, etc.)
scale = 0.8
matplotlib.rcParams.update({'axes.titlesize': 24 * scale})
matplotlib.rcParams.update({'axes.labelsize': 24 * scale})
matplotlib.rcParams.update({'lines.linewidth': 3})
matplotlib.rcParams.update({'lines.markersize': 8})
matplotlib.rcParams.update({'xtick.labelsize': 20 * scale})
matplotlib.rcParams.update({'ytick.labelsize': 20 * scale})
matplotlib.rcParams.update({'legend.fontsize': 19 * scale})

f, ax = plt.subplots(ncols=2, figsize=(16,6))

data = np.genfromtxt('./outputs/liftDistribution_case_10.dat')
ax[0].plot(data[:,0], data[:,1], '-')
# data = np.genfromtxt('/work/blondelf/work/Calculs/MexNext/MexNext3_Axial_Third_Round_February_2017/To_MexNext/IFPEN_VL_2D/Loads_CASTOR.dat', skip_header=3)
# ax[0].plot(data[:,0], data[:,3], '--')
ax[0].legend(['PITCHOU', 'CASTOR'])
ax[0].grid()
ax[0].set_xlabel('Distance to hub center (m)')
ax[0].set_ylabel('Normal force (N/m)')
ax[0].set_xlim([0., 2.25])
ax[0].set_ylim([0., 100.])

data = np.genfromtxt('./outputs/liftDistribution_case_10.dat')
ax[1].plot(data[:,0], data[:,2], '-')
# data = np.genfromtxt('/work/blondelf/work/Calculs/MexNext/MexNext3_Axial_Third_Round_February_2017/To_MexNext/IFPEN_VL_2D/Loads_CASTOR.dat', skip_header=3)
# ax[1].plot(data[:,0], data[:,4], '--')
#plt.legend(['PITCHOU', 'CASTOR'])
ax[1].grid()
ax[1].set_xlabel('Distance to hub center (m)')
ax[1].set_ylabel('Tangential force (N/m)')
ax[1].set_xlim([0., 2.25])
ax[1].set_ylim([0., 60.])

plt.tight_layout()

plt.savefig('ForceComparison.png', format='png', dpi=300)
plt.show()
