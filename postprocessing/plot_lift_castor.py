import numpy as np
import matplotlib.pyplot as plt

span = 18.

data = np.genfromtxt('liftDistribution.dat')
plt.plot(data[:,0], data[:,1], 'C0-', label='Vortex')

castor = np.genfromtxt('straight_wing/aero_nodes/rotor_1_blade_1_span.dat', skip_header=1)
castor = castor[-10:,:]
plt.plot(castor[:,0]*18., castor[:,12], 'C1-', label='Castor')
plt.hlines(  1.13696667, min(data[:,0]), max(data[:,0]), colors='C2', label='None')

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
