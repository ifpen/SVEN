import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('bladeForces_359.dat')
plt.plot(data[:,0], data[:,1], '-')

data = np.genfromtxt('/work/blondelf/work/Calculs/MexNext/MexNext3_Axial_Third_Round_February_2017/To_MexNext/IFPEN_VL_2D/Loads_CASTOR.dat', skip_header=3)
plt.plot(data[:,0], data[:,3])

plt.legend(['PITCHOU', 'CASTOR'])

plt.show()


data = np.genfromtxt('bladeForces_359.dat')
plt.plot(data[:,0], data[:,2], '-')

data = np.genfromtxt('/work/blondelf/work/Calculs/MexNext/MexNext3_Axial_Third_Round_February_2017/To_MexNext/IFPEN_VL_2D/Loads_CASTOR.dat', skip_header=3)
plt.plot(data[:,0], data[:,4])

plt.legend(['PITCHOU', 'CASTOR'])

plt.show()
