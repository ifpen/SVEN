import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('bladeForces.dat')
plt.plot(data[:,0], data[:,1])
plt.show()

# MORBIER: MOdelling Rotor Blades Induction ... 
# PITCHOU: 
# MUSHROOM
