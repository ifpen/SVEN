# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 07:37:15 2022

@author: leguerca
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *
from matplotlib import rc

#span = 2.04

CL_vortex = np.loadtxt('liftDistribution.dat')
#CL_castor = np.genfromtxt('rotor_1_blade_1_span_Flts.dat', skip_header=1)
#castor = CL_castor[-34:, :]

blade_centers_vortex = CL_vortex[:, 0]
lift_coeff_vortex = CL_vortex[:, 1]

#blade_centers_castor = 0.21 + castor[:, 0] * span
#lift_coeff_castor = castor[:, 11]

#plt.plot(blade_centers_vortex, CL_vortex[:, 3], 'C0-', label='No induction')
plt.plot(blade_centers_vortex, lift_coeff_vortex, 'C1--', label='Vortex')
#plt.plot(blade_centers_castor, lift_coeff_castor, 'C2-.', label='Castor')

plt.legend()
plt.xlabel('Blade center positions')
plt.ylabel('CL')
#plt.ylim([5., 15.])
plt.grid()
plt.show()
