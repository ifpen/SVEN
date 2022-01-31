# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 07:37:15 2022

@author: leguerca
"""

import numpy as np
import matplotlib.pyplot as plt
from math import *
from matplotlib import rc
import matplotlib

matplotlib.rcParams.update({'font.size':14})

span = 2.04

CL_vortex = np.loadtxt('liftDistribution.dat')
CL_castor = np.genfromtxt('/work/blondelf/work/Calculs/MexNext/MexNext3_Axial_Third_Round_February_2017/To_MexNext/IFPEN_VL_2D/Lifting_Line_Variables.dat', skip_header=1)
castor = CL_castor #[-34:, :]

blade_centers_vortex = CL_vortex[:, 0]
lift_coeff_vortex = CL_vortex[:, 3]

#blade_centers_castor = 0.21 + castor[:, 0] * span
#lift_coeff_castor = castor[:, 7]

#plt.plot(blade_centers_vortex, CL_vortex[:, 3], 'C0-', label='No induction')
plt.plot(CL_vortex[:,0], CL_vortex[:,3], 'C1--', label='Vortex')
plt.plot(castor[:,0], castor[:,7], 'C2-.', label='Castor')

plt.legend()
plt.xlabel('Blade center positions (m)')
plt.ylabel('uEffective (m/s)')
#plt.ylim([5., 15.])
plt.grid()
plt.show()
