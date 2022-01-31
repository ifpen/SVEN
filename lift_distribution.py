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
CL_castor = np.genfromtxt('../../../2021/Codes/aerowt/Castor/TestCase/mexico_standard/aero_nodes/rotor_1_blade_1_span.dat', skip_header=1)
castor = CL_castor[-34:, :]

blade_centers_vortex = CL_vortex[:, 0]
lift_coeff_vortex = CL_vortex[:, 2]

blade_centers_castor = 0.21 + castor[:, 0] * span
lift_coeff_castor = castor[:, 11]

plt.plot(blade_centers_vortex, CL_vortex[:, 3], 'C0-', label='No induction')
plt.plot(blade_centers_vortex, lift_coeff_vortex, 'C1--', label='PITCHOU')
plt.plot(blade_centers_castor, lift_coeff_castor, 'C2-.', label='Castor Ptcles')

#CL_castor = np.genfromtxt('/work/blondelf/work/Calculs/MexNext/MexNext3_Axial_Third_Round_February_2017/To_MexNext/IFPEN_VL_2D/Loads.dat', skip_header=1)
#plt.plot(CL_castor[:,0], CL_castor[0,3], 'C3-.', label='Castor Flts')

plt.legend()
plt.xlabel('Blade center positions')
plt.ylabel('CL')
plt.ylim([0., 15.])
plt.grid()
plt.show()


f, ax = plt.subplots(ncols=2, figsize=(16,6))

CL_vortex = np.loadtxt('bladeForces.dat')
CL_castor = np.genfromtxt('../../../2021/Codes/aerowt/Castor/TestCase/mexico_standard/castor_particles.dat', skip_header=1)
CL_castor_Flts = np.genfromtxt('/work/blondelf/work/Calculs/MexNext/MexNext3_Axial_Third_Round_February_2017/To_MexNext/IFPEN_VL_2D/Loads_CASTOR.dat', skip_header=2)

castor = CL_castor[-34:, :]




blade_centers_vortex = CL_vortex[:, 0]
lift_coeff_vortex = CL_vortex[:, 1]

blade_centers_castor = 0.21 + castor[:, 0] * span
lift_coeff_castor = castor[:, 15]

ax[0].plot(blade_centers_vortex, lift_coeff_vortex, 'C1--', label='PITCHOU')
ax[0].plot(blade_centers_castor, lift_coeff_castor, 'C2-.', label='Castor Ptcles')
ax[0].plot(CL_castor_Flts[:,0], CL_castor_Flts[:,3], 'C3-.', label='Castor Flts')

ax[0].legend()
ax[0].set_xlabel('Blade center positions (m)')
ax[0].set_ylabel('Fn (N/m)')
ax[0].grid()



blade_centers_vortex = CL_vortex[:, 0]
lift_coeff_vortex = CL_vortex[:, 2]

blade_centers_castor = 0.21 + castor[:, 0] * span
lift_coeff_castor = castor[:, 16]

ax[1].plot(blade_centers_vortex, +lift_coeff_vortex, 'C1--', label='PITCHOU')
ax[1].plot(blade_centers_castor, -lift_coeff_castor, 'C2-.', label='Castor Ptcles')
#CL_castor = np.genfromtxt('/work/blondelf/work/Calculs/MexNext/MexNext3_Axial_Third_Round_February_2017/To_MexNext/IFPEN_VL_2D/Loads_CASTOR.dat', skip_header=2)
ax[1].plot(CL_castor_Flts[:,0], CL_castor_Flts[:,4], 'C4-.', label='Castor Flts')

ax[1].legend()
ax[1].set_xlabel('Blade center positions (m)')
ax[1].set_ylabel('Ft (N/m)')
ax[1].grid()
plt.savefig('bladeForces.png', dpi=450, format='png')
plt.show()
