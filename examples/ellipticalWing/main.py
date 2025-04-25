import os
import sys
# Get the directory containing the examples folder
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)
sys.path.append(parent_of_project_dir)

from sven.windTurbine import *
from sven.airfoil import *
from sven.blade import *
from sven.solver import *
from utils.io import *
from utils.definitions import *

outDir = 'outputs'
if not os.path.exists(outDir):
    os.makedirs(outDir)
else:
    print("Directory ", outDir, " already exists")

bladePitch = 5.
nBladeCenters = 40
AR = 6.
bladeLength = 10.
nearWakeLength = 100

uInfty = 1.
deltaFlts = np.sqrt(1e-1)


Blades = Wing(bladePitch, nBladeCenters, AR, bladeLength, nearWakeLength)


timeStep = 0.1
timeEnd  = nearWakeLength*timeStep
innerIter = 10
timeSteps = np.arange(0., timeEnd, timeStep)

deltaPtcles = 1e-4
partsPerFil = 1

eps_conv = 1e-4
timeSimulation = 0.
iterationVect = []
startTime = time.time()
for (it, t) in enumerate(timeSteps):

    print('iteration, time, finaltime: ', it, t, timeSteps[-1])
    timeSimulation += timeStep
    update(Blades, uInfty, timeStep, timeSimulation, innerIter, deltaFlts, startTime, iterationVect)
    
    postProcess = False
    if(postProcess):
        write_blade_tp(Blades, outDir)
        write_filaments_tp(Blades, outDir)

    output = open('./outputs/liftDistribution_elliptical.dat', 'w')
    centers = Blades[0].centers
    liftDistribution = Blades[0].lift
    for i in range(len(centers)):
        output.write(str(centers[i][1]) + ' ' + str(liftDistribution[i])  + '\n')
    output.close()        

    

