import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)
sys.path.append(parent_of_project_dir)

from sven.windTurbine import *
from sven.airfoil import *
from sven.blade import *
from sven.solver import *

from scipy import interpolate

# Post-processing directory
outDir = 'outputs'
if not os.path.exists(outDir):
    os.makedirs(outDir)
else:
    print("Directory ", outDir, " already exists")



def NewMexicoWindTurbine(windVelocity,density,nearWakeLength):
    sign = -1.
    hubRadius = 0.210
    nBlades = 3
    rotationalVelocity = 44.5163679
    bladePitch = sign * 0.040143
    
    dataAirfoils = np.genfromtxt(
        './geometry/mexico.blade', skip_header=1, usecols=(7), dtype='U')
    intAirfoils = np.arange(0, len(dataAirfoils))

    data = np.genfromtxt('./geometry/mexico.blade', skip_header=1)

    refRadius = data[:,2] 

    inputNodesTwistAngles = -sign * np.radians(data[:, 5])
    inputNodesChord = data[:, 6]

    f = interpolate.interp1d(data[:, 2], intAirfoils, kind='nearest')
    centersAirfoils = []
    for i in range(len(refRadius)):
        foilName = str(dataAirfoils[int(f(refRadius[i]))])
        centersAirfoils.append(Airfoil('./geometry/' + foilName, headerLength=1))

    nodesRadius = hubRadius + refRadius
    nodesTwistAngles = np.interp(refRadius, data[:, 2], inputNodesTwistAngles)
    nodesChord = np.interp(refRadius, data[:, 2], inputNodesChord)

    # Build wind turbine
    hubCenter = [0., 0., 0.]
    myWT = windTurbine(
        nBlades, hubCenter, hubRadius, rotationalVelocity, 
        windVelocity, bladePitch)
    blades = myWT.initializeTurbine(
        nodesRadius, nodesChord, nearWakeLength, centersAirfoils, 
        nodesTwistAngles, myWT.nBlades)

    return blades, myWT, windVelocity, density, 0.01, 1e-4

# -----------------------------------------------------------------------------
# Some functions for filament outputs 
# -----------------------------------------------------------------------------

def write_filaments_tp(blades, outDir, it):

    for (iBlade, blade) in enumerate(blades):
        shape = np.shape(blade.wakeNodes)

        output = open(
            outDir + '/Filaments_Nodes_' + '_Blade_'+str(iBlade)+'_tStep_'+
            str(it)+'.tp', 'w')
        output.write('TITLE="Near-wake nodes"\n')
        output.write('VARIABLES="X" "Y" "Z" "Circulation"\n')
        output.write(
            'ZONE T="Near-wake" I='+str(shape[0])+' J='+str(shape[1]-1)+
            ', K=1, DT=(SINGLE SINGLE SINGLE SINGLE)\n')
        for j in range(np.shape(blade.wakeNodes)[1]-1):
            for i in range(np.shape(blade.wakeNodes)[0]):
                output.write(
                    str(blade.wakeNodes[i,j,0]) + " " + 
                    str(blade.wakeNodes[i,j,1]) + " " + 
                    str(blade.wakeNodes[i,j,2]) + " " +
                    str(blade.trailFilamentsCirculation[i,j]) + "\n")
        output.close()

    return

def write_blade_tp(blades, outDir, it):

    for (iBlade, blade) in enumerate(blades):
        shape = len(blade.bladeNodes)

        output = open(
            outDir + '/Blade_'+str(iBlade)+'_Nodes_tStep_'+str(it)+'.tp', 'w')
        output.write('TITLE="Near-wake nodes"\n')
        output.write('VARIABLES="X" "Y" "Z"\n')
        output.write(
            'ZONE T="Near-wake" I='+str(shape)+' J='+str(2)+
            ', K=1, DT=(SINGLE SINGLE SINGLE)\n')
        for i in range(shape):
                output.write(
                    str(blade.bladeNodes[i,0]-1./4.*blade.nodeChords[i]) + " " + 
                    str(blade.bladeNodes[i,1]) + " " + 
                    str(blade.bladeNodes[i,2]) + "\n")
        for i in range(shape):
                output.write(
                    str(blade.trailingEdgeNode[i,0]) + " " + 
                    str(blade.trailingEdgeNode[i,1]) + " " + 
                    str(blade.trailingEdgeNode[i,2]) + "\n")
        output.close()
    return

# -----------------------------------------------------------------------------
# Choose a wind velocity and wake parameters 
# -----------------------------------------------------------------------------

cases = ["15"]#, "15", "24"]

nearWakeLength = 3600
innerIter  = 12
nRotations = 10.
DegreesPerTimeStep = 10.

# Choose if you want post process files to be written 

wakePostProcess = True
forcesPostProcess = True


for caseID in cases:
    if(caseID == "10"):
        windVelocity = 10.05
        density = 1.197
    if(caseID == "15"):
        windVelocity = 15.06
        density = 1.191
    if(caseID == "24"):
        windVelocity = 24.05
        density = 1.195
    Blades, WindTurbine, uInfty, density, deltaFlts, deltaPtcles = (
         NewMexicoWindTurbine(windVelocity,density,nearWakeLength))
    
    timeStep = np.radians(DegreesPerTimeStep) / WindTurbine.rotationalVelocity
    
    timeEnd = np.radians(nRotations * 360.) / WindTurbine.rotationalVelocity
    refAzimuth = -WindTurbine.rotationalVelocity * timeStep
    timeSteps = np.arange(0., timeEnd, timeStep)

# -----------------------------------------------------------------------------
# Time loop 
# -----------------------------------------------------------------------------

    timeSimulation = 0.
    iterationVect = []
    startTime = time.time()

    for (it, t) in enumerate(timeSteps):

        refAzimuth += WindTurbine.rotationalVelocity * timeStep
        WindTurbine.updateTurbine(refAzimuth)

        print('iteration, time, finaltime: ', it, t, timeSteps[-1])
        partsPerFil = 1
        timeSimulation += timeStep
        update(
            Blades, uInfty, timeStep, timeSimulation, innerIter, deltaFlts, 
            startTime, iterationVect)

        

        if(wakePostProcess):
            write_blade_tp(Blades, outDir, it)
            write_filaments_tp(Blades, outDir, it)

        if(forcesPostProcess):
            centers = Blades[0].centers

            Fn, Ft = WindTurbine.evaluateForces(density)
            output = open(
                'outputs/bladeForces_case_'+caseID+'_'+str(it)+'.dat', 'w')
            for i in range(len(centers)):
                output.write(
                    str(np.linalg.norm(centers[i])) + ' ' + str(Fn[i]) + ' ' +
                    str(Ft[i]) + '\n')
            output.close()

            output = open('outputs/liftDistribution_case_'+caseID+'.dat', 'w')
            liftDistribution = Blades[0].lift
            for i in range(len(centers)):
                TwistAndPitch = WindTurbine.bladePitch + .5 * (
                    WindTurbine.nodesTwistAngles[i] + 
                    WindTurbine.nodesTwistAngles[i + 1])
                aoa_th = np.degrees(
                    np.arctan2(
                        uInfty, 
                        WindTurbine.rotationalVelocity * centers[i][1]
                        ) - TwistAndPitch)
                output.write(
                    str(np.linalg.norm(centers[i])) + ' ' +
                    str(liftDistribution[i]) + ' ' +
                    str(np.degrees(Blades[0].attackAngle[i])) + ' ' +
                    str(Blades[0].effectiveVelocity[i]) + '\n')
            output.close()

    print('Total simulation time: ', time.time() - startTime)


