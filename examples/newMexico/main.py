import os

from pitchou.windTurbine import *
from pitchou.airfoil import *
from pitchou.blade import *
from pitchou.solver import *
from scipy import interpolate

# Post-processing directory
# Create target directory & all intermediate directories if don't exists
outDir = 'outputs'
if not os.path.exists(outDir):
    os.makedirs(outDir)
else:
    print("Directory ", outDir, " already exists")


def NewMexicoWindTurbine():
    sign = -1.
    hubRadius = 0.210
    nBlades = 3
    rotationalVelocity = 44.5163679
    windVelocity = 15.06 #10.05 #15.06
    bladePitch = sign * 0.040143
    nearWakeLength = 359

    dataAirfoils = np.genfromtxt('../../turbineModels/NewMexico/reference_files/mexico.blade', skip_header=1, usecols=(7),
                                 dtype='U')
    intAirfoils = np.arange(0, len(dataAirfoils))

    data = np.genfromtxt('../../turbineModels/NewMexico/reference_files/mexico.blade', skip_header=1)

    refRadius = data[:,2] #np.linspace(0., 2.04, 45) # data[:,2]

    # Radius start at blade root, not hub center!
    inputNodesTwistAngles = -sign * np.radians(data[:, 5])
    inputNodesChord = data[:, 6]

    f = interpolate.interp1d(data[:, 2], intAirfoils, kind='nearest')
    centersAirfoils = []
    for i in range(len(refRadius)):
        foilName = str(dataAirfoils[int(f(refRadius[i]))])
        centersAirfoils.append(Airfoil('../../turbineModels/NewMexico/reference_files/' + foilName, headerLength=1))

    nodesRadius = hubRadius + refRadius
    nodesTwistAngles = np.interp(refRadius, data[:, 2], inputNodesTwistAngles)
    nodesChord = np.interp(refRadius, data[:, 2], inputNodesChord)

    # Build wind turbine
    hubCenter = [0., 0., 0.]
    myWT = windTurbine(nBlades, hubCenter, hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)

    return blades, myWT, windVelocity, 0.01, 1e-4


def EllipticalWing(bladePitch):
    # Blade discretisation
    nBladeCenters = 10
    nearWakeLength = 10

    AR = 6.
    bladeLength = 10.
    cRoot = 4 * bladeLength / (AR * np.pi)
    uInfty = 1.

    # Multiple straight wings
    nodes = np.zeros([nBladeCenters + 1, 3])
    nodes[:, 1] = (np.linspace(0., 1., nBladeCenters + 1) - 0.5) * bladeLength

    nodeChords = np.zeros(len(nodes))
    for i in range(len(nodes)):
        nodeChords[i] = np.sqrt(np.abs(cRoot ** 2. * (1. - 4. * (nodes[i, 1] / bladeLength) ** 2.)))

    airfoils = []
    for i in range(len(nodes) - 1):
        airfoils.append(Airfoil('../data/flatPlate.foil'))

    centersOrientationMatrix = np.zeros([len(nodes) - 1, 3, 3])
    for i in range(len(nodes) - 1):
        r = R.from_euler('y', bladePitch, degrees=True)
        centersOrientationMatrix[i] = r.as_matrix()

    nodesOrientationMatrix = np.zeros([len(nodes), 3, 3])
    for i in range(len(nodes)):
        r = R.from_euler('y', bladePitch, degrees=True)
        nodesOrientationMatrix[i] = r.as_matrix()

    liftingLine1 = Blade(nodes, nodeChords, nearWakeLength, airfoils, centersOrientationMatrix, nodesOrientationMatrix,
                         np.zeros([len(nodes) - 1, 3]), np.zeros([len(nodes), 3]))
    liftingLine2 = Blade(nodes+[0., 0., 10.], nodeChords, nearWakeLength, airfoils, centersOrientationMatrix, nodesOrientationMatrix,
                         np.zeros([len(nodes) - 1, 3]), np.zeros([len(nodes), 3]))

    liftingLine1.updateFirstWakeRow()
    liftingLine1.initializeWake()

    liftingLine2.updateFirstWakeRow()
    liftingLine2.initializeWake()

    Blades = []
    Blades.append(liftingLine1)
    Blades.append(liftingLine2)

    deltaFlts = np.sqrt(1e-2)

    return Blades, uInfty, deltaFlts


def write_particles(outDir):
    output = open(outDir + '/New_Particles_tStep_' + str(it) + '.particles', 'w')
    for i in range(len(wake.particlesPositionX)):
        output.write(str(wake.particlesPositionX[i]) + " " + str(wake.particlesPositionY[i]) + " " + str(
            wake.particlesPositionZ[i]) + " " + "\n")
    output.close()

    return

def write_filaments_tp(blades, outDir):

    for (iBlade, blade) in enumerate(blades):
        shape = np.shape(blade.wakeNodes)

        output = open(outDir + '/Filaments_Nodes_' + '_Blade_'+str(iBlade)+'_tStep_'+str(it)+'.tp', 'w')
        output.write('TITLE="Near-wake nodes"\n')
        output.write('VARIABLES="X" "Y" "Z" "Circulation"\n')
        output.write('ZONE T="Near-wake" I='+str(shape[0])+' J='+str(shape[1]-1)+', K=1, DT=(SINGLE SINGLE SINGLE SINGLE)\n')
        for j in range(np.shape(blade.wakeNodes)[1]-1):
            for i in range(np.shape(blade.wakeNodes)[0]):
                output.write(str(blade.wakeNodes[i,j,0]) + " " + str(blade.wakeNodes[i,j,1]) + " " + str(blade.wakeNodes[i,j,2]) + " " +str(blade.trailFilamentsCirculation[i,j]) + "\n")
        output.close()

    return

def write_blade_tp(blades, outDir):

    for (iBlade, blade) in enumerate(blades):
        shape = len(blade.bladeNodes)

        output = open(outDir + '/Blade_'+str(iBlade)+'_Nodes_tStep_'+str(it)+'.tp', 'w')
        output.write('TITLE="Near-wake nodes"\n')
        output.write('VARIABLES="X" "Y" "Z"\n')
        output.write('ZONE T="Near-wake" I='+str(shape)+' J='+str(2)+', K=1, DT=(SINGLE SINGLE SINGLE)\n')
        for i in range(shape):
                output.write(str(blade.bladeNodes[i,0]) + " " + str(blade.bladeNodes[i,1]) + " " + str(blade.bladeNodes[i,2]) + "\n")
        for i in range(shape):
                output.write(str(blade.trailingEdgeNode[i,0]) + " " + str(blade.trailingEdgeNode[i,1]) + " " + str(blade.trailingEdgeNode[i,2]) + "\n")
        output.close()
    return


windTurbineCase = True
# Blades, WindTurbine, uInfty, deltaFlts = NewMexicoWindTurbine()  # EllipticalWing() #StraightWingCastor() #StraightBlade() #EllipticalWing()

if(windTurbineCase == True):
    Blades, WindTurbine, uInfty, deltaFlts, deltaPtcles = NewMexicoWindTurbine()
    DegreesPerTimeStep = 10.
    timeStep = np.radians(
        DegreesPerTimeStep) / WindTurbine.rotationalVelocity
    innerIter  = 12
    nRotations = 10.
    timeEnd = np.radians(nRotations * 360.) / WindTurbine.rotationalVelocity
    eps_conv = 1e-4

    refAzimuth = -WindTurbine.rotationalVelocity * timeStep

    timeSteps = np.arange(0., timeEnd, timeStep)
else:
    timeStep = 0.1
    timeEnd  = 10.
    innerIter = 10
    timeSteps = np.arange(0., timeEnd, timeStep)
    Blades, uInfty, deltaFlts = EllipticalWing(5.)
    deltaPtcles = 1e-4

wake = Wake()

eps_conv = 1e-4

timeSimulation = 0.

startTime = time.time()
for (it, t) in enumerate(timeSteps):

    if(windTurbineCase == True):
        refAzimuth += WindTurbine.rotationalVelocity * timeStep
        WindTurbine.updateTurbine(refAzimuth)

    print('iteration, time, finaltime: ', it, t, timeSteps[-1])
    partsPerFil = 1
    timeSimulation += timeStep
    update(Blades, wake, uInfty, timeStep, timeSimulation, innerIter, deltaFlts, deltaPtcles, eps_conv, partsPerFil)

    postProcess = False
    if(postProcess):
        write_particles(outDir)
        write_blade_tp(Blades, outDir)
        write_filaments_tp(Blades, outDir)

        if (windTurbineCase == True):
            centers = Blades[0].centers

            Fn, Ft = WindTurbine.evaluateForces(1.191) #(1.197)
            output = open('outputs/bladeForces_'+str(it)+'.dat', 'w')
            for i in range(len(centers)):
                output.write(str(np.linalg.norm(centers[i])) + ' ' + str(Fn[i]) + ' ' + str(Ft[i]) + '\n')
            output.close()

            output = open('outputs/liftDistribution.dat', 'w')
            liftDistribution = Blades[0].lift
            for i in range(len(centers)):
                if (windTurbineCase == True):
                    TwistAndPitch = WindTurbine.bladePitch + .5 * (
                            WindTurbine.nodesTwistAngles[i] + WindTurbine.nodesTwistAngles[i + 1])
                    aoa_th = np.degrees(np.arctan2(uInfty, WindTurbine.rotationalVelocity * centers[i][1]) - TwistAndPitch)
                else:
                    output = open('outputs/liftDistribution.dat', 'w')
                    aoa_th = 0.
                output.write(str(np.linalg.norm(centers[i])) + ' ' + str(liftDistribution[i]) + ' ' + str(
                    np.degrees(Blades[0].attackAngle[i])) + ' ' + str(Blades[0].effectiveVelocity[i]) + '\n')
            output.close()
        else :
            output = open('liftDistribution_elliptical.dat', 'w')
            centers = Blades[0].centers
            liftDistribution = Blades[0].lift
            for i in range(len(centers)):
                output.write(str(centers[i][1]) + ' ' + str(liftDistribution[i])  + '\n')
            output.close()
print('Total simulation time: ', time.time() - startTime)


