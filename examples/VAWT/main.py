import os
import matplotlib.pyplot as plt

from sven.windTurbine import *
from sven.airfoil import *
from sven.blade import *
from sven.solver import *
from utils.io import *

from scipy import interpolate

# Post-processing directory
# Create target directory & all intermediate directories if don't exists
outDir = 'outputs'
if not os.path.exists(outDir):
    os.makedirs(outDir)
else:
    print("Directory ", outDir, " already exists")


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



def writeHubAndTower():
    xBase = +10.
    radius = 2.5
    z = [-120.,0.]

    thetas = np.linspace(0., 2.*np.pi, 30)
    x = radius*np.cos(thetas) + xBase
    y = radius*np.sin(thetas)

    out = open('outputs/cylinderTower.tp', 'w')
    out.write('TITLE="Near-wake nodes"\n')
    out.write('VARIABLES="X" "Y" "Z"\n')
    out.write('ZONE T="Near-wake" I=30 J=1, K=2, DT=(SINGLE SINGLE SINGLE)\n')
    for zi in z:
        for (yi, xi) in zip(y, x):
            out.write(str(xi)+' '+str(yi)+' '+str(zi)+'\n')
    out.close()


    xBase = +0.
    radius = 3.5
    z = [-0.25,10.+radius]
    thetas = np.linspace(0., 2.*np.pi, 30)
    x = radius*np.cos(thetas) + xBase
    y = radius*np.sin(thetas)   

    out = open('outputs/cylinderHub.tp', 'w')
    out.write('TITLE="Near-wake nodes"\n')
    out.write('VARIABLES="X" "Y" "Z"\n')
    out.write('ZONE T="Near-wake" I=2 J=30, K=30, DT=(SINGLE SINGLE SINGLE)\n')
    for zi in z:    
        # Not the way it is supposed to be but works
        for (yi, xi) in zip(y, x):
            out.write(str(zi)+' '+str(xi)+' '+str(yi)+'\n')
            out.write(str(zi)+' '+str(xBase)+' '+str(yi)+'\n')       
    out.close()
    return

def VAWT():

    nbNodes = 50
    sign = -1.
    nBlades = 3
    rotationalVelocity = 9.6 * np.pi / 30.
    windVelocity = 11.4
    bladePitch = 0.0
    nearWakeLength = 359
    rotorRadius = 10.
    bLength = 20.
    bChord = 1.

    centersAirfoils = []
    for i in range(nbNodes):
        centersAirfoils.append(Airfoil('../../turbineModels/DTU10MW/reference_files/airfoil.foil', headerLength=1))

    nodesRadius = np.linspace(0., bLength, nbNodes, endpoint=True)
    nodesTwistAngles = np.zeros(nbNodes) #np.interp(refRadius, data[:, 2], inputNodesTwistAngles)
    nodesChord = np.ones(nbNodes) * bChord #np.interp(refRadius, data[:, 2], inputNodesChord)

    # Build wind turbine
    hubCenter = [0., 0., 0.]
    myWT = VAWT_windTurbine(nBlades, hubCenter, hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, myWT.nBlades)

    return blades, myWT, windVelocity, 0.01, 1e-4


def VAWT_rotor(bladePitch, bladeLength, rotorRadius, rotationSpeed, nBlades, wingType = "Rectangular"):

    if (wingType != "Elliptical" and wingType != "Rectangular"):
        print("Non-existing wing type: ", wingType, " please use \"Elliptical\" or \"Rectangular\"")

    # Blade discretisation
    nBladeCenters = 50
    nearWakeLength = 1000

    #bladeLength = 30.
    if (wingType == "Elliptical"):
        AR = 8.
        cRoot = 4 * bladeLength / (AR * np.pi)
    elif (wingType == "Rectangular"):
        P = 6.
        # P = 2. / np.pi * bladeLength / cRoot
        cRoot = 2. / np.pi * bladeLength / P
        # cRoot = bladeLength / AR

    cRoot = 4. / nBlades
    uInfty = 6.

    nodeChords = np.ones(nBladeCenters + 1) * cRoot
    airfoils = []
    for i in range(nBladeCenters):
        airfoils.append(Airfoil('../../data/flatPlate.foil'))

    # Multiple straight wings
    initialNodes = np.zeros([nBladeCenters + 1, 3])
    initialNodes[:, 2] = (np.linspace(0., 1., nBladeCenters + 1) - 0.5) * bladeLength
    initialNodes[:, 1] = rotorRadius

    Blades = []
    dAz = 2.*np.pi / nBlades
    for ib in range(nBlades):

        azimuth = ib * dAz

        print('INITIAL BLADE NODES: ', initialNodes)

        nodes = np.copy(initialNodes)
        # Rotate them with the azimuth
        for i in range(len(initialNodes[:,0])):

            x = initialNodes[i,0]
            y = initialNodes[i,1]

            nodes[i,0] = x * np.cos(azimuth) - y * np.sin(azimuth)
            nodes[i,1] = x * np.sin(azimuth) + y * np.cos(azimuth)
            print(i, x, y, nodes[i,0], nodes[i,1])

        # print('nodes: ', nodes, ' azimuth= ', np.degrees(azimuth))
        # print()

        bladePitch = 0.
        centersOrientationMatrix = np.zeros([len(nodes) - 1, 3, 3])
        for i in range(len(nodes) - 1):
            r1 = R.from_euler('x', 90., degrees=True)
            r2 = R.from_euler('y', bladePitch+azimuth, degrees=True)
            R1 = r1.as_matrix()
            R2 = r2.as_matrix()

            centersOrientationMatrix[i] = np.dot(R1, R2)

        nodesOrientationMatrix = np.zeros([len(nodes), 3, 3])
        for i in range(len(nodes)):
            r1 = R.from_euler('x', 90., degrees=True)
            r2 = R.from_euler('y', bladePitch+azimuth, degrees=True)
            R1 = r1.as_matrix()
            R2 = r2.as_matrix()
            nodesOrientationMatrix[i] = np.dot(R1, R2)

        liftingLine = Blade(nodes, nodeChords, nearWakeLength, airfoils, centersOrientationMatrix, nodesOrientationMatrix,
                             np.zeros([len(nodes) - 1, 3]), np.zeros([len(nodes), 3]))
        print('FINAL BLADE NODES: ', liftingLine.bladeNodes)
        liftingLine.updateFirstWakeRow()
        liftingLine.initializeWake()

        Blades.append(liftingLine)

    updateWing(Blades, 0, 0, 0., rotationSpeed)

    deltaFlts = 1e-2 #np.sqrt(1e-2)

    print('Solidity: ', nBlades * nodeChords[0] / 2. / rotorRadius)
    print('TSR     : ', 9.*np.pi/30 * rotorRadius / uInfty)
    input()

    return Blades, uInfty, deltaFlts

def updateWing(Blades, iteration, time, azimuth, rotationSpeed):
    #
    nBlades = len(Blades)
    #
    print('########### Azimuth, attack angle: ', np.degrees(azimuth), np.degrees(Blades[0].attackAngle[25]))
    #
    n0 = Blades[0].bladeNodes[0]
    nE = Blades[0].bladeNodes[-1]
    bladeLength = nE[2] - n0[2]
    #
    n0[2] = 0.
    rotorRadius = np.linalg.norm(n0)
    #
    elementVelocity = np.zeros(3)
    elementVelocity[0] = - rotationSpeed * rotorRadius

    for (ib,blade) in enumerate(Blades):

        dAz = ib * 2.*np.pi / nBlades

        # Multiple straight wings
        nCenters = len(blade.centers)
        nodes = np.zeros([nCenters + 1, 3])

        # z positions do not change
        nodes[:, 2] = (np.linspace(0., 1., nCenters + 1) - 0.5) * bladeLength
        # initial positions:
        nodes[:, 1] = rotorRadius
        nodes[:, 0] = 0.
        # Rotate them with the azimuth
        for i in range(len(nodes[:,0])):
            x = nodes[i,0]
            y = nodes[i,1]

            nodes[i,0] = x * np.cos(azimuth+dAz) - y * np.sin(azimuth+dAz)
            nodes[i,1] = x * np.sin(azimuth+dAz) + y * np.cos(azimuth+dAz)

        blade.bladeNodes = nodes
        blade.centers = .5 * (nodes[1:] + nodes[:-1])


        # Update orientation matrices and velocities
        centersOrientationMatrix = np.zeros([len(nodes) - 1, 3, 3])
        centersVelocities = np.zeros([len(nodes) - 1, 3])
        for i in range(len(nodes) - 1):
            r1 = R.from_euler('x', 90., degrees=True)
            r2 = R.from_euler('y', np.degrees(azimuth), degrees=True)
            R1 = r1.as_matrix()
            R2 = r2.as_matrix()
            centersOrientationMatrix[i] = np.dot(R1, R2)

            r = R.from_matrix(centersOrientationMatrix[i])
            centersVelocities[i,:] = r.apply(elementVelocity, inverse=False)
        blade.centersOrientationMatrix = centersOrientationMatrix
        blade.centersTranslationVelocity = centersVelocities

        nodesOrientationMatrix = np.zeros([len(nodes), 3, 3])
        nodesVelocities = np.zeros([len(nodes), 3])
        for i in range(len(nodes)):
            r1 = R.from_euler('x', 90., degrees=True)
            r2 = R.from_euler('y', np.degrees(azimuth), degrees=True)
            R1 = r1.as_matrix()
            R2 = r2.as_matrix()
            nodesOrientationMatrix[i] = np.dot(R1, R2)

            r = R.from_matrix(nodesOrientationMatrix[i])
            nodesVelocities[i,:] = r.apply(elementVelocity, inverse=False)
        blade.nodesOrientationMatrix = nodesOrientationMatrix
        blade.nodesTranslationVelocities = nodesVelocities

    return

windTurbineCase = False

if(windTurbineCase == True):
    Blades, WindTurbine, uInfty, deltaFlts, deltaPtcles = VAWT()
    DegreesPerTimeStep = 10.0
    timeStep = np.radians(
        DegreesPerTimeStep) / WindTurbine.rotationalVelocity
    innerIter  = 4
    nRotations = 10.
    timeEnd = np.radians(nRotations * 360.) / WindTurbine.rotationalVelocity
    eps_conv = 1e-4

    refAzimuth = -WindTurbine.rotationalVelocity * timeStep

    timeSteps = np.arange(0., timeEnd, timeStep)
else:
    nBlades = 3 # input, integer
    rotationRPM = 9. # input
    nRotations = 4.  # input
    rotorRadius = 20. # input
    rotationSpeed = rotationRPM * np.pi / 30.
    timeOneRotation = 60. / rotationRPM
    timeStep = timeOneRotation / 36.
    timeEnd  = timeOneRotation * nRotations
    innerIter = 10
    timeSteps = np.arange(0., timeEnd, timeStep)
    bladePitch = 0.
    bladeLength = 30.
    Blades, uInfty, deltaFlts = VAWT_rotor(bladePitch, bladeLength, rotorRadius, rotationSpeed, nBlades, "Rectangular")

    deltaPtcles = 1e-4

eps_conv = 1e-4

timeSimulation = 0.
azimuth = 0.

azims = []
midBladeAoA = []
startTime = time.time()
for (it, t) in enumerate(timeSteps):

    azimuth = it * rotationSpeed * timeStep

    myAzimuth = np.degrees(azimuth) % 360.
    print("###############################", myAzimuth)
    azims . append(myAzimuth)
    midBladeAoA . append( np.degrees(Blades[0].attackAngle[25]))

    print('azimuth: ', np.degrees(azimuth))

    if(windTurbineCase == True):
        refAzimuth += WindTurbine.rotationalVelocity * timeStep
        WindTurbine.updateTurbine(refAzimuth)

    updateWing(Blades, it, time, azimuth, rotationSpeed)

    print('iteration, time, finaltime: ', it, t, timeSteps[-1])
    partsPerFil = 1
    timeSimulation += timeStep
    update(Blades, uInfty, timeStep, timeSimulation, innerIter, deltaFlts, startTime, iterationVect)

    postProcess = True
    if(postProcess):
        if(it == 0):
            writeHubAndTower()
            
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
            output = open('outputs/liftDistribution_elliptical.dat', 'w')
            centers = Blades[0].centers
            liftDistribution = Blades[0].lift
            for i in range(len(centers)):
                output.write(str(centers[i][1]) + ' ' + str(liftDistribution[i])  + '\n')
            output.close()
print('Total simulation time: ', time.time() - startTime)

plt.plot(azims, midBladeAoA, 'o-')
plt.savefig('aoaEvolution.png', format='png', dpi=200)
