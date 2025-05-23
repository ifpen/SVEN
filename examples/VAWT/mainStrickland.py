import os
import matplotlib.pyplot as plt

from sven.airfoil import *
from sven.blade import *
from sven.solver import *

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
    z = [-120., 0.]

    thetas = np.linspace(0., 2. * np.pi, 30)
    x = radius * np.cos(thetas) + xBase
    y = radius * np.sin(thetas)

    out = open('outputs/cylinderTower.tp', 'w')
    out.write('TITLE="Near-wake nodes"\n')
    out.write('VARIABLES="X" "Y" "Z"\n')
    out.write('ZONE T="Near-wake" I=30 J=1, K=2, DT=(SINGLE SINGLE SINGLE)\n')
    for zi in z:
        for (yi, xi) in zip(y, x):
            out.write(str(xi) + ' ' + str(yi) + ' ' + str(zi) + '\n')
    out.close()

    xBase = +0.
    radius = 3.5
    z = [-0.25, 10. + radius]
    thetas = np.linspace(0., 2. * np.pi, 30)
    x = radius * np.cos(thetas) + xBase
    y = radius * np.sin(thetas)

    out = open('outputs/cylinderHub.tp', 'w')
    out.write('TITLE="Near-wake nodes"\n')
    out.write('VARIABLES="X" "Y" "Z"\n')
    out.write('ZONE T="Near-wake" I=2 J=30, K=30, DT=(SINGLE SINGLE SINGLE)\n')
    for zi in z:
        # Not the way it is supposed to be but works
        for (yi, xi) in zip(y, x):
            out.write(str(zi) + ' ' + str(xi) + ' ' + str(yi) + '\n')
            out.write(str(zi) + ' ' + str(xBase) + ' ' + str(yi) + '\n')
    out.close()
    return

def CosineNodesDistribution(nPoints):
    xis = np.linspace(-np.pi, 0., nPoints + 1)
    return .5 * (np.cos(xis) + 1.)


def VAWT_rotor(bladePitch, distribution, bladeLength, rotorRadius, rotationSpeed, nBlades, windVelocity, nearWakeLength):

    # Blade discretisation
    nBladeCenters  = 50

    # Blade chord
    nodeChords = np.ones(nBladeCenters + 1) * 0.0914

    airfoils = []
    for i in range(nBladeCenters):
        airfoils.append(Airfoil('./geometry/stricklandAirfoil.foil'))

    # Multiple straight wings
    initialNodes = np.zeros([nBladeCenters + 1, 3])
    if distribution == "linear":
        initialNodes[:, 2] = (np.linspace(0., 1., nBladeCenters + 1) - 0.5) * bladeLength
    else:
        initialNodes[:, 2] = np.asarray((CosineNodesDistribution(nBladeCenters) - 0.5) * bladeLength)
    initialNodes[:, 1] = rotorRadius

    Blades = []
    dAz = 2. * np.pi / nBlades
    for ib in range(nBlades):

        azimuth = ib * dAz

        nodes = np.copy(initialNodes)
        # Rotate the nodes with the azimuth
        for i in range(len(initialNodes[:, 0])):
            x = initialNodes[i, 0]
            y = initialNodes[i, 1]

            nodes[i, 0] = x * np.cos(azimuth) - y * np.sin(azimuth)
            nodes[i, 1] = x * np.sin(azimuth) + y * np.cos(azimuth)

        bladePitch = 0.
        centersOrientationMatrix = np.zeros([len(nodes) - 1, 3, 3])
        for i in range(len(nodes) - 1):
            r1 = R.from_euler('x', 90., degrees=True)
            r2 = R.from_euler('y', bladePitch + azimuth, degrees=True)
            R1 = r1.as_matrix()
            R2 = r2.as_matrix()

            centersOrientationMatrix[i] = np.dot(R1, R2)

        nodesOrientationMatrix = np.zeros([len(nodes), 3, 3])
        for i in range(len(nodes)):
            r1 = R.from_euler('x', 90., degrees=True)
            r2 = R.from_euler('y', bladePitch + azimuth, degrees=True)
            R1 = r1.as_matrix()
            R2 = r2.as_matrix()
            nodesOrientationMatrix[i] = np.dot(R1, R2)

        liftingLine = Blade(nodes, nodeChords, nearWakeLength, airfoils, centersOrientationMatrix,
                            nodesOrientationMatrix,
                            np.zeros([len(nodes) - 1, 3]), np.zeros([len(nodes), 3]))
        liftingLine.updateFirstWakeRow()
        liftingLine.initializeWake()

        Blades.append(liftingLine)

    updateWing(Blades, distribution, 0, 0, 0., rotationSpeed)

    deltaFlts = 1e-1

    print('Solidity: ', nBlades * nodeChords[0] / 2. / rotorRadius)
    print('TSR     : ', rotationSpeed * rotorRadius / windVelocity)

    return Blades, deltaFlts


def updateWing(Blades, distribution, iteration, time, azimuth, rotationSpeed):
    #
    nBlades = len(Blades)
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

    for (ib, blade) in enumerate(Blades):

        dAz = ib * 2. * np.pi / nBlades

        # Multiple straight wings
        nCenters = len(blade.centers)
        nodes = np.zeros([nCenters + 1, 3])

        # z positions do not change
        if distribution == "linear":
            nodes[:, 2] = (np.linspace(0., 1., nCenters + 1) - 0.5) * bladeLength
        else:
            nodes[:, 2] = np.asarray((CosineNodesDistribution(nCenters) - 0.5) * bladeLength)

        # initial positions:
        nodes[:, 1] = rotorRadius
        nodes[:, 0] = 0.
        # Rotate them with the azimuth
        for i in range(len(nodes[:, 0])):
            x = nodes[i, 0]
            y = nodes[i, 1]

            nodes[i, 0] = x * np.cos(azimuth + dAz) - y * np.sin(azimuth + dAz)
            nodes[i, 1] = x * np.sin(azimuth + dAz) + y * np.cos(azimuth + dAz)

        Blades[ib].bladeNodes = nodes
        Blades[ib].centers = .5 * (nodes[1:] + nodes[:-1])

        # Update orientation matrices and velocities
        centersOrientationMatrix = np.zeros([len(nodes) - 1, 3, 3])
        centersVelocities = np.zeros([len(nodes) - 1, 3])
        for i in range(len(nodes) - 1):
            r1 = R.from_euler('x', 90., degrees=True)
            r2 = R.from_euler('y', np.degrees(azimuth + dAz), degrees=True)
            R1 = r1.as_matrix()
            R2 = r2.as_matrix()
            centersOrientationMatrix[i] = np.dot(R1, R2)

            r = R.from_matrix(centersOrientationMatrix[i])
            centersVelocities[i, :] = r.apply(elementVelocity, inverse=False)
        Blades[ib].centersOrientationMatrix = centersOrientationMatrix
        Blades[ib].centersTranslationVelocity = centersVelocities

        nodesOrientationMatrix = np.zeros([len(nodes), 3, 3])
        nodesVelocities = np.zeros([len(nodes), 3])
        for i in range(len(nodes)):
            r1 = R.from_euler('x', 90., degrees=True)
            r2 = R.from_euler('y', np.degrees(azimuth + dAz), degrees=True)
            R1 = r1.as_matrix()
            R2 = r2.as_matrix()
            nodesOrientationMatrix[i] = np.dot(R1, R2)

            r = R.from_matrix(nodesOrientationMatrix[i])
            nodesVelocities[i, :] = r.apply(elementVelocity, inverse=False)
        Blades[ib].nodesOrientationMatrix = nodesOrientationMatrix
        Blades[ib].nodesTranslationVelocities = nodesVelocities

    return

inputs = [
 #{'nBlades':1, 'TSR':2.5},
 {'nBlades':1, 'TSR':5.0},
 #{'nBlades':1, 'TSR':7.5}
 #{'nBlades': 3, 'TSR': 5.0},
 #{'nBlades':2, 'TSR':2.5},
 #{'nBlades':2, 'TSR':5.0},
 #{'nBlades':2, 'TSR':7.5},
 #{'nBlades':3, 'TSR':5.0},
 #{'nBlades':1, 'TSR':5.0},
 #{'nBlades':2, 'TSR':2.5},
 #{'nBlades':2, 'TSR':5.0},
 #{'nBlades':2, 'TSR':7.5}
]

# Blade nodes distribution type
distribution = "cosine"
#distribution = "linear"

for input in inputs:
    # User inputs
    nBlades = input['nBlades']  # input, integer
    nRotations = 4.             # input
    rotorRadius = 0.61          # input
    bladeLength = 0.914
    bladePitch = 0.
    TSR = input['TSR']

    if (TSR == 2.5):
        windVelocity = 18.3e-2
        # Post-processing directory
        # Create target directory & all intermediate directories if don't exists
        outDir = 'outputs/outputs_TSR2.5'
    elif (TSR == 5.0):
        windVelocity = 9.1 * 1e-2
        outDir = 'outputs/outputs_TSR5.0'        
    elif(TSR == 7.5):
        windVelocity = 6.1 * 1e-2
        outDir = 'outputs/outputs_TSR7.5'        
    else:
        print('TSR not recognized!')
        exit(1)

    if not os.path.exists(outDir):
        os.makedirs(outDir)
    else:
        print("Directory ", outDir, " already exists")
    
    # Pre-processing
    rotationRPM = TSR * windVelocity / rotorRadius * 30. / np.pi
    #itersPerTour = 144.
    azimuthStep = 5.0
    itersPerTour = 360./azimuthStep
    rotationSpeed = rotationRPM * np.pi / 30.
    timeOneRotation = 60. / rotationRPM
    timeStep = timeOneRotation / itersPerTour
    timeEnd = timeOneRotation * nRotations
    innerIter = 10
    timeSteps = np.arange(0., timeEnd, timeStep)

    Blades, deltaFlts = VAWT_rotor(bladePitch, distribution, bladeLength, rotorRadius, rotationSpeed, nBlades, windVelocity, len(timeSteps))

    timeSimulation = 0.
    azimuth = 0.

    azims = []
    midBladeCn = []
    iterationVect = []
    startTime = time.time()
    for (it, t) in enumerate(timeSteps):

        updateWing(Blades, distribution, it, time, np.radians(azimuth), rotationSpeed)

        timeSimulation += timeStep

        update(Blades, windVelocity, timeStep, timeSimulation, innerIter, deltaFlts, startTime, iterationVect)
                
        postProcess = True
        if (postProcess):
            if (it == 0):
                writeHubAndTower()

            write_blade_tp(Blades, outDir, it)
            write_filaments_tp(Blades, outDir, it)

            if(it >= itersPerTour*(nRotations-1)):
                midSpanLift = Blades[0].lift[25]
                midSpanDrag = Blades[0].drag[25]
                midSpanAOA = Blades[0].attackAngle[25]
                midSpanUEff = Blades[0].effectiveVelocity[25]
                cn = midSpanLift * np.cos(midSpanAOA) + midSpanDrag*np.sin(midSpanAOA)
                ct = midSpanLift * np.sin(midSpanAOA) - midSpanDrag*np.cos(midSpanAOA)
                midBladeCn.append(cn * (midSpanUEff / windVelocity)**2.)
                azims.append((it-itersPerTour*(nRotations-1)-1) * rotationSpeed * timeStep)

        azimuth += azimuthStep
    print('Total simulation time: ', time.time() - startTime)

    plt.cla()
    plt.close()
    plt.plot(np.degrees(azims), midBladeCn, 'o-', label='SVEN')
    ref = np.genfromtxt('Measurements/Fn_Nb'+str(input['nBlades'])+'_TSR'+str(input['TSR'])+'.dat')
    ref[:,0] = ref[:,0] % 360.
    plt.plot(ref[:,0], -ref[:,1], 'o', label='Expe')
    plt.legend()
    plt.xlabel('Azimuth angle (o)')
    plt.ylabel(r'$Cn^*$ force coefficient (-)')
    plt.tight_layout()
    plt.grid()
    plt.savefig('cnEvolution_Nb'+str(input['nBlades'])+'_TSR'+str(input['TSR'])+'.png', format='png', dpi=200)
