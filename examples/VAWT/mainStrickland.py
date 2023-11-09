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


def VAWT_rotor(bladePitch, bladeLength, rotorRadius, rotationSpeed, nBlades, windVelocity):
    # if (wingType != "Elliptical" and wingType != "Rectangular"):
    #     print("Non-existing wing type: ", wingType, " please use \"Elliptical\" or \"Rectangular\"")

    # Blade discretisation
    nBladeCenters = 50
    nearWakeLength = 1000

    # if (wingType == "Elliptical"):
    #     AR = 8.
    #     cRoot = 4 * bladeLength / (AR * np.pi)
    # elif (wingType == "Rectangular"):
    #     P = 6.
    #     # P = 2. / np.pi * bladeLength / cRoot
    #     cRoot = 2. / np.pi * bladeLength / P
    #     # cRoot = bladeLength / AR
    # cRoot = 4. / nBlades
    # uInfty = 6.

    cRoot = 0.0914

    nodeChords = np.ones(nBladeCenters + 1) * cRoot
    airfoils = []
    for i in range(nBladeCenters):
        airfoils.append(Airfoil('../../turbineModels/VAWT/reference_files/airfoils/stricklandAirfoil.foil'))

    # Multiple straight wings
    initialNodes = np.zeros([nBladeCenters + 1, 3])
    initialNodes[:, 2] = (np.linspace(0., 1., nBladeCenters + 1) - 0.5) * bladeLength
    #print('v0: ', initialNodes[:, 2])
    initialNodes[:, 2] = np.asarray((CosineNodesDistribution(nBladeCenters) - 0.5) * bladeLength)
    #print('v1: ', initialNodes[:, 2])
    #input()

    initialNodes[:, 1] = rotorRadius

    Blades = []
    dAz = 2. * np.pi / nBlades
    for ib in range(nBlades):

        azimuth = ib * dAz

        print('INITIAL BLADE NODES: ', initialNodes)

        nodes = np.copy(initialNodes)
        # Rotate them with the azimuth
        for i in range(len(initialNodes[:, 0])):
            x = initialNodes[i, 0]
            y = initialNodes[i, 1]

            nodes[i, 0] = x * np.cos(azimuth) - y * np.sin(azimuth)
            nodes[i, 1] = x * np.sin(azimuth) + y * np.cos(azimuth)
            print(i, x, y, nodes[i, 0], nodes[i, 1])

        # print('nodes: ', nodes, ' azimuth= ', np.degrees(azimuth))
        # print()

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
        print('FINAL BLADE NODES: ', liftingLine.bladeNodes)
        liftingLine.updateFirstWakeRow()
        liftingLine.initializeWake()

        Blades.append(liftingLine)

    updateWing(Blades, 0, 0, 0., rotationSpeed)

    deltaFlts = 5e-2  # np.sqrt(1e-2)

    print('Solidity: ', nBlades * nodeChords[0] / 2. / rotorRadius)
    print('TSR     : ', rotationSpeed * rotorRadius / windVelocity)
    print(rotationSpeed)

    return Blades, deltaFlts


def updateWing(Blades, iteration, time, azimuth, rotationSpeed):
    #
    nBlades = len(Blades)
    print('NBLADES', nBlades)
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

    for (ib, blade) in enumerate(Blades):

        dAz = ib * 2. * np.pi / nBlades

        # Multiple straight wings
        nCenters = len(blade.centers)
        nodes = np.zeros([nCenters + 1, 3])

        # z positions do not change
        #nodes[:, 2] = (np.linspace(0., 1., nCenters + 1) - 0.5) * bladeLength
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


def write_particles(outDir):
    output = open(outDir + '/New_Particles_tStep_' + str(it) + '.particles', 'w')
    for i in range(len(wake.particlesPositionX)):
        # if (wake.particlesVorticityX[i] +wake.particlesVorticityY[i] +wake.particlesVorticityZ[i] > 1e-6):
        output.write(str(wake.particlesPositionX[i]) + " " + str(wake.particlesPositionY[i]) + " " + str(
            wake.particlesPositionZ[i]) + " " + "\n")
    output.close()

    return


def write_filaments_tp(blades, outDir):
    for (iBlade, blade) in enumerate(blades):
        shape = np.shape(blade.wakeNodes)

        output = open(outDir + '/Filaments_Nodes_' + '_Blade_' + str(iBlade) + '_tStep_' + str(it) + '.tp', 'w')
        output.write('TITLE="Near-wake nodes"\n')
        output.write('VARIABLES="X" "Y" "Z" "Circulation"\n')
        output.write('ZONE T="Near-wake" I=' + str(shape[0]) + ' J=' + str(
            shape[1] - 1) + ', K=1, DT=(SINGLE SINGLE SINGLE SINGLE)\n')
        for j in range(np.shape(blade.wakeNodes)[1] - 1):
            for i in range(np.shape(blade.wakeNodes)[0]):
                output.write(str(blade.wakeNodes[i, j, 0]) + " " + str(blade.wakeNodes[i, j, 1]) + " " + str(
                    blade.wakeNodes[i, j, 2]) + " " + str(blade.trailFilamentsCirculation[i, j]) + "\n")
        output.close()

    return


def write_blade_tp(blades, outDir):
    for (iBlade, blade) in enumerate(blades):
        shape = len(blade.bladeNodes)

        output = open(outDir + '/Blade_' + str(iBlade) + '_Nodes_tStep_' + str(it) + '.tp', 'w')
        output.write('TITLE="Near-wake nodes"\n')
        output.write('VARIABLES="X" "Y" "Z"\n')
        output.write('ZONE T="Near-wake" I=' + str(shape) + ' J=' + str(2) + ', K=1, DT=(SINGLE SINGLE SINGLE)\n')
        for i in range(shape):
            output.write(str(blade.bladeNodes[i, 0]) + " " + str(blade.bladeNodes[i, 1]) + " " + str(
                blade.bladeNodes[i, 2]) + "\n")
        for i in range(shape):
            output.write(str(blade.trailingEdgeNode[i, 0]) + " " + str(blade.trailingEdgeNode[i, 1]) + " " + str(
                blade.trailingEdgeNode[i, 2]) + "\n")
        output.close()
    return

inputs = [
 {'nBlades': 3, 'TSR': 5.0},
 {'nBlades':1, 'TSR':5.0},
 {'nBlades':2, 'TSR':2.5},
 {'nBlades':2, 'TSR':5.0},
 {'nBlades':2, 'TSR':7.5},
 {'nBlades':3, 'TSR':5.0},
 {'nBlades':1, 'TSR':5.0},
 {'nBlades':2, 'TSR':2.5},
 {'nBlades':2, 'TSR':5.0},
 {'nBlades':2, 'TSR':7.5}
]

for input in inputs:
    # User inputs
    nBlades = input['nBlades']  # input, integer
    nRotations = 4.  # input
    rotorRadius = 0.61  # input
    bladeLength = 0.914
    bladePitch = 0.
    TSR = input['TSR']
    #windVelocity = input['uInfs'] #18.3 * 1e-2
    if (TSR == 2.5):
        windVelocity = 18.3e-2
    elif (TSR == 5.0):
        windVelocity = 9.1 * 1e-2
    elif(TSR == 7.5):
        windVelocity = 6.1 * 1e-2
    else:
        print('TSR not recognized!')
        exit(1)

    # Pre-processing
    rotationRPM = TSR * windVelocity / rotorRadius * 30. / np.pi
    itersPerTour = 36.
    rotationSpeed = rotationRPM * np.pi / 30.
    timeOneRotation = 60. / rotationRPM
    timeStep = timeOneRotation / itersPerTour
    timeEnd = timeOneRotation * nRotations
    innerIter = 10
    timeSteps = np.arange(0., timeEnd, timeStep)

    Blades, deltaFlts = VAWT_rotor(bladePitch, bladeLength, rotorRadius, rotationSpeed, nBlades, windVelocity)

    deltaPtcles = 1e-4

    wake = Wake()

    eps_conv = 1e-4

    timeSimulation = 0.
    azimuth = 0.

    azims = []
    midBladeCn = []
    startTime = time.time()
    for (it, t) in enumerate(timeSteps):

        azimuth = it * rotationSpeed * timeStep

        myAzimuth = np.degrees(azimuth) % 360.
        print("###############################", myAzimuth)
        azims.append(myAzimuth)

        print('azimuth: ', np.degrees(azimuth))
        updateWing(Blades, it, time, azimuth, rotationSpeed)

        print('iteration, time, finaltime: ', it, t, timeSteps[-1])
        partsPerFil = 1
        timeSimulation += timeStep
        update(Blades, wake, windVelocity, timeStep, timeSimulation, innerIter, deltaFlts, deltaPtcles, eps_conv, partsPerFil)

        postProcess = True
        if (postProcess):
            if (it == 0):
                writeHubAndTower()

            write_particles(outDir)
            write_blade_tp(Blades, outDir)
            write_filaments_tp(Blades, outDir)

            if(it >= itersPerTour*(nRotations-1)):
                midSpanLift = Blades[0].lift[25]
                midSpanDrag = Blades[0].drag[25]
                midSpanAOA = Blades[0].attackAngle[25]
                midSpanUEff = Blades[0].effectiveVelocity[25]
                cn = midSpanLift * np.cos(midSpanAOA) + midSpanDrag*np.sin(midSpanAOA)
                ct = midSpanLift * np.sin(midSpanAOA) - midSpanDrag*np.cos(midSpanAOA)
                midBladeCn.append(cn * (midSpanUEff / windVelocity)**2.)

    print('Total simulation time: ', time.time() - startTime)

    import matplotlib.pyplot as plt

    plt.cla()
    plt.close()
    plt.plot(np.linspace(0., 360., int(itersPerTour))-10., midBladeCn, 'o-', label='PITCHOU-10deg')
    plt.plot(np.linspace(0., 360., int(itersPerTour)), midBladeCn, 'o-', label='PITCHOU')
    #ref = np.genfromtxt('V2D_Results.dat')
    #plt.plot(ref[:,0], ref[:,1], label='VertiGO V2D')
    #ref = np.genfromtxt('AC_Results.dat')
    #plt.plot(ref[:,0], ref[:,1], label='VertiGO AC')
    ref = np.genfromtxt('Measurements/Fn_Nb'+str(input['nBlades'])+'_TSR'+str(input['TSR'])+'.dat')
    ref[:,0] = ref[:,0] % 360.
    plt.plot(ref[:,0], -ref[:,1], 'o', label='Expe')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig('cnEvolution_Nb'+str(input['nBlades'])+'_TSR'+str(input['TSR'])+'.png', format='png', dpi=200)