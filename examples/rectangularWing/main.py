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

def Wing(bladePitch, wingType, aspectRatio, nIterations):    
    
    if(wingType != "Elliptical" and wingType != "Rectangular"):
        print("Non-existing wing type: ", wingType, " please use \"Elliptical\" or \"Rectangular\"")
    
    # Blade discretisation
    nBladeCenters = 50
    nearWakeLength = nIterations

    bladeLength = 2.
    if(wingType == "Elliptical"):
        AR = 8.
        cRoot = 4 * bladeLength / (AR * np.pi)    
    elif(wingType == "Rectangular"):
        #P = 6.
        # P = 2. / np.pi * bladeLength / cRoot
        #cRoot = 2. / np.pi * bladeLength / P
        # cRoot = bladeLength / AR
        
        AR = aspectRatio
        cRoot = bladeLength / AR
        
    print(2./np.pi * bladeLength / cRoot)
    # input()

    uInfty = 1.

    # Multiple straight wings
    nodes = np.zeros([nBladeCenters + 1, 3])
    nodes[:, 1] = (np.linspace(0., 1., nBladeCenters + 1) - 0.5) * bladeLength
    
    if(wingType == "Rectangular"):
        nodeChords = np.ones_like(nodes[:,0]) * cRoot
    elif(wingType == "Elliptical"):        
        nodeChords = np.zeros(len(nodes))
        for i in range(len(nodes)):
            nodeChords[i] = np.sqrt(np.abs(cRoot ** 2. * (1. - 4. * (nodes[i, 1] / bladeLength) ** 2.)))

    airfoils = []
    for i in range(len(nodes) - 1):
        airfoils.append(Airfoil('../../data/flatPlate.foil'))

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

    liftingLine1.updateFirstWakeRow()
    liftingLine1.initializeWake()

    Blades = []
    Blades.append(liftingLine1)

    deltaFlts = np.sqrt(1e-6)

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
                output.write(str(blade.bladeNodes[i,0]-1./4.*blade.nodeChords[i]) + " " + str(blade.bladeNodes[i,1]) + " " + str(blade.bladeNodes[i,2]) + "\n")
        for i in range(shape):
                output.write(str(blade.trailingEdgeNode[i,0]) + " " + str(blade.trailingEdgeNode[i,1]) + " " + str(blade.trailingEdgeNode[i,2]) + "\n")
        output.close()
    return

wingType = "Rectangular"
bladePitch = 5.

aspectRatios = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
for aspectRatio in aspectRatios:
    #aspectRatio = 20

    BladesTmp, uInfty, deltaFlts = Wing(bladePitch, wingType, aspectRatio, 1) # Initialze the blade without time step knowledge

    nodes = BladesTmp[0].bladeNodes
    nNodes = len(nodes)
    distances = np.zeros(len(nodes)-1)
    for i in range(len(nodes)-1):
        distances[i] = np.linalg.norm( nodes[i+1,:] - nodes[i,:] )
    meanDistance = np.mean(distances)

    relax = 0.5
    timeStep = meanDistance / uInfty * relax #0.025
    timeEnd  = 15.
    innerIter = 10
    timeSteps = np.arange(0., timeEnd, timeStep)

    Blades, uInfty, deltaFlts = Wing(bladePitch, wingType, aspectRatio, len(timeSteps))

    deltaPtcles = 1e-4
    partsPerFil = 1

    wake = Wake()

    eps_conv = 1e-4

    timeSimulation = 0.

    startTime = time.time()
    for (it, t) in enumerate(timeSteps):

        print('iteration, time, finaltime: ', it, t, timeSteps[-1])
        timeSimulation += timeStep
        update(Blades, wake, uInfty, timeStep, timeSimulation, innerIter, deltaFlts, deltaPtcles, eps_conv, partsPerFil)

        postProcess = True
        if(postProcess):
            write_particles(outDir)
            write_blade_tp(Blades, outDir)
            write_filaments_tp(Blades, outDir)

            if(wingType == "Elliptical"):
                output = open('liftDistribution_elliptical.dat', 'w')
            else:
                output = open('liftDistribution_rectangular_AR'+str(aspectRatio)+'.dat', 'w')
            centers = Blades[0].centers
            liftDistribution = Blades[0].lift / (2.*np.pi*np.radians(bladePitch))
            for i in range(len(centers)):
                output.write(str(centers[i][1]/2.+0.5) + ' ' + str(liftDistribution[i]) + '\n')
            output.close()
            
                #G0 = 0.07 # 0.4659
                #bladeLength = 2.
                #if(wingType == 'Elliptical'):
                    #theory = G0 * np.sqrt(1.-(2.*np.abs(centers[i][1])/bladeLength)**2.) # .5*(cl0+cl0 * np.sqrt(1.-(2.*np.abs(centers[i][1])/bladeLength)**2.))
                #else:
                    #theory = .5 * (G0 + G0 * np.sqrt(1.-(2.*np.abs(centers[i][1])/bladeLength)**2.) )        
    print('Total simulation time: ', time.time() - startTime)


