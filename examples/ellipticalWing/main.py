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

def Wing(bladePitch, wingType):    
    
    if(wingType != "Elliptical" and wingType != "Rectangular"):
        print("Non-existing wing type: ", wingType, " please use \"Elliptical\" or \"Rectangular\"")
    
    # Blade discretisation
    nBladeCenters = 50
    nearWakeLength = 1000

    AR = 8.
    bladeLength = 2.
    if(wingType == "Elliptical"):
        cRoot = 4 * bladeLength / (AR * np.pi)    
    elif(wingType == "Rectangular"):
        cRoot = bladeLength / AR 
        
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

Blades, uInfty, deltaFlts = Wing(5., "Rectangular")

nodes = Blades[0].bladeNodes
nNodes = len(nodes)
distances = np.zeros(len(nodes)-1)
for i in range(len(nodes)-1):
    distances[i] = np.linalg.norm( nodes[i+1,:] - nodes[i,:] )
meanDistance = np.mean(distances)

relax = 0.5
timeStep = meanDistance / uInfty * relax #0.025
timeEnd  = 10.
innerIter = 10
timeSteps = np.arange(0., timeEnd, timeStep)

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

        output = open('outputs/liftDistribution_rectangular.dat', 'w')
        centers = Blades[0].centers
        liftDistribution = Blades[0].lift
        for i in range(len(centers)):
            output.write(str(centers[i][1]) + ' ' + str(liftDistribution[i])  + '\n')
        output.close()
print('Total simulation time: ', time.time() - startTime)


