import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)
sys.path.append(parent_of_project_dir)

from sven import Airfoil
from sven import Blade
from sven.solver import update 
import time 

outDir = 'outputs'
os.makedirs(outDir, exist_ok=True)
if os.path.exists(outDir):
    print(f"The '{outDir}' directory already exists.")

# -----------------------------------------------------------------------------
# Filament writing functions 
# -----------------------------------------------------------------------------

def write_filaments_tp(blades, outDir, it):
    for iBlade, blade in enumerate(blades):
        shape = np.shape(blade.wakeNodes)

        file_path = os.path.join(
            outDir, 
            f'Filaments_Nodes_Blade_{iBlade}_tStep_{it}.tp')
        with open(file_path, 'w') as output:
            output.write('TITLE="Near-wake nodes"\n')
            output.write('VARIABLES="X" "Y" "Z" "Circulation"\n')
            output.write(
                f'ZONE T="Near-wake" I={shape[0]} J={shape[1]-1}, K=1, '
                'DT=(SINGLE SINGLE SINGLE SINGLE)\n'
            )
            for j in range(shape[1] - 1):
                for i in range(shape[0]):
                    output.write(
                        f"{blade.wakeNodes[i, j, 0]} {blade.wakeNodes[i, j, 1]} "
                        f"{blade.wakeNodes[i, j, 2]} {blade.trailFilamentsCirculation[i, j]}\n"
                    )

def write_blade_tp(blades, outDir, it):
    for iBlade, blade in enumerate(blades):
        shape = len(blade.bladeNodes)

        file_path = os.path.join(outDir, f'Blade_{iBlade}_Nodes_tStep_{it}.tp')
        with open(file_path, 'w') as output:
            output.write('TITLE="Near-wake nodes"\n')
            output.write('VARIABLES="X" "Y" "Z"\n')
            output.write(
                f'ZONE T="Near-wake" I={shape} J=2, K=1, '
                'DT=(SINGLE SINGLE SINGLE)\n'
            )
            for i in range(shape):
                output.write(
                    f"{blade.bladeNodes[i, 0] - 1. / 4. * blade.nodeChords[i]} "
                    f"{blade.bladeNodes[i, 1]} {blade.bladeNodes[i, 2]}\n"
                )
            for i in range(shape):
                output.write(
                    f"{blade.trailingEdgeNode[i, 0]} {blade.trailingEdgeNode[i, 1]} "
                    f"{blade.trailingEdgeNode[i, 2]}\n"
                )

# -----------------------------------------------------------------------------
# Wing definition 
# -----------------------------------------------------------------------------

# wing and simulation parameters

bladePitch = 5.0
nBladeCenters = 40
AR = 6.0
bladeLength = 10.0
nearWakeLength = 100

uInfty = 1.0
deltaFlts = np.sqrt(1e-3)


def Wing(bladePitch, nBladeCenters, AR, bladeLength, nearWakeLength):
    cRoot = 4 * bladeLength / (AR * np.pi)
    nodes = np.zeros([nBladeCenters + 1, 3])
    nodes[:, 1] = (np.linspace(0., 1., nBladeCenters + 1) - 0.5) * bladeLength

    nodeChords = np.sqrt(
        np.abs(cRoot ** 2. * (1. - 4. * (nodes[:, 1] / bladeLength) ** 2.))
    )

    airfoils = [Airfoil('./geometry/flatPlate.foil') for _ in range(len(nodes) - 1)]

    centersOrientationMatrix = np.array([
        R.from_euler('y', bladePitch, degrees=True).as_matrix()
        for _ in range(len(nodes) - 1)
    ])

    nodesOrientationMatrix = np.array([
        R.from_euler('y', bladePitch, degrees=True).as_matrix()
        for _ in range(len(nodes))
    ])

    liftingLine1 = Blade(
        nodes, nodeChords, nearWakeLength, airfoils,
        centersOrientationMatrix, nodesOrientationMatrix,
        np.zeros([len(nodes) - 1, 3]), np.zeros([len(nodes), 3])
    )

    liftingLine1.updateFirstWakeRow()
    liftingLine1.initializeWake()

    return [liftingLine1]

Blades = Wing(bladePitch, nBladeCenters, AR, bladeLength, nearWakeLength)


# -----------------------------------------------------------------------------
# Time loop 
# -----------------------------------------------------------------------------

# Time loop parameters
timeStep = 0.1
timeEnd = nearWakeLength * timeStep
innerIter = 10
timeSteps = np.arange(0., timeEnd, timeStep)
timeSimulation = 0.
iterationVect = []

startTime = time.time()
for it, t in enumerate(timeSteps):
    print(f"Iteration: {it}, Temps: {t:.2f}, Temps final: {timeSteps[-1]:.2f}")
    timeSimulation += timeStep
    update(
        Blades, uInfty, timeStep, timeSimulation,
        innerIter, deltaFlts, startTime, iterationVect
    )

    # Write post processing files (deactivated by default)
    postProcess = True
    if postProcess:
        write_blade_tp(Blades, outDir, it)
        write_filaments_tp(Blades, outDir, it)

# Saving data for lift distribution 

output_file = os.path.join(outDir, 'liftDistribution_elliptical.dat')
with open(output_file, 'w') as output:
    centers = Blades[0].centers
    liftDistribution = Blades[0].lift
    for center, lift in zip(centers, liftDistribution):
        output.write(f"{center[1]} {lift}\n")
