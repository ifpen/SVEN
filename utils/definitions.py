import os
import sys
import numpy as np 
# Get the directory containing the examples folder
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)

# Add the project directory to the system path
sys.path.append(parent_of_project_dir)

from sven.airfoil import *
from sven.blade import *


def Wing(bladePitch, nBladeCenters, AR, bladeLength, nearWakeLength ):    
         
    cRoot = 4 * bladeLength / (AR * np.pi)    
    nodes = np.zeros([nBladeCenters + 1, 3])
    nodes[:, 1] = (np.linspace(0., 1., nBladeCenters + 1) - 0.5) * bladeLength
    
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

    return Blades


