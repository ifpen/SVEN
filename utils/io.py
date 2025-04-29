import os
import sys
import numpy as np 
# Get the directory containing the examples folder
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
parent_of_project_dir = os.path.dirname(project_dir)

# Add the project directory to the system path
sys.path.append(parent_of_project_dir)



def write_filaments_tp(blades, outDir, it):

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

def write_blade_tp(blades, outDir, it):

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
