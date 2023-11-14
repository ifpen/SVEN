import numpy as np

data = np.genfromtxt('liftDistribution_rectangular_AR7.dat')

nFiles = 35-4

outData = np.zeros( [ nFiles+1, np.shape(data)[0] ] )
#outData[0,:] = data[:,0]

lifts = np.zeros([nFiles,len(data[:,1])+2])
    
for i in range(nFiles):
    fileIndex = i + 5
    data = np.genfromtxt('liftDistribution_rectangular_AR'+str(fileIndex)+'.dat')
    outData[i,:] = data[:,1]
    lifts[i,1:-1] = data[:,1]
np.savetxt('tabulatedLiftDistributions.dat', np.transpose(lifts), delimiter=' ')

innerPoints = np.arange(0.01, 1., 0.02)
spanLocations = np.zeros(len(innerPoints)+2)
spanLocations[1:-1] = innerPoints
spanLocations[-1] = 1.

aspectRatios = np.arange(5., 36., 1.);

output = open("tipLossInputs.h", "w")

output.write('#include <vector>\n')

output.write('std::vector<double> m_SpanwiseLocations={')
comma = ','
for (i,sl) in enumerate(spanLocations):
    if(i == len(spanLocations)-1):
        comma = ''
    output.write(str(sl)+comma)
output.write('};\n')

output.write('std::vector<double> m_AspectRatios={')
comma = ','
for (i,sl) in enumerate(aspectRatios):
    if(i == len(aspectRatios)-1):
        comma = ''
    output.write(str(sl)+comma)
output.write('};\n')

output.write('std::vector< std::vector<double> > m_liftDistribution={')
shapeLift = np.shape(lifts)
lastIndex = shapeLift[0]*shapeLift[1]
comma = ','

iter = 0
for i in range(shapeLift[0]):
    
    comma2 = ','
    output.write('{')
    
    for j in range(shapeLift[1]):
        if(j == shapeLift[1]-1):
            comma = ''
        else:
            comma = ','
        output.write(str(lifts[i][j])+comma)
        
    if(i == shapeLift[0]-1):
        comma2 = ''
    else:
        comma2 = ','
    output.write('}'+comma2)
        
output.write('};\n')
output.close()
