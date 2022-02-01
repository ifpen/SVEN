import os
import numpy as np

from windTurbine import *
from airfoil import *
from blade import *
from solver import *
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

    dataAirfoils = np.genfromtxt('turbineModels/NewMexico/reference_files/mexico.blade', skip_header=1, usecols=(7),
                                 dtype='U')
    intAirfoils = np.arange(0, len(dataAirfoils))

    data = np.genfromtxt('turbineModels/NewMexico/reference_files/mexico.blade', skip_header=1)

    refRadius = data[:,2]#np.linspace(0., 2.04, 50) # data[:,2] #

    # Radius start at blade root, not hub center!
    inputNodesTwistAngles = -sign * np.radians(data[:, 5])
    inputNodesChord = data[:, 6]

    f = interpolate.interp1d(data[:, 2], intAirfoils, kind='nearest')
    centersAirfoils = []
    for i in range(len(refRadius)):
        foilName = str(dataAirfoils[int(f(refRadius[i]))])
        centersAirfoils.append(Airfoil('turbineModels/NewMexico/reference_files/' + foilName, headerLength=1))

    nodesRadius = hubRadius + refRadius
    nodesTwistAngles = np.interp(refRadius, data[:, 2], inputNodesTwistAngles)
    nodesChord = np.interp(refRadius, data[:, 2], inputNodesChord)

    # Build wind turbine
    hubCenter = [0., 0., 0.]
    myWT = windTurbine(nBlades, hubCenter, hubRadius, rotationalVelocity, windVelocity, bladePitch)
    blades = myWT.initializeTurbine(nodesRadius, nodesChord, centersAirfoils, nodesTwistAngles, myWT.nBlades)

    return blades, myWT, windVelocity, 1e-5


def StraightBlade():
    # Blade discretisation
    nBladeCenters = 30
    bladeLength = 18.
    bladePitch = 5.
    uInfty = 8.
    deltaFlts = 0.5

    # Multiple straight wings
    nodes = np.zeros([nBladeCenters + 1, 3])
    nodes[:, 1] = (np.linspace(0., 1., nBladeCenters + 1) + 0.75) * bladeLength

    nodeChords = np.ones(len(nodes)) * 2.
    airfoils = []
    for i in range(len(nodes) - 1):
        airfoils.append(Airfoil('naca64418.foil'))

    centersOrientationMatrix = np.zeros([len(nodes) - 1, 3, 3])
    for i in range(len(nodes) - 1):
        r = R.from_euler('y', bladePitch, degrees=True)
        centersOrientationMatrix[i] = r.as_matrix()

    nodesOrientationMatrix = np.zeros([len(nodes), 3, 3])
    for i in range(len(nodes)):
        r = R.from_euler('y', bladePitch, degrees=True)
        nodesOrientationMatrix[i] = r.as_matrix()

    liftingLine1 = Blade(nodes, nodeChords, airfoils, centersOrientationMatrix, nodesOrientationMatrix,
                         np.zeros([len(nodes) - 1, 3]))

    nodes = np.zeros([nBladeCenters + 1, 3])
    nodes[:, 0] = 4. * 2.
    nodes[:, 1] = (np.linspace(0., 1., nBladeCenters + 1) - 0.0) * bladeLength
    liftingLine2 = Blade(nodes, nodeChords, airfoils, centersOrientationMatrix, nodesOrientationMatrix,
                         np.zeros([len(nodes) - 1, 3]))

    nodes = np.zeros([nBladeCenters + 1, 3])
    nodes[:, 0] = 8. * 2.
    nodes[:, 1] = (np.linspace(0., 1., nBladeCenters + 1) + 1.5) * bladeLength
    liftingLine3 = Blade(nodes, nodeChords, airfoils, centersOrientationMatrix, nodesOrientationMatrix,
                         np.zeros([len(nodes) - 1, 3]))

    Blades = []
    Blades.append(liftingLine1)
    Blades.append(liftingLine2)
    Blades.append(liftingLine3)

    return Blades, uInfty, deltaFlts


def StraightWingCastor():
    # Blade discretisation
    nBladeCenters = 10
    bladeLength = 18.
    bladePitch = +8.
    uInfty = 15.
    deltaFlts = 0.5
    chord = 1.8

    # Multiple straight wings
    nodes = np.zeros([nBladeCenters + 1, 3])
    nodes[:, 1] = (np.linspace(0., 1., nBladeCenters + 1)) * bladeLength
    nodeChords = np.ones(len(nodes)) * chord

    airfoils = []
    for i in range(len(nodes) - 1):
        airfoils.append(Airfoil('naca64418.foil'))

    centersOrientationMatrix = np.zeros([len(nodes) - 1, 3, 3])
    for i in range(len(nodes) - 1):
        r = R.from_euler('y', bladePitch, degrees=True)
        centersOrientationMatrix[i] = r.as_matrix()

    nodesOrientationMatrix = np.zeros([len(nodes), 3, 3])
    for i in range(len(nodes)):
        r = R.from_euler('y', bladePitch, degrees=True)
        nodesOrientationMatrix[i] = r.as_matrix()
    liftingLine1 = Blade(nodes, nodeChords, airfoils, centersOrientationMatrix, nodesOrientationMatrix,
                         np.zeros([len(nodes) - 1, 3]))

    Blades = []
    Blades.append(liftingLine1)

    return Blades, uInfty, deltaFlts


def EllipticalWing(bladePitch):
    # Blade discretisation
    nBladeCenters = 40

    AR = 18.
    bladeLength = 10.
    cRoot = 4 * bladeLength / (AR * np.pi)
    bladePitch = bladePitch
    uInfty = 1.

    # Multiple straight wings
    nodes = np.zeros([nBladeCenters + 1, 3])
    nodes[:, 1] = (np.linspace(0., 1., nBladeCenters + 1) - 0.5) * bladeLength
    centers = .5 * (nodes[1:] + nodes[:-1])

    nodeChords = np.zeros(len(nodes))
    for i in range(len(nodes)):
        nodeChords[i] = np.sqrt(np.abs(cRoot ** 2. * (1. - 4. * (nodes[i, 1] / bladeLength) ** 2.)))

    airfoils = []
    for i in range(len(nodes) - 1):
        airfoils.append(Airfoil('flatPlate.foil'))

    centersOrientationMatrix = np.zeros([len(nodes) - 1, 3, 3])
    for i in range(len(nodes) - 1):
        r = R.from_euler('y', bladePitch, degrees=True)
        centersOrientationMatrix[i] = r.as_matrix()

    nodesOrientationMatrix = np.zeros([len(nodes), 3, 3])
    for i in range(len(nodes)):
        r = R.from_euler('y', bladePitch, degrees=True)
        nodesOrientationMatrix[i] = r.as_matrix()

    liftingLine1 = Blade(nodes, nodeChords, airfoils, centersOrientationMatrix, nodesOrientationMatrix,
                         np.zeros([len(nodes) - 1, 3]), np.zeros([len(nodes), 3]))

    Blades = []
    Blades.append(liftingLine1)

    deltaFlts = np.sqrt(0.01)

    return Blades, uInfty, deltaFlts







#def write_particles(outDir):
#    output = open(outDir + '/New_Particles_tStep_' + str(it) + '.particles', 'w')
#    for i in range(len(wake.particlesPositionX)):
#        output.write(str(wake.particlesPositionX[i]) + " " + str(wake.particlesPositionY[i]) + " " + str(
#            wake.particlesPositionZ[i]) + " " + str(np.sqrt(
#            wake.particlesVorticityX[i] ** 2. + wake.particlesVorticityY[i] ** 2. + wake.particlesVorticityZ[
#                i] ** 2.)) + "\n")
#    output.close()

#    return

def write_particles(outDir):
    output = open(outDir + '/New_Particles_tStep_' + str(it) + '.particles', 'w')
    for i in range(len(wake.particlesPositionX)):
        output.write(str(wake.particlesPositionX[i]) + " " + str(wake.particlesPositionY[i]) + " " + str(
            wake.particlesPositionZ[i]) + " " + "\n")
    output.close()

    return


def write_blade(blades, outDir):
    for (i, blade) in enumerate(blades):

        output = open(outDir + '/NearWake_Blade_' + str(i) + '_tStep_' + str(it) + '.particles', 'w')
        for i in range(len(blade.bladeNodes)):
            output.write(str(blade.bladeNodes[i][0]) + " " + str(blade.bladeNodes[i][1]) + " " + str(
                blade.bladeNodes[i][2]) + " 0.0\n")
        for i in range(len(blade.trailingEdgeNode)):
            output.write(str(blade.trailingEdgeNode[i][0]) + " " + str(blade.trailingEdgeNode[i][1]) + " " + str(
                blade.trailingEdgeNode[i][2]) + " 0.0\n")
        output.close()
    return

############ Elliptical change of pitch case #############################
#
#timeStep = 0.1
#timeEnd = 20.
## timeEnd = 0.2
#innerIter = 4
#eps_conv = 1e-4
#timeSteps = np.arange(0., timeEnd, timeStep)
#bladePitch = 2.
#bladePitch1 = 8.
#Blades, uInfty, deltaFlts = EllipticalWing(bladePitch)
#centers = Blades[0].centers
#liftDistribution = Blades[0].lift
#
#wake = Wake()
#output = open('liftDistribution_elliptical_2.dat', 'w')
#Blades1, uInfty1, deltaFlts1 = EllipticalWing(bladePitch1)
#liftDistribution1 = Blades1[0].lift
#liftDistributions = 0.
#liftDistributions1 = 0.
#
#for (it, t) in enumerate(timeSteps):
#    if t<10 :
#        print('iteration, time, finaltime: ', it, t, timeSteps[-1])
#        update(Blades, wake, uInfty, timeStep, innerIter, deltaFlts, eps_conv, 2)
#        write_particles(outDir)
#        write_blade(Blades, outDir)
#        for i in range(len(centers)):
#            liftDistributions += liftDistribution[i]
#        liftDistributions = liftDistributions/(len(centers)+1)
#        print('#########LIFTdIST :', liftDistributions)
#        print('########Comparison :', liftDistribution[20])
#        output.write(str(centers[20][1]) + ' ' + str(liftDistributions) + ' ' + str(t) + ' ' + str(
#            np.degrees(Blades[0].attackAngle[20])) + '\n')
#    else :
#        update(Blades1, wake, uInfty1, timeStep, innerIter, deltaFlts1, eps_conv, 2)
#        write_particles(outDir)
#        write_blade(Blades1,outDir)
#        for i in range(len(centers)):
#            liftDistributions1 += liftDistribution1[i]
#        liftDistributions1 = liftDistributions1/(len(centers)+1)
#        output.write(str(centers[20][1]) + ' ' + str(liftDistributions1) + ' ' + str(t) + ' ' + str(
#            np.degrees(Blades[0].attackAngle[20])) + '\n')
##for i in range(len(centers)):
#    #output.write(str(centers[20][1]) + ' ' + str(liftDistribution[20]) + ' ' + str(t) + ' ' + str(np.degrees(Blades[0].attackAngle[20]))+ '\n')
#




#output.close()




############ END Elliptical change of pitch case #############################







windTurbineCase = True
# Blades, WindTurbine, uInfty, deltaFlts = NewMexicoWindTurbine()  # EllipticalWing() #StraightWingCastor() #StraightBlade() #EllipticalWing()

if(windTurbineCase == True):
    Blades, WindTurbine, uInfty, deltaFlts = NewMexicoWindTurbine()
    DegreesPerTimeStep = 10.
    timeStep = np.radians(
        DegreesPerTimeStep) / WindTurbine.rotationalVelocity
    innerIter  = 6
    nRotations = 10.
    timeEnd = np.radians(nRotations * 360.) / WindTurbine.rotationalVelocity
    eps_conv = 1e-4

    refAzimuth = -WindTurbine.rotationalVelocity * timeStep

    timeSteps = np.arange(0., timeEnd, timeStep)
else:
    timeStep = 0.1
    timeEnd = 10.
    innerIter = 4
    timeSteps = np.arange(0., timeEnd, timeStep)
    Blades, uInfty, deltaFlts = EllipticalWing()

wake = Wake()

eps_conv = 1e-4

startTime = time.time()
for (it, t) in enumerate(timeSteps):

    if(windTurbineCase == True):
        refAzimuth += WindTurbine.rotationalVelocity * timeStep
        WindTurbine.updateTurbine(refAzimuth)

    print('iteration, time, finaltime: ', it, t, timeSteps[-1])
    update(Blades, wake, uInfty, timeStep, innerIter, deltaFlts, eps_conv, 1)

    write_particles(outDir)
    write_blade(Blades, outDir)

    if (windTurbineCase == True):
        centers = Blades[0].centers

        Fn, Ft = WindTurbine.evaluateForces(1.191) #(1.197)
        output = open('bladeForces.dat', 'w')
        for i in range(len(centers)):
            output.write(str(np.linalg.norm(centers[i])) + ' ' + str(Fn[i]) + ' ' + str(Ft[i]) + '\n')
        output.close()

        output = open('liftDistribution.dat', 'w')
        liftDistribution = Blades[0].lift
        for i in range(len(centers)):
            if (windTurbineCase == True):
                TwistAndPitch = WindTurbine.bladePitch + .5 * (
                        WindTurbine.nodesTwistAngles[i] + WindTurbine.nodesTwistAngles[i + 1])
                aoa_th = np.degrees(np.arctan2(uInfty, WindTurbine.rotationalVelocity * centers[i][1]) - TwistAndPitch)
            else:
                output = open('liftDistribution.dat', 'w')
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



#centers = Blades[0].centers
#liftDistribution = Blades[0].lift
#
#
#
## if(windTurbineCase == True):
##     Fn, Ft = WindTurbine.evaluateForces(1.191)
##     output = open('bladeForces.dat', 'w')
##     for i in range(len(centers)):
##         output.write(str(centers[i][1]) + ' ' + str(Fn[i]) + ' ' + str(Ft[i]) + '\n')
##     output.close()
