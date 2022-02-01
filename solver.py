import time

from kernels import *


def update(blades, wake, uInfty, timeStep, innerIter, deltaFlts, eps_conv, particlesPerFil):
    iterationTime = time.time()



    # Attachment point of the very first filaments row
    for blade in blades:
        blade.updateFirstWakeRow()

    # Generate new particles #######################################################
    t0 = time.time()
    leftNodes = np.zeros((0, 3))
    rightNodes = np.zeros((0, 3))
    circulations = np.zeros(0)
    for blade in blades:
        # Emission after first row, uses (uInfty+induction)*dt to create right Nodes, left Nodes are taken as trailing edge nodes
        bladeLeftNodes, bladeRightNodes, bladeCirculations = blade.getFilamentsInfo(uInfty, timeStep)

        leftNodes    = np.concatenate((leftNodes, bladeLeftNodes), axis=0)
        rightNodes   = np.concatenate((rightNodes, bladeRightNodes), axis=0)
        circulations = np.concatenate((circulations, bladeCirculations), axis=0)

    #wake.addParticlesFromFilaments(leftNodes, rightNodes, circulations)

    wake.addParticlesFromFilaments_50(leftNodes, rightNodes, circulations, particlesPerFil)

   #for numPart in range(particlesPerFil):
   #    numPart = numPart + 1
   #    denom = particlesPerFil + 1
   #    rightCoef = numPart
   #    leftCoef = denom - numPart
   #    wake.addParticlesFromFilaments_2(leftNodes, rightNodes, circulations, denom, leftCoef, rightCoef, particlesPerFil)


    #wake.addParticlesFromFilaments_2(leftNodes, rightNodes, circulations, 4, 3, 1,3)
    #wake.addParticlesFromFilaments_2(leftNodes, rightNodes, circulations, 4, 2, 2,3)
    #wake.addParticlesFromFilaments_2(leftNodes, rightNodes, circulations, 4, 1, 3,3)
    t1 = time.time()
    print('addParticles: ', t1 - t0)


    t0 = time.time()
    # Includes induction at blade nodes. TODO: clarify this
    if (len(wake.particlesRadius) > 0):
        for blade in blades:
            wakeInductionsOnBlade(blade, wake)
    t1 = time.time()
    print('wakeInductionsOnBlade: ', t1 - t0)

    # Necessary? TODO: check necessity
    for i in range(len(blade.gammaShed)):
        blade.gammaShed[i] = 0.
    for i in range(len(blade.gammaTrail)):
        blade.gammaTrail[i] = 0.

    t0 = time.time()
    biotTime = 0
    # Convergence loop over gammaBound #############################################
    # Let's start with no convergence first
    # Induction for the first filament over the blade
    bladesGammaBounds = []
    for i in range(len(blades)):
        bladesGammaBounds.append(0.)

    for i in range(innerIter):
        tb0 = time.time()
        nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)
        tb1 = time.time()
        biotTime += tb1 - tb0

        iBlade = 0
        for (blade, bladeInducedVelocities) in zip(blades, nearWakeInducedVelocities):
            bladesGammaBounds[iBlade] = blade.estimateGammaBound(uInfty, bladeInducedVelocities)
            blade.updateSheds(bladesGammaBounds[iBlade])
            blade.updateTrails(bladesGammaBounds[iBlade])

            blade.gammaBound = bladesGammaBounds[iBlade]
            iBlade += 1

    #####################################################################################
    ################Beginning of second convergence technique##########################################
    #####################################################################################

    # conv_crit = 1
    # Iter = 1

    # while((Iter < innerIter) or (conv_crit>eps_conv)):
    #         print('INNER ITERATIOOOOOON :', Iter)
    #         nearWakeInducedVelocities = nearWakeInduction(blades, deltaFlts)

    #         iBlade = 0
    #         for (blade, bladeInducedVelocities) in zip(blades, nearWakeInducedVelocities):
    #             bladesGammaBounds[iBlade] = blade.estimateGammaBound(uInfty, bladeInducedVelocities)
    #             blade.updateSheds(bladesGammaBounds[iBlade])
    #             blade.updateTrails(bladesGammaBounds[iBlade])

    #             deltaGammaBound    = np.abs(bladesGammaBounds[iBlade]-blade.gammaBound)
    #             print('deltaGammaBound : ', deltaGammaBound)
    #             maxGammaBound = deltaGammaBound / np.abs(blade.gammaBound)
    #             conv_crit = np.amax(maxGammaBound)

    #             for (iBlade, blade) in enumerate(blades) :
    #                 blade.gammaBound = bladesGammaBounds[iBlade]
    #             iBlade += 1

    #         #print('Trail Filaments: ', blade.gammaTrail)
    #         #print('Shed Filaments: ', blade.gammaShed)
    #         #print('oldGammaBound:',  )
    #         print('Bounds:', bladesGammaBounds[0])
    #         print('CONVERGEEEEEENCE CRITERIOOOOON :', conv_crit)
    #         Iter+=1
    #         # print('New: ')
    #         # print(bladesGammaBounds[0])

    #####################################################################################
    ################End of second convergence technique##########################################
    #####################################################################################

    ################################################################################

    # Now that we are converged, set the new values
    for (iBlade, blade) in enumerate(blades):
        blade.storeOldGammaBound(bladesGammaBounds[iBlade])

    t1 = time.time()
    print('gammaBoundUpdate: ', t1 - t0, biotTime)

    # Advect the wake ##############################################################
    t0 = time.time()
    wakeInductionsOnWake(wake)
    t1 = time.time()
    print('wakeOnWake: ', t1 - t0)

    t0 = time.time()
    bladeInductionsOnWake(blades, wake, deltaFlts)
    t1 = time.time()
    print('bladeOnWake: ', t1 - t0)

    t0 = time.time()
    wake.advectParticles(uInfty, timeStep)
    t1 = time.time()
    print('advection: ', t1 - t0)
    ################################################################################


    ################################################################################

    print('Full iteration time: ', time.time() - iterationTime)

    return


def wakeInductionsOnBlade(blade, wake):
    ptclesPosX = wake.particlesPositionX.astype(np.float32)
    ptclesPosY = wake.particlesPositionY.astype(np.float32)
    ptclesPosZ = wake.particlesPositionZ.astype(np.float32)
    ptclesVorX = wake.particlesVorticityX.astype(np.float32)
    ptclesVorY = wake.particlesVorticityY.astype(np.float32)
    ptclesVorZ = wake.particlesVorticityZ.astype(np.float32)
    ptclesRad = wake.particlesRadius.astype(np.float32)

    particlesOnBladesKernel = modPtcles.get_function("particlesOnBladesKernel")

    fullLength = int(len(blade.centers) + len(blade.bladeNodes))
    destUx = np.zeros(fullLength).astype(np.float32)
    destUy = np.zeros(fullLength).astype(np.float32)
    destUz = np.zeros(fullLength).astype(np.float32)

    bladeNodesX = np.zeros(fullLength)
    bladeNodesY = np.zeros(fullLength)
    bladeNodesZ = np.zeros(fullLength)
    for i in range(len(blade.centers)):
        bladeNodesX[i] = blade.centers[i, 0]
        bladeNodesY[i] = blade.centers[i, 1]
        bladeNodesZ[i] = blade.centers[i, 2]

    for i in range(len(blade.bladeNodes)):
        bladeNodesX[i + len(blade.centers) ] = blade.trailingEdgeNode[i, 0]
        bladeNodesY[i + len(blade.centers) ] = blade.trailingEdgeNode[i, 1]
        bladeNodesZ[i + len(blade.centers) ] = blade.trailingEdgeNode[i, 2]
    bladeNodesX = bladeNodesX.astype(np.float32)
    bladeNodesY = bladeNodesY.astype(np.float32)
    bladeNodesZ = bladeNodesZ.astype(np.float32)

    numParticles = np.int32(len(ptclesPosX))
    numBladePoint = np.int32(fullLength)
    threadsPerBlock = 256
    blocksPerGrid = int((len(ptclesPosX) + threadsPerBlock - 1) / threadsPerBlock)

    particlesOnBladesKernel(
        drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), drv.In(bladeNodesX), drv.In(bladeNodesY),
        drv.In(bladeNodesZ), drv.In(ptclesPosX), drv.In(ptclesPosY),
        drv.In(ptclesPosZ), drv.In(ptclesVorX), drv.In(ptclesVorY),
        drv.In(ptclesVorZ), drv.In(ptclesRad), numBladePoint, numParticles,
        block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1))

    for i in range(len(blade.centers)):
        blade.inductionsFromWake[i, 0] = destUx[i] / (4.*np.pi)
        blade.inductionsFromWake[i, 1] = destUy[i] / (4.*np.pi)
        blade.inductionsFromWake[i, 2] = destUz[i] / (4.*np.pi)

    for i in range(len(blade.bladeNodes)):
        blade.inductionsAtNodes[i, 0] = destUx[i + len(blade.centers)] / (4.*np.pi)
        blade.inductionsAtNodes[i, 1] = destUy[i + len(blade.centers)] / (4.*np.pi)
        blade.inductionsAtNodes[i, 2] = destUz[i + len(blade.centers)] / (4.*np.pi)

    return

# def wakeInductionsOnBlade(blade, wake):
#     # First reshape wake induced velocities to new particle number
#     inducedVelocities = np.zeros([len(wake.particlesRadius), 3])
#     ptclesPosX = wake.particlesPositionX
#     ptclesPosY = wake.particlesPositionY
#     ptclesPosZ = wake.particlesPositionZ
#     ptclesVorX = wake.particlesVorticityX
#     ptclesVorY = wake.particlesVorticityY
#     ptclesVorZ = wake.particlesVorticityZ
#     ptclesRad = wake.particlesRadius
#     for i in range(len(blade.centers)):
#         inducedVelocity = np.sum(biotSavartParticles(blade.centers[i], ptclesPosX, ptclesPosY, ptclesPosZ, ptclesVorX, ptclesVorY, ptclesVorZ, ptclesRad, inducedVelocities), axis=0)
#         blade.inductionsFromWake[i,:] = inducedVelocity
#         blade.inductionsAtNodes[i,:] = np.zeros(3)
#
#
#     return inducedVelocity




# @jit(nopython=True)
# def biotSavartParticles(evaluationPoint, ptclesPosX, ptclesPosY, ptclesPosZ, ptclesVorX, ptclesVorY, ptclesVorZ, ptclesRad, inducedVelocities):
#     #inducedVelocities = np.zeros([len(wake.particlesRadius),3])
#     for i in range(len(ptclesPosX)):
#         #ptclePosition = np.asarray([wake.particlesPositionX[i], wake.particlesPositionY[i], wake.particlesPositionZ[i]])
#         #ptcleVorticiy = np.asarray([wake.particlesVorticityX[i], wake.particlesVorticityY[i], wake.particlesVorticityZ[i]])
#         #ptcleRadius = wake.particlesRadius[i]
#         #coreSize = wake.particlesCoreSize[i]
#         ptclePosition = np.asarray([ptclesPosX[i], ptclesPosY[i], ptclesPosZ[i]])
#         ptcleVorticiy = np.asarray([ptclesVorX[i], ptclesVorY[i], ptclesVorZ[i]])
#         ptcleRadius = ptclesRad[i]
#         #coreSize = wake.particlesCoreSize[i]
#         evalPoint_Minus_Particles = evaluationPoint - ptclePosition
#         norm = np.linalg.norm(evalPoint_Minus_Particles)
#         if(np.abs(norm) > 1e-6):
#             # Epsilon for particles regularization
#             delta = 0.15
#             epsilon = ptcleRadius * delta;
#             # Rosenhead regularisation - G. Pinon thesis
#             d = norm*norm + epsilon*epsilon;
#             cst = 1. / (d*np.sqrt(d));
#             numer = cst * evalPoint_Minus_Particles;
#             crossProduct = np.cross(numer, ptcleVorticiy)
#             inducedVelocities[i,:] = - crossProduct / (4.*np.pi)
#         else:
#             inducedVelocities[i,:] = [0., 0., 0.]
#     return inducedVelocities

def wakeInductionsOnWake(wake):
    if (len(wake.particlesRadius) > 0):
        # First reshape wake induced velocities to new particle number
        wake.inducedVelocities = np.zeros([len(wake.particlesRadius), 3])

        ptclesPosX = wake.particlesPositionX.astype(np.float32)
        ptclesPosY = wake.particlesPositionY.astype(np.float32)
        ptclesPosZ = wake.particlesPositionZ.astype(np.float32)
        ptclesVorX = wake.particlesVorticityX.astype(np.float32)
        ptclesVorY = wake.particlesVorticityY.astype(np.float32)
        ptclesVorZ = wake.particlesVorticityZ.astype(np.float32)
        ptclesRad = wake.particlesRadius.astype(np.float32)

        particlesOnParticlesKernel = mod.get_function("particlesOnParticlesKernel")

        destUx = np.zeros_like(ptclesPosX).astype(np.float32)
        destUy = np.zeros_like(ptclesPosY).astype(np.float32)
        destUz = np.zeros_like(ptclesPosZ).astype(np.float32)
        numParticles = np.int32(len(ptclesPosX))
        threadsPerBlock = 256
        blocksPerGrid = int((len(ptclesPosX) + threadsPerBlock - 1) / threadsPerBlock)

        particlesOnParticlesKernel(
            drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), drv.In(ptclesPosX), drv.In(ptclesPosY),
            drv.In(ptclesPosZ), drv.In(ptclesVorX), drv.In(ptclesVorY),
            drv.In(ptclesVorZ), drv.In(ptclesRad), numParticles,
            block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1))

        wake.inducedVelocities[:, 0] = destUx[:] / (4.*np.pi)
        wake.inducedVelocities[:, 1] = destUy[:] / (4.*np.pi)
        wake.inducedVelocities[:, 2] = destUz[:] / (4.*np.pi)

    return


def nearWakeInduction(blades, deltaFlts):
    useBoundFilaments = True

    leftNodes = np.zeros((0, 3))
    rightNodes = np.zeros((0, 3))
    circulations = np.zeros(0)
    for blade in blades:
        bladeLeftNodes, bladeRightNodes, bladeCirculations = blade.getNodesAndCirculations(useBoundFilaments)
        leftNodes = np.concatenate((leftNodes, bladeLeftNodes))
        rightNodes = np.concatenate((rightNodes, bladeRightNodes))
        circulations = np.concatenate((circulations, bladeCirculations))

    allBladeInductions = []
    for blade in blades:
        inducedVelocity = np.zeros([len(blade.centers), 3])
        for i in range(len(blade.centers)):
            inducedVelocity[i, :] = biotSavartFilaments(blade.centers[i], leftNodes, rightNodes, circulations,
                                                        deltaFlts)
        allBladeInductions.append(inducedVelocity)

    return allBladeInductions


def bladeInductionsOnWake(blades, wake, deltaFlts):
    if (len(wake.particlesRadius) > 0):
        # useBoundFilaments = False
        useBoundFilaments = True

        bladeOnParticlesKernel = modFlts.get_function("bladeOnParticlesKernel")

        # Destination positions
        ptclesPosX = wake.particlesPositionX.astype(np.float32)
        ptclesPosY = wake.particlesPositionY.astype(np.float32)
        ptclesPosZ = wake.particlesPositionZ.astype(np.float32)

        # Destination velocities
        destUx = np.zeros_like(ptclesPosX).astype(np.float32)
        destUy = np.zeros_like(ptclesPosY).astype(np.float32)
        destUz = np.zeros_like(ptclesPosZ).astype(np.float32)

        # Input filaments positions and circulations
        leftNodes = np.zeros((0, 3))
        rightNodes = np.zeros((0, 3))
        circulations = np.zeros(0)
        for blade in blades:
            bladeLeftNodes, bladeRightNodes, bladeCirculations = blade.getNodesAndCirculations(useBoundFilaments)
            leftNodes = np.concatenate((leftNodes, bladeLeftNodes), axis=0)
            rightNodes = np.concatenate((rightNodes, bladeRightNodes))
            circulations = np.concatenate((circulations, bladeCirculations))

        fltsLeftNodesX = leftNodes[:, 0].astype(np.float32)
        fltsLeftNodesY = leftNodes[:, 1].astype(np.float32)
        fltsLeftNodesZ = leftNodes[:, 2].astype(np.float32)
        fltsRightNodesX = rightNodes[:, 0].astype(np.float32)
        fltsRightNodesY = rightNodes[:, 1].astype(np.float32)
        fltsRightNodesZ = rightNodes[:, 2].astype(np.float32)
        fltsCirculations = circulations.astype(np.float32)

        numParticles = np.int32(len(ptclesPosX))
        numFilaments = np.int32(len(fltsCirculations))
        deltaFlts = np.float32(deltaFlts)

        threadsPerBlock = 256
        blocksPerGrid = int((len(ptclesPosX) + threadsPerBlock - 1) / threadsPerBlock)

        bladeOnParticlesKernel(
            drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), drv.In(ptclesPosX), drv.In(ptclesPosY),
            drv.In(ptclesPosZ), drv.In(fltsLeftNodesX), drv.In(fltsLeftNodesY),
            drv.In(fltsLeftNodesZ), drv.In(fltsRightNodesX), drv.In(fltsRightNodesY),
            drv.In(fltsRightNodesZ), drv.In(fltsCirculations), numParticles, numFilaments, deltaFlts,
            block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1))

        wake.inducedVelocities[:, 0] += destUx[:]
        wake.inducedVelocities[:, 1] += destUy[:]
        wake.inducedVelocities[:, 2] += destUz[:]

    return
