import time

from kernels import *

def update(blades, wake, uInfty, timeStep, timeSimulation, innerIter, deltaFlts, deltaPtcles, eps_conv, particlesPerFil):

    iteration = timeSimulation / timeStep

    iterationTime = time.time()

    # Initialize stuff
    for (iBlade, blade) in enumerate(blades):
        blade.inductionsFromWake[:,:] = 0.
        blade.inductionsAtNodes[:,:] = 0.
        blade.wakeNodesInductions[:, :, :] = 0.

    # Attachment point of the very first filaments row
    for blade in blades:
        blade.updateFirstWakeRow()

    # Generate new particles #######################################################
    t0 = time.time()
    leftNodes = np.zeros((0, 3))
    rightNodes = np.zeros((0, 3))
    circulations = np.zeros(0)
    nearWakeLength = 0
    for blade in blades:
        # Emission after first row, uses (uInfty+induction)*dt to create right Nodes, left Nodes are taken as trailing edge nodes
        if(blade.nearWakeLength == 2):
            bladeLeftNodes, bladeRightNodes, bladeCirculations = blade.getFilamentsInfo(uInfty, timeStep)
        else:
            bladeLeftNodes, bladeRightNodes, bladeCirculations = blade.getLastFilamentsInfo(uInfty, timeStep)
        nearWakeLength = blade.nearWakeLength

        leftNodes = np.concatenate((leftNodes, bladeLeftNodes), axis=0)
        rightNodes = np.concatenate((rightNodes, bladeRightNodes), axis=0)
        circulations = np.concatenate((circulations, bladeCirculations), axis=0)

    # if(iteration > nearWakeLength):
    wake.addParticlesFromFilaments_50(leftNodes, rightNodes, circulations, particlesPerFil)

    t1 = time.time()
    print('addParticles: ', t1 - t0)

    t0 = time.time()
    if(nearWakeLength > 2):
        bladeOrWake = "blade"
        nearWakeInductionsOnBladeOrWake(blades, wake, deltaFlts, bladeOrWake)

    # Includes induction at blade nodes. TODO: clarify this
    if (len(wake.particlesRadius) > 0):
        for blade in blades:
            wakeInductionsOnBlade(blade, wake, deltaPtcles)

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
    wakeInductionsOnWake(wake, deltaPtcles)
    t1 = time.time()
    print('wakeOnWake: ', t1 - t0)

    # if(timeSimulation > 30.*timeStep):
    if(nearWakeLength > 2):
        nearWakeInductionsOnBladeOrWake(blades, wake, deltaFlts, "wake")

    t0 = time.time()
    bladeInductionsOnWake(blades, wake, deltaFlts)
    t1 = time.time()
    print('bladeOnWake: ', t1 - t0)

    t0 = time.time()
    wake.advectParticles(uInfty, timeStep)

    if(nearWakeLength > 2):
        for blade in blades:
            blade.advectFilaments(uInfty, timeStep)
    t1 = time.time()
    print('advection: ', t1 - t0)
    ################################################################################

    if(nearWakeLength > 2):
        for blade in blades:
            blade.spliceNearWake()
            blade.updateFilamentCirulations()

    ################################################################################
    print('Full iteration time: ', time.time() - iterationTime)

    return


def wakeInductionsOnBlade(blade, wake, deltaPtcles):

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
        bladeNodesX[i + len(blade.centers)] = blade.trailingEdgeNode[i, 0]
        bladeNodesY[i + len(blade.centers)] = blade.trailingEdgeNode[i, 1]
        bladeNodesZ[i + len(blade.centers)] = blade.trailingEdgeNode[i, 2]
    bladeNodesX = bladeNodesX.astype(np.float32)
    bladeNodesY = bladeNodesY.astype(np.float32)
    bladeNodesZ = bladeNodesZ.astype(np.float32)

    numParticles = np.int32(len(ptclesPosX))
    numBladePoint = np.int32(fullLength)
    threadsPerBlock = 256
    blocksPerGrid = int((len(ptclesPosX) + threadsPerBlock - 1) / threadsPerBlock)

    deltaParticles = np.float32(deltaPtcles)

    particlesOnBladesKernel(
        drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), drv.In(bladeNodesX), drv.In(bladeNodesY),
        drv.In(bladeNodesZ), drv.In(ptclesPosX), drv.In(ptclesPosY),
        drv.In(ptclesPosZ), drv.In(ptclesVorX), drv.In(ptclesVorY),
        drv.In(ptclesVorZ), drv.In(ptclesRad), deltaParticles, numBladePoint, numParticles,
        block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1))

    for i in range(len(blade.centers)):
        blade.inductionsFromWake[i, 0] += destUx[i] / (4. * np.pi)
        blade.inductionsFromWake[i, 1] += destUy[i] / (4. * np.pi)
        blade.inductionsFromWake[i, 2] += destUz[i] / (4. * np.pi)

    for i in range(len(blade.bladeNodes)):
        blade.inductionsAtNodes[i, 0] += destUx[i + len(blade.centers)] / (4. * np.pi)
        blade.inductionsAtNodes[i, 1] += destUy[i + len(blade.centers)] / (4. * np.pi)
        blade.inductionsAtNodes[i, 2] += destUz[i + len(blade.centers)] / (4. * np.pi)

    return

def wakeInductionsOnWake(wake, deltaPtcles):
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

        deltaParticles = np.float32(deltaPtcles)
        particlesOnParticlesKernel(
            drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), drv.In(ptclesPosX), drv.In(ptclesPosY),
            drv.In(ptclesPosZ), drv.In(ptclesVorX), drv.In(ptclesVorY),
            drv.In(ptclesVorZ), drv.In(ptclesRad), deltaParticles, numParticles,
            block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1))

        wake.inducedVelocities[:, 0] += destUx[:] / (4. * np.pi)
        wake.inducedVelocities[:, 1] += destUy[:] / (4. * np.pi)
        wake.inducedVelocities[:, 2] += destUz[:] / (4. * np.pi)

    return


def nearWakeInduction(blades, deltaFlts):
    useBoundFilaments = True

    leftNodes = np.zeros((0, 3))
    rightNodes = np.zeros((0, 3))
    circulations = np.zeros(0)

    allBladeInductions = np.zeros((len(blades), len(blades[0].centers), 3))

    for blade in blades:
        bladeLeftNodes, bladeRightNodes, bladeCirculations = blade.getNodesAndCirculations(useBoundFilaments)
        leftNodes = np.concatenate((leftNodes, bladeLeftNodes))
        rightNodes = np.concatenate((rightNodes, bladeRightNodes))
        circulations = np.concatenate((circulations, bladeCirculations))

    for (iblade,blade) in enumerate(blades):
        allBladeInductions[iblade, :, :] = biotSavartFilaments_v4(blade.centers, leftNodes, rightNodes, circulations, deltaFlts)

    return allBladeInductions


def bladeInductionsOnWake(blades, wake, deltaFlts):

    # useBoundFilaments = False
    useBoundFilaments = True

    bladeOnParticlesKernel = modFlts.get_function("bladeOnParticlesKernel")

    # Destination positions
    nodesPosX = np.zeros(0)
    nodesPosY = np.zeros(0)
    nodesPosZ = np.zeros(0)

    for blade in blades:
        nodesX = blade.wakeNodes[:, :, 0].flatten()
        nodesPosX = np.concatenate((nodesPosX, nodesX))
        nodesY = blade.wakeNodes[:, :, 1].flatten()
        nodesPosY = np.concatenate((nodesPosY, nodesY))
        nodesZ = blade.wakeNodes[:, :, 2].flatten()
        nodesPosZ = np.concatenate((nodesPosZ, nodesZ))

    fltsNodesSize = len(nodesPosX)

    nodesPosX = np.concatenate((nodesPosX, wake.particlesPositionX))
    nodesPosY = np.concatenate((nodesPosY, wake.particlesPositionY))
    nodesPosZ = np.concatenate((nodesPosZ, wake.particlesPositionZ))
    wakePtclesSize = len(wake.particlesPositionX)

    nodesPosX = nodesPosX.astype(np.float32)
    nodesPosY = nodesPosY.astype(np.float32)
    nodesPosZ = nodesPosZ.astype(np.float32)

    # Destination velocities
    destUx = np.zeros_like(nodesPosX).astype(np.float32)
    destUy = np.zeros_like(nodesPosY).astype(np.float32)
    destUz = np.zeros_like(nodesPosZ).astype(np.float32)

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

    numParticles = np.int32(len(nodesPosX))
    numFilaments = np.int32(len(fltsCirculations))
    deltaFlts = np.float32(deltaFlts)

    threadsPerBlock = 256
    blocksPerGrid = int((len(nodesPosX) + threadsPerBlock - 1) / threadsPerBlock)

    bladeOnParticlesKernel(
        drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), drv.In(nodesPosX), drv.In(nodesPosY),
        drv.In(nodesPosZ), drv.In(fltsLeftNodesX), drv.In(fltsLeftNodesY),
        drv.In(fltsLeftNodesZ), drv.In(fltsRightNodesX), drv.In(fltsRightNodesY),
        drv.In(fltsRightNodesZ), drv.In(fltsCirculations), numParticles, numFilaments, deltaFlts,
        block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1))

    # WARNING: works only for 1 blade!
    filamentsInductionsUx = destUx[:fltsNodesSize]
    filamentsInductionsUy = destUy[:fltsNodesSize]
    filamentsInductionsUz = destUz[:fltsNodesSize]

    shape = (len(blades),len(blade.trailFilamentsCirculation), blade.nearWakeLength)
    filamentsInductionsUx = np.reshape(filamentsInductionsUx, shape)
    filamentsInductionsUy = np.reshape(filamentsInductionsUy, shape)
    filamentsInductionsUz = np.reshape(filamentsInductionsUz, shape)
    for (iBlade,blade) in enumerate(blades):
        blade.wakeNodesInductions[:,:,0] += filamentsInductionsUx[iBlade,:,:]
        blade.wakeNodesInductions[:,:,1] += filamentsInductionsUy[iBlade,:,:]
        blade.wakeNodesInductions[:,:,2] += filamentsInductionsUz[iBlade,:,:]

    particlesInductionsUx = destUx[fltsNodesSize:]
    particlesInductionsUy = destUy[fltsNodesSize:]
    particlesInductionsUz = destUz[fltsNodesSize:]

    wake.inducedVelocities[:, 0] += particlesInductionsUx[:]
    wake.inducedVelocities[:, 1] += particlesInductionsUy[:]
    wake.inducedVelocities[:, 2] += particlesInductionsUz[:]

    return


def nearWakeInductionsOnBladeOrWake(blades, wake, deltaFlts, bladeOrWake):

    # GPU version
    bladeOnParticlesKernel_v2 = modFlts.get_function("bladeOnParticlesKernel")

    # Input nodes over which inductions are computed
    nodesPosX = np.zeros(0)
    nodesPosY = np.zeros(0)
    nodesPosZ = np.zeros(0)

    # Input filaments positions and circulations
    leftNodesX = np.zeros(0)
    leftNodesY = np.zeros(0)
    leftNodesZ = np.zeros(0)
    rightNodesX = np.zeros(0)
    rightNodesY = np.zeros(0)
    rightNodesZ = np.zeros(0)
    circulations = np.zeros(0)

    for blade in blades:

        # Destination (nodes)
        if(bladeOrWake == "wake"):
            nodesX = blade.wakeNodes[:,:,0].flatten()
            nodesPosX = np.concatenate((nodesPosX, nodesX))
            nodesY = blade.wakeNodes[:,:,1].flatten()
            nodesPosY = np.concatenate((nodesPosY, nodesY))
            nodesZ = blade.wakeNodes[:,:,2].flatten()
            nodesPosZ = np.concatenate((nodesPosZ, nodesZ))
        elif(bladeOrWake == "blade"):
            nodesPosX = np.concatenate((nodesPosX, blade.centers[:,0]))
            nodesPosY = np.concatenate((nodesPosY, blade.centers[:,1]))
            nodesPosZ = np.concatenate((nodesPosZ, blade.centers[:,2]))
            nodesPosX = np.concatenate((nodesPosX, blade.trailingEdgeNode[:,0]))
            nodesPosY = np.concatenate((nodesPosY, blade.trailingEdgeNode[:,1]))
            nodesPosZ = np.concatenate((nodesPosZ, blade.trailingEdgeNode[:,2]))

        else:
            print('bladeOrWake= ', bladeOrWake, ' is invalid.')
            exit(0)


        # Filament (wake) - Trail
        leftNodesX = np.concatenate((blade.wakeNodes[:,:-1,0].flatten(), leftNodesX))
        rightNodesX = np.concatenate((blade.wakeNodes[:,1:,0].flatten(), rightNodesX))
        leftNodesY = np.concatenate((blade.wakeNodes[:,:-1,1].flatten(), leftNodesY))
        rightNodesY = np.concatenate((blade.wakeNodes[:,1:,1].flatten(), rightNodesY))
        leftNodesZ = np.concatenate((blade.wakeNodes[:,:-1,2].flatten(), leftNodesZ))
        rightNodesZ = np.concatenate((blade.wakeNodes[:,1:,2].flatten(), rightNodesZ))
        circulations = np.concatenate((blade.trailFilamentsCirculation.flatten(), circulations))

        # Filament (wake) - Shed
        leftNodesX = np.concatenate((blade.wakeNodes[:-1,:,0].flatten(), leftNodesX))
        rightNodesX = np.concatenate((blade.wakeNodes[1:,:,0].flatten(), rightNodesX))
        leftNodesY = np.concatenate((blade.wakeNodes[:-1,:,1].flatten(), leftNodesY))
        rightNodesY = np.concatenate((blade.wakeNodes[1:,:,1].flatten(), rightNodesY))
        leftNodesZ = np.concatenate((blade.wakeNodes[:-1,:,2].flatten(), leftNodesZ))
        rightNodesZ = np.concatenate((blade.wakeNodes[1:,:,2].flatten(), rightNodesZ))
        circulations = np.concatenate((blade.shedFilamentsCirculation.flatten(), circulations))

    # Disregard filaments with zero circulation
    idZeroCirc = np.where(circulations == 0.)
    leftNodesX = np.delete(leftNodesX,idZeroCirc)
    rightNodesX = np.delete(rightNodesX,idZeroCirc)
    leftNodesY = np.delete(leftNodesY,idZeroCirc)
    rightNodesY = np.delete(rightNodesY,idZeroCirc)
    leftNodesZ = np.delete(leftNodesZ,idZeroCirc)
    rightNodesZ = np.delete(rightNodesZ,idZeroCirc)
    circulations = np.delete(circulations,idZeroCirc)

    if(len(circulations) > 0):

        nodesPosX = nodesPosX.astype(np.float32)
        nodesPosY = nodesPosY.astype(np.float32)
        nodesPosZ = nodesPosZ.astype(np.float32)

        # Destination velocities
        destUx = np.zeros_like(nodesPosX).astype(np.float32)
        destUy = np.zeros_like(nodesPosY).astype(np.float32)
        destUz = np.zeros_like(nodesPosZ).astype(np.float32)

        fltsLeftNodesX = leftNodesX[:].astype(np.float32)
        fltsLeftNodesY = leftNodesY[:].astype(np.float32)
        fltsLeftNodesZ = leftNodesZ[:].astype(np.float32)
        fltsRightNodesX = rightNodesX[:].astype(np.float32)
        fltsRightNodesY = rightNodesY[:].astype(np.float32)
        fltsRightNodesZ = rightNodesZ[:].astype(np.float32)
        fltsCirculations = circulations.astype(np.float32)

        numParticles = np.int32(len(nodesPosX))
        numFilaments = np.int32(len(fltsCirculations))
        deltaFlts = np.float32(deltaFlts)

        threadsPerBlock = 256
        blocksPerGrid = int((len(nodesPosX) + threadsPerBlock - 1) / threadsPerBlock)

        bladeOnParticlesKernel_v2(
            drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), drv.In(nodesPosX), drv.In(nodesPosY),
            drv.In(nodesPosZ), drv.In(fltsLeftNodesX), drv.In(fltsLeftNodesY),
            drv.In(fltsLeftNodesZ), drv.In(fltsRightNodesX), drv.In(fltsRightNodesY),
            drv.In(fltsRightNodesZ), drv.In(fltsCirculations), numParticles, numFilaments, deltaFlts,
            block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1))


        if(bladeOrWake == "blade"):
            for blade in blades:
                for i in range(len(blade.centers)):
                    blade.inductionsFromWake[i, 0] += destUx[i] #/ (4. * np.pi)
                    blade.inductionsFromWake[i, 1] += destUy[i] #/ (4. * np.pi)
                    blade.inductionsFromWake[i, 2] += destUz[i] #/ (4. * np.pi)

                for i in range(len(blade.bladeNodes)):
                    blade.inductionsAtNodes[i, 0] += destUx[i + len(blade.centers)] #/ (4. * np.pi)
                    blade.inductionsAtNodes[i, 1] += destUy[i + len(blade.centers)] #/ (4. * np.pi)
                    blade.inductionsAtNodes[i, 2] += destUz[i + len(blade.centers)] #/ (4. * np.pi)

        else:
            shape = (len(blades), len(blade.trailFilamentsCirculation), blade.nearWakeLength)  # np.shape(blade.wakeNodesInductions[:,:])
            inducedVelX = np.reshape(destUx, shape)
            inducedVelY = np.reshape(destUy, shape)
            inducedVelZ = np.reshape(destUz, shape)
            for (iBlade, myBlade) in enumerate(blades):
                myBlade.wakeNodesInductions[:, :, 0] += inducedVelX[iBlade, :, :]
                myBlade.wakeNodesInductions[:, :, 1] += inducedVelY[iBlade, :, :]
                myBlade.wakeNodesInductions[:, :, 2] += inducedVelZ[iBlade, :, :]

    return