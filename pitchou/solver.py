import time


from pitchou.inductions import *
from pitchou.wake import *

def update(blades, wake, uInfty, timeStep, timeSimulation, innerIter, deltaFlts, deltaPtcles, eps_conv, particlesPerFil):

    iteration = timeSimulation / timeStep
    iterationTime = time.time()

    #############################################################################################
    # Initialize all inductions
    #############################################################################################
    for (iBlade, blade) in enumerate(blades):
        blade.inductionsFromWake[:,:] = 0.
        blade.inductionsAtNodes[:,:] = 0.
        blade.wakeNodesInductions[:, :, :] = 0.

    ##############################################################################################
    # Calculates the attachment point of the very first filament row (or trailing edge position)
    ##############################################################################################
    for blade in blades:
        blade.updateFirstWakeRow()

    ##############################################################################################
    # Getting info on left and right nodes from wake filaments in order to
    # generate wake particles :
    # (1) "getFilamentsInfo" : allows for emission after the first row of filaments
    #                        where (uInfty+induction)*dt is used to create right Nodes
    #                        and trailing edge nodes are used as left Nodes
    # (2) "getLastFilamentsInfo" : allows for particle emission after last filament row in the wake
    ###############################################################################################
    t0 = time.time()
    leftNodes = np.zeros((0, 3))
    rightNodes = np.zeros((0, 3))
    circulations = np.zeros(0)
    nearWakeLength = 0
    for blade in blades:
        if(blade.nearWakeLength == 2):
            bladeLeftNodes, bladeRightNodes, bladeCirculations = blade.getFilamentsInfo(uInfty, timeStep)
        else:
            bladeLeftNodes, bladeRightNodes, bladeCirculations = blade.getLastFilamentsInfo(uInfty, timeStep)
        nearWakeLength = blade.nearWakeLength

        leftNodes = np.concatenate((leftNodes, bladeLeftNodes), axis=0)
        rightNodes = np.concatenate((rightNodes, bladeRightNodes), axis=0)
        circulations = np.concatenate((circulations, bladeCirculations), axis=0)

    ##############################################################################################################
    # "addParticlesFromFilaments_50" : takes left and right nodes of previously defined filaments and adds a user
    #                                  defined "particlesPerFil" number of particles between the nodes.
    ##############################################################################################################
    if(iteration > nearWakeLength):
        wake.addParticlesFromFilaments_50(leftNodes, rightNodes, circulations, particlesPerFil)
    t1 = time.time()
    print('addParticles: ', t1 - t0)

    ############################################################################################
    # If the wake is composed of filaments : compute the filaments' induction on blade centers
    ############################################################################################
    t0 = time.time()
    if(nearWakeLength > 2):
        bladeOrWake = "blade"
        wakeFilamentsInductionsOnBladeOrWake(blades, wake, deltaFlts, bladeOrWake)

    #############################################################################################
    # If the wake is composed of particles : compute the particles' induction on blade centers
    #############################################################################################
    if (len(wake.particlesRadius) > 0):
        for blade in blades:
            wakeParticlesInductionsOnBlade(blade, wake, deltaPtcles)

    t1 = time.time()
    print('wakeInductionsOnBlade: ', t1 - t0)

    #################################################################################################
    # Not clear but important : these have to be set back to zero before gamma bound convergence loop
    #################################################################################################
    for i in range(len(blade.gammaShed)):
        blade.gammaShed[i] = 0.
    for i in range(len(blade.gammaTrail)):
        blade.gammaTrail[i] = 0.

    t0 = time.time()
    biotTime = 0


    ############################################################################################
    # Convergence loop over gammaBound
    ############################################################################################

    bladesGammaBounds = []
    for i in range(len(blades)):
        bladesGammaBounds.append(0.)
    for i in range(innerIter):
        tb0 = time.time()
        #######################################################################################################
        # (1) "nearWakeInduction" : calculates induced velocities of bound filaments from one blade to another
        # (2) "estimateGammaBound": knowing all induced velocities on the blade -> calculate the blade's
        #                           effective velocity + angle of attack + lift coefficient -> determine new
        #                           bound circulation value.
        # (3) "updateSheds/updateTrails" : knowing new bound circulation -> shed and trail circulations can be
        #                                  deduced.
        #######################################################################################################
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

    ################################################################################
    # Store bound circulation value after convergence : important for next timestep
    ################################################################################
    for (iBlade, blade) in enumerate(blades):
        blade.storeOldGammaBound(bladesGammaBounds[iBlade])

    t1 = time.time()
    print('gammaBoundUpdate: ', t1 - t0, biotTime)

    #########################################################################################################
    # Compute all inductions on wake elements : 
    # (1) "wakeFilamentsInductionsOnBladeOrWake" : inductions from wake filaments on all other wake filaments
    # (2) "particlesInductionOnFilaments"        : inductions from wake Particles on wake filaments
    # (3) "filamentsInductionOnParticles"        : inductions from wake Filaments on wake particles
    # (4) "bladeInductionsOnWake"                : inductions from blades on wake filaments and/or particles
    #########################################################################################################
    t0 = time.time()
    wakeInductionsOnWake(wake, deltaPtcles)
    t1 = time.time()
    print('wakeOnWake: ', t1 - t0)

    # if(timeSimulation > 30.*timeStep):
    if(nearWakeLength > 2):
        wakeFilamentsInductionsOnBladeOrWake(blades, wake, deltaFlts, "wake")
        particlesInductionOnFilaments(blades, wake, deltaPtcles,wake.ptclesPosX, wake.ptclesPosY, wake.ptclesPosZ, wake.ptclesVorX, wake.ptclesVorY, wake.ptclesVorZ, wake.ptclesRad)
        filamentsInductionOnParticles(blades, wake, deltaFlts)
    
    t0 = time.time()
    bladeInductionsOnWake(blades, wake, deltaFlts)
    t1 = time.time()
    print('bladeOnWake: ', t1 - t0)

    ######################################################################################
    # Once all inductions are known, the induced wake velocity is used to advect particles
    # and filaments in the wake.
    ######################################################################################
    t0 = time.time()
    wake.advectParticles(uInfty, timeStep)

    if(nearWakeLength > 2):
        for blade in blades:
            blade.advectFilaments(uInfty, timeStep)
    t1 = time.time()
    print('advection: ', t1 - t0)
    
    ###########################################################################################
    # (1) "spliceNearWake"            : trail and shed filaments from the second to last row
    #                                   take values of sheds and trails from first to second
    #                                   to last row.
    # (2) "updateFilamentCirculation" : first row of filaments take trail and shed circulations
    #                                   values computed after gammaBound convergence loop.
    ###########################################################################################

    if(nearWakeLength > 2):
        for blade in blades:
            blade.spliceNearWake()
            blade.updateFilamentCirulations()
   

    print('Full iteration time: ', time.time() - iterationTime)
    return


def wakeParticlesInductionsOnBlade(blade, wake, deltaPtcles):
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

    for (iblade, blade) in enumerate(blades):
        allBladeInductions[iblade, :, :] = biotSavartFilaments_v4(blade.centers, leftNodes, rightNodes, circulations,
                                                                  deltaFlts)

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

    shape = (len(blades), len(blade.trailFilamentsCirculation), blade.nearWakeLength)
    filamentsInductionsUx = np.reshape(filamentsInductionsUx, shape)
    filamentsInductionsUy = np.reshape(filamentsInductionsUy, shape)
    filamentsInductionsUz = np.reshape(filamentsInductionsUz, shape)
    for (iBlade, blade) in enumerate(blades):
        blade.wakeNodesInductions[:, :, 0] += filamentsInductionsUx[iBlade, :, :] / (4. * np.pi)
        blade.wakeNodesInductions[:, :, 1] += filamentsInductionsUy[iBlade, :, :] / (4. * np.pi)
        blade.wakeNodesInductions[:, :, 2] += filamentsInductionsUz[iBlade, :, :] / (4. * np.pi)

    particlesInductionsUx = destUx[fltsNodesSize:]
    particlesInductionsUy = destUy[fltsNodesSize:]
    particlesInductionsUz = destUz[fltsNodesSize:]

    wake.inducedVelocities[:, 0] += particlesInductionsUx[:] / (4. * np.pi)
    wake.inducedVelocities[:, 1] += particlesInductionsUy[:] / (4. * np.pi)
    wake.inducedVelocities[:, 2] += particlesInductionsUz[:] / (4. * np.pi)

    return




def wakeFilamentsInductionsOnBladeOrWake(blades, wake, deltaFlts, bladeOrWake):
    # GPU versiond
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
        if (bladeOrWake == "wake"):
            nodesX = blade.wakeNodes[:, :, 0].flatten()
            nodesPosX = np.concatenate((nodesPosX, nodesX))
            nodesY = blade.wakeNodes[:, :, 1].flatten()
            nodesPosY = np.concatenate((nodesPosY, nodesY))
            nodesZ = blade.wakeNodes[:, :, 2].flatten()
            nodesPosZ = np.concatenate((nodesPosZ, nodesZ))
        elif (bladeOrWake == "blade"):
            nodesPosX = np.concatenate((nodesPosX, blade.centers[:, 0]))
            nodesPosY = np.concatenate((nodesPosY, blade.centers[:, 1]))
            nodesPosZ = np.concatenate((nodesPosZ, blade.centers[:, 2]))
            nodesPosX = np.concatenate((nodesPosX, blade.trailingEdgeNode[:, 0]))
            nodesPosY = np.concatenate((nodesPosY, blade.trailingEdgeNode[:, 1]))
            nodesPosZ = np.concatenate((nodesPosZ, blade.trailingEdgeNode[:, 2]))

        else:
            print('bladeOrWake= ', bladeOrWake, ' is invalid.')
            exit(0)

        # Filament (wake) - Trail
        leftNodesX = np.concatenate((blade.wakeNodes[:, :-1, 0].flatten(), leftNodesX))
        rightNodesX = np.concatenate((blade.wakeNodes[:, 1:, 0].flatten(), rightNodesX))
        leftNodesY = np.concatenate((blade.wakeNodes[:, :-1, 1].flatten(), leftNodesY))
        rightNodesY = np.concatenate((blade.wakeNodes[:, 1:, 1].flatten(), rightNodesY))
        leftNodesZ = np.concatenate((blade.wakeNodes[:, :-1, 2].flatten(), leftNodesZ))
        rightNodesZ = np.concatenate((blade.wakeNodes[:, 1:, 2].flatten(), rightNodesZ))
        circulations = np.concatenate((blade.trailFilamentsCirculation.flatten(), circulations))

        # Filament (wake) - Shed
        leftNodesX = np.concatenate((blade.wakeNodes[:-1, :, 0].flatten(), leftNodesX))
        rightNodesX = np.concatenate((blade.wakeNodes[1:, :, 0].flatten(), rightNodesX))
        leftNodesY = np.concatenate((blade.wakeNodes[:-1, :, 1].flatten(), leftNodesY))
        rightNodesY = np.concatenate((blade.wakeNodes[1:, :, 1].flatten(), rightNodesY))
        leftNodesZ = np.concatenate((blade.wakeNodes[:-1, :, 2].flatten(), leftNodesZ))
        rightNodesZ = np.concatenate((blade.wakeNodes[1:, :, 2].flatten(), rightNodesZ))
        circulations = np.concatenate((blade.shedFilamentsCirculation.flatten(), circulations))

    # Disregard filaments with zero circulation
    idZeroCirc = np.where(circulations == 0.)
    leftNodesX = np.delete(leftNodesX, idZeroCirc)
    rightNodesX = np.delete(rightNodesX, idZeroCirc)
    leftNodesY = np.delete(leftNodesY, idZeroCirc)
    rightNodesY = np.delete(rightNodesY, idZeroCirc)
    leftNodesZ = np.delete(leftNodesZ, idZeroCirc)
    rightNodesZ = np.delete(rightNodesZ, idZeroCirc)
    circulations = np.delete(circulations, idZeroCirc)

    if (len(circulations) > 0):

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

        if (bladeOrWake == "blade"):
            for blade in blades:
                for i in range(len(blade.centers)):
                    blade.inductionsFromWake[i, 0] += destUx[i] / (4. * np.pi)
                    blade.inductionsFromWake[i, 1] += destUy[i] / (4. * np.pi)
                    blade.inductionsFromWake[i, 2] += destUz[i] / (4. * np.pi)

                for i in range(len(blade.bladeNodes)):
                    blade.inductionsAtNodes[i, 0] += destUx[i + len(blade.centers)] / (4. * np.pi)
                    blade.inductionsAtNodes[i, 1] += destUy[i + len(blade.centers)] / (4. * np.pi)
                    blade.inductionsAtNodes[i, 2] += destUz[i + len(blade.centers)] / (4. * np.pi)

        else:
            shape = (len(blades), len(blade.trailFilamentsCirculation),
                     blade.nearWakeLength)  # np.shape(blade.wakeNodesInductions[:,:])
            inducedVelX = np.reshape(destUx, shape)
            inducedVelY = np.reshape(destUy, shape)
            inducedVelZ = np.reshape(destUz, shape)
            for (iBlade, myBlade) in enumerate(blades):
                myBlade.wakeNodesInductions[:, :, 0] += inducedVelX[iBlade, :, :] / (4. * np.pi)
                myBlade.wakeNodesInductions[:, :, 1] += inducedVelY[iBlade, :, :] / (4. * np.pi)
                myBlade.wakeNodesInductions[:, :, 2] += inducedVelZ[iBlade, :, :] / (4. * np.pi)

    return

def bladeInductionOnWakeFilaments(blades, wake, deltaFlts, bladeOrWake):

    bladeOnParticlesKernel_v2 = modFlts.get_function("bladeOnParticlesKernel")

    # Input nodes over which inductions are computed
    nodesPosX = np.zeros(0)
    nodesPosY = np.zeros(0)
    nodesPosZ = np.zeros(0)


    nodesX = blade.wakeNodes[:, :, 0].flatten()
    nodesPosX = np.concatenate((nodesPosX, nodesX))
    nodesY = blade.wakeNodes[:, :, 1].flatten()
    nodesPosY = np.concatenate((nodesPosY, nodesY))
    nodesZ = blade.wakeNodes[:, :, 2].flatten()
    nodesPosZ = np.concatenate((nodesPosZ, nodesZ))


def particlesInductionOnFilaments(blades, wake,
                                  deltaPtcles, ptclesPosX, ptclesPosY, ptclesPosZ, ptclesVorX, ptclesVorY, ptclesVorZ, ptclesRad):

    if (len(wake.particlesRadius) > 0):
        ptclesPosX = wake.particlesPositionX.astype(np.float32)
        ptclesPosY = wake.particlesPositionY.astype(np.float32)
        ptclesPosZ = wake.particlesPositionZ.astype(np.float32)
        ptclesVorX = wake.particlesVorticityX.astype(np.float32)
        ptclesVorY = wake.particlesVorticityY.astype(np.float32)
        ptclesVorZ = wake.particlesVorticityZ.astype(np.float32)
        ptclesRad = wake.particlesRadius.astype(np.float32)

        particlesInductionOnFilamentsKernel = modPtcles.get_function("particlesOnBladesKernel")

        nodesPosX = np.zeros(0)
        nodesPosY = np.zeros(0)
        nodesPosZ = np.zeros(0)

        # Destination wake filament (nodes)
        for blade in blades:
            nodesX = blade.wakeNodes[:, :, 0].flatten()
            nodesPosX = np.concatenate((nodesPosX, nodesX))
            nodesY = blade.wakeNodes[:, :, 1].flatten()
            nodesPosY = np.concatenate((nodesPosY, nodesY))
            nodesZ = blade.wakeNodes[:, :, 2].flatten()
            nodesPosZ = np.concatenate((nodesPosZ, nodesZ))

        nodesPosX = nodesPosX.astype(np.float32)
        nodesPosY = nodesPosY.astype(np.float32)
        nodesPosZ = nodesPosZ.astype(np.float32)

        # Destination velocities
        destUx = np.zeros_like(nodesPosX).astype(np.float32)
        destUy = np.zeros_like(nodesPosY).astype(np.float32)
        destUz = np.zeros_like(nodesPosZ).astype(np.float32)

        numBladePoint = np.int32(len(nodesPosX))
        numParticles = np.int32(len(ptclesPosX))
        threadsPerBlock = 256
        blocksPerGrid = int((len(ptclesPosX) + threadsPerBlock - 1) / threadsPerBlock)
        deltaParticles = np.float32(deltaPtcles)

        particlesInductionOnFilamentsKernel(
            drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), drv.In(nodesPosX), drv.In(nodesPosY),
            drv.In(nodesPosZ), drv.In(ptclesPosX), drv.In(ptclesPosY),
            drv.In(ptclesPosZ), drv.In(ptclesVorX), drv.In(ptclesVorY),
            drv.In(ptclesVorZ), drv.In(ptclesRad), deltaParticles, numBladePoint, numParticles,
            block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1))

        shape = (len(blades), len(blade.trailFilamentsCirculation),
                 blade.nearWakeLength)  # np.shape(blade.wakeNodesInductions[:,:])
        inducedVelX = np.reshape(destUx, shape)
        inducedVelY = np.reshape(destUy, shape)
        inducedVelZ = np.reshape(destUz, shape)
        for (iBlade, myBlade) in enumerate(blades):
            myBlade.wakeNodesInductions[:, :, 0] += inducedVelX[iBlade, :, :] / (4. * np.pi)
            myBlade.wakeNodesInductions[:, :, 1] += inducedVelY[iBlade, :, :] / (4. * np.pi)
            myBlade.wakeNodesInductions[:, :, 2] += inducedVelZ[iBlade, :, :] / (4. * np.pi)

    return


def filamentsInductionOnParticles(blades, wake, deltaFlts):
    if (len(wake.particlesRadius) > 0):

        filamentsInductionOnParticlesKernel = modFlts.get_function("bladeOnParticlesKernel")

        # Destination positions
        ptclesPosX = wake.particlesPositionX.astype(np.float32)
        ptclesPosY = wake.particlesPositionY.astype(np.float32)
        ptclesPosZ = wake.particlesPositionZ.astype(np.float32)

        # Destination velocities
        destUx = np.zeros_like(ptclesPosX).astype(np.float32)
        destUy = np.zeros_like(ptclesPosY).astype(np.float32)
        destUz = np.zeros_like(ptclesPosZ).astype(np.float32)

        # Input filaments positions and circulations
        leftNodesX = np.zeros(0)
        leftNodesY = np.zeros(0)
        leftNodesZ = np.zeros(0)
        rightNodesX = np.zeros(0)
        rightNodesY = np.zeros(0)
        rightNodesZ = np.zeros(0)
        circulations = np.zeros(0)

        for blade in blades:
            # Filament (wake) - Trail
            leftNodesX = np.concatenate((blade.wakeNodes[:, :-1, 0].flatten(), leftNodesX))
            rightNodesX = np.concatenate((blade.wakeNodes[:, 1:, 0].flatten(), rightNodesX))
            leftNodesY = np.concatenate((blade.wakeNodes[:, :-1, 1].flatten(), leftNodesY))
            rightNodesY = np.concatenate((blade.wakeNodes[:, 1:, 1].flatten(), rightNodesY))
            leftNodesZ = np.concatenate((blade.wakeNodes[:, :-1, 2].flatten(), leftNodesZ))
            rightNodesZ = np.concatenate((blade.wakeNodes[:, 1:, 2].flatten(), rightNodesZ))
            circulations = np.concatenate((blade.trailFilamentsCirculation.flatten(), circulations))

            # Filament (wake) - Shed
            leftNodesX = np.concatenate((blade.wakeNodes[:-1, :, 0].flatten(), leftNodesX))
            rightNodesX = np.concatenate((blade.wakeNodes[1:, :, 0].flatten(), rightNodesX))
            leftNodesY = np.concatenate((blade.wakeNodes[:-1, :, 1].flatten(), leftNodesY))
            rightNodesY = np.concatenate((blade.wakeNodes[1:, :, 1].flatten(), rightNodesY))
            leftNodesZ = np.concatenate((blade.wakeNodes[:-1, :, 2].flatten(), leftNodesZ))
            rightNodesZ = np.concatenate((blade.wakeNodes[1:, :, 2].flatten(), rightNodesZ))
            circulations = np.concatenate((blade.shedFilamentsCirculation.flatten(), circulations))

        # Disregard filaments with zero circulation
        idZeroCirc = np.where(circulations == 0.)
        leftNodesX = np.delete(leftNodesX, idZeroCirc)
        rightNodesX = np.delete(rightNodesX, idZeroCirc)
        leftNodesY = np.delete(leftNodesY, idZeroCirc)
        rightNodesY = np.delete(rightNodesY, idZeroCirc)
        leftNodesZ = np.delete(leftNodesZ, idZeroCirc)
        rightNodesZ = np.delete(rightNodesZ, idZeroCirc)
        circulations = np.delete(circulations, idZeroCirc)

        if (len(circulations) > 0):
            fltsLeftNodesX = leftNodesX[:].astype(np.float32)
            fltsLeftNodesY = leftNodesY[:].astype(np.float32)
            fltsLeftNodesZ = leftNodesZ[:].astype(np.float32)
            fltsRightNodesX = rightNodesX[:].astype(np.float32)
            fltsRightNodesY = rightNodesY[:].astype(np.float32)
            fltsRightNodesZ = rightNodesZ[:].astype(np.float32)
            fltsCirculations = circulations.astype(np.float32)

            numParticles = np.int32(len(ptclesPosX))
            numFilaments = np.int32(len(fltsCirculations))
            deltaFlts = np.float32(deltaFlts)

            threadsPerBlock = 256
            blocksPerGrid = int((len(ptclesPosX) + threadsPerBlock - 1) / threadsPerBlock)

            filamentsInductionOnParticlesKernel(
                drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), drv.In(ptclesPosX), drv.In(ptclesPosY),
                drv.In(ptclesPosZ), drv.In(fltsLeftNodesX), drv.In(fltsLeftNodesY),
                drv.In(fltsLeftNodesZ), drv.In(fltsRightNodesX), drv.In(fltsRightNodesY),
                drv.In(fltsRightNodesZ), drv.In(fltsCirculations), numParticles, numFilaments, deltaFlts,
                block=(threadsPerBlock, 1, 1), grid=(blocksPerGrid, 1))

            wake.inducedVelocities[:, 0] += destUx[:] / (4. * np.pi)
            wake.inducedVelocities[:, 1] += destUy[:] / (4. * np.pi)
            wake.inducedVelocities[:, 2] += destUz[:] / (4. * np.pi)

    return


