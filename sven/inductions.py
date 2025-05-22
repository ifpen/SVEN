from sven.kernels import *

def nearWakeInduction(blades, deltaFlts):
    """
    Computes induced velocities of bound filaments from one blade on another.
    """
    useBoundFilaments = True

    leftNodes = np.zeros((0, 3))
    rightNodes = np.zeros((0, 3))
    circulations = np.zeros(0)

    allBladeInductions = np.zeros((len(blades), len(blades[0].centers), 3))

    for blade in blades:
        bladeLeftNodes, bladeRightNodes, bladeCirculations = (
            blade.getNodesAndCirculations(useBoundFilaments))
        leftNodes = np.concatenate((leftNodes, bladeLeftNodes))
        rightNodes = np.concatenate((rightNodes, bladeRightNodes))
        circulations = np.concatenate((circulations, bladeCirculations))

    for (iblade, blade) in enumerate(blades):
        allBladeInductions[iblade, :, :] = biotSavartFilaments(
            blade.centers, 
            leftNodes, 
            rightNodes, 
            circulations,
            deltaFlts)

    return allBladeInductions


def bladeInductionsOnWake(blades, deltaFlts):
    """
    Computes induced velocities from blades on wake filaments.
    """

    # useBoundFilaments = False
    useBoundFilaments = True

    bladeOnParticlesKernel = modFlts.get_function("inducedVelocityKernel")

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

    nodesPosX = nodesPosX
    nodesPosY = nodesPosY
    nodesPosZ = nodesPosZ
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
        bladeLeftNodes, bladeRightNodes, bladeCirculations = (
            blade.getNodesAndCirculations(useBoundFilaments))
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
    fltsLengthsFix = ((fltsRightNodesX - fltsLeftNodesX) ** 2.
                      + (fltsRightNodesY - fltsLeftNodesY) ** 2.
                      + (fltsRightNodesZ - fltsLeftNodesZ) ** 2.)* deltaFlts**2.

    numParticles = np.int32(len(nodesPosX))
    numFilaments = np.int32(len(fltsCirculations))
    deltaFlts = np.float32(deltaFlts)



    threadsPerBlock = 256
    blocksPerGrid = int((len(nodesPosX) + threadsPerBlock - 1) / threadsPerBlock)
    bladeOnParticlesKernel(
        drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), 
        drv.In(nodesPosX), drv.In(nodesPosY), drv.In(nodesPosZ), 
        drv.In(fltsLeftNodesX), drv.In(fltsLeftNodesY), drv.In(fltsLeftNodesZ), 
        drv.In(fltsRightNodesX), drv.In(fltsRightNodesY), drv.In(fltsRightNodesZ), 
        drv.In(fltsCirculations), drv.In(fltsLengthsFix), numParticles, 
        numFilaments, deltaFlts, block=(threadsPerBlock, 1, 1), 
        grid=(blocksPerGrid, 1))
   


    # WARNING: works only for 1 blade!
    filamentsInductionsUx = destUx[:fltsNodesSize]
    filamentsInductionsUy = destUy[:fltsNodesSize]
    filamentsInductionsUz = destUz[:fltsNodesSize]

    shape = (len(blades), len(blade.trailFilamentsCirculation), blade.nearWakeLength)
    filamentsInductionsUx = np.reshape(filamentsInductionsUx, shape)
    filamentsInductionsUy = np.reshape(filamentsInductionsUy, shape)
    filamentsInductionsUz = np.reshape(filamentsInductionsUz, shape)
    for (iBlade, blade) in enumerate(blades):
        blade.wakeNodesInductions[:, :, 0] += (
            filamentsInductionsUx[iBlade, :, :] / (4. * np.pi))
        blade.wakeNodesInductions[:, :, 1] += (
            filamentsInductionsUy[iBlade, :, :] / (4. * np.pi))
        blade.wakeNodesInductions[:, :, 2] += (
            filamentsInductionsUz[iBlade, :, :] / (4. * np.pi))

    return


def wakeFilamentsInductionsOnBladeOrWake(blades, deltaFlts, bladeOrWake):
    """
    Computes induced velocities from wake filaments on either blade filaments
    or other wake filaments.
    """
    # GPU version
    bladeOnParticlesKernel_v2 = modFlts.get_function("inducedVelocityKernel")

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
        leftNodesX = np.concatenate(
            (blade.wakeNodes[:, :-1, 0].flatten(), leftNodesX))
        rightNodesX = np.concatenate(
            (blade.wakeNodes[:, 1:, 0].flatten(), rightNodesX))
        leftNodesY = np.concatenate(
            (blade.wakeNodes[:, :-1, 1].flatten(), leftNodesY))
        rightNodesY = np.concatenate(
            (blade.wakeNodes[:, 1:, 1].flatten(), rightNodesY))
        leftNodesZ = np.concatenate(
            (blade.wakeNodes[:, :-1, 2].flatten(), leftNodesZ))
        rightNodesZ = np.concatenate(
            (blade.wakeNodes[:, 1:, 2].flatten(), rightNodesZ))
        circulations = np.concatenate(
            (blade.trailFilamentsCirculation.flatten(), circulations))

        # Filament (wake) - Shed
        leftNodesX = np.concatenate(
            (blade.wakeNodes[:-1, :, 0].flatten(), leftNodesX))
        rightNodesX = np.concatenate(
            (blade.wakeNodes[1:, :, 0].flatten(), rightNodesX))
        leftNodesY = np.concatenate(
            (blade.wakeNodes[:-1, :, 1].flatten(), leftNodesY))
        rightNodesY = np.concatenate(
            (blade.wakeNodes[1:, :, 1].flatten(), rightNodesY))
        leftNodesZ = np.concatenate(
            (blade.wakeNodes[:-1, :, 2].flatten(), leftNodesZ))
        rightNodesZ = np.concatenate(
            (blade.wakeNodes[1:, :, 2].flatten(), rightNodesZ))
        circulations = np.concatenate(
            (blade.shedFilamentsCirculation.flatten(), circulations))

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
        destUx = np.zeros_like(nodesPosX, dtype=np.float32)
        destUy = np.zeros_like(nodesPosY, dtype=np.float32)
        destUz = np.zeros_like(nodesPosZ, dtype=np.float32)

        fltsLeftNodesX = leftNodesX[:].astype(np.float32)
        fltsLeftNodesY = leftNodesY[:].astype(np.float32)
        fltsLeftNodesZ = leftNodesZ[:].astype(np.float32)
        fltsRightNodesX = rightNodesX[:].astype(np.float32)
        fltsRightNodesY = rightNodesY[:].astype(np.float32)
        fltsRightNodesZ = rightNodesZ[:].astype(np.float32)
        fltsCirculations = circulations.astype(np.float32)

        fltsLengthsFix = ((fltsRightNodesX-fltsLeftNodesX)**2.
                          +(fltsRightNodesY-fltsLeftNodesY)**2.
                          +(fltsRightNodesZ-fltsLeftNodesZ)**2.)* deltaFlts**2.

        numParticles = np.int32(len(nodesPosX))
        numFilaments = np.int32(len(fltsCirculations))
        deltaFlts = np.float32(deltaFlts)

        threadsPerBlock = 256
        blocksPerGrid = int(
            (len(nodesPosX) + threadsPerBlock - 1) / threadsPerBlock)
        
        bladeOnParticlesKernel_v2(
            drv.Out(destUx), drv.Out(destUy), drv.Out(destUz), 
            drv.In(nodesPosX), drv.In(nodesPosY), drv.In(nodesPosZ), 
            drv.In(fltsLeftNodesX), drv.In(fltsLeftNodesY), drv.In(fltsLeftNodesZ), 
            drv.In(fltsRightNodesX), drv.In(fltsRightNodesY), drv.In(fltsRightNodesZ), 
            drv.In(fltsCirculations), drv.In(fltsLengthsFix), numParticles, 
            numFilaments, deltaFlts, block=(threadsPerBlock, 1, 1), 
            grid=(blocksPerGrid, 1))
        


        if (bladeOrWake == "blade"):
            for blade in blades:
                for i in range(len(blade.centers)):
                    blade.inductionsFromWake[i, 0] += destUx[i] / (4. * np.pi)
                    blade.inductionsFromWake[i, 1] += destUy[i] / (4. * np.pi)
                    blade.inductionsFromWake[i, 2] += destUz[i] / (4. * np.pi)

                for i in range(len(blade.bladeNodes)):
                    blade.inductionsAtNodes[i, 0] += (
                        destUx[i + len(blade.centers)] / (4. * np.pi))
                    blade.inductionsAtNodes[i, 1] += (
                        destUy[i + len(blade.centers)] / (4. * np.pi))
                    blade.inductionsAtNodes[i, 2] += (
                        destUz[i + len(blade.centers)] / (4. * np.pi))

        else:
            shape = (len(blades), len(blade.trailFilamentsCirculation),
                     blade.nearWakeLength)  
            inducedVelX = np.reshape(destUx, shape)
            inducedVelY = np.reshape(destUy, shape)
            inducedVelZ = np.reshape(destUz, shape)
            for (iBlade, myBlade) in enumerate(blades):
                myBlade.wakeNodesInductions[:, :, 0] += (
                    inducedVelX[iBlade, :, :] / (4. * np.pi))
                myBlade.wakeNodesInductions[:, :, 1] += (
                    inducedVelY[iBlade, :, :] / (4. * np.pi))
                myBlade.wakeNodesInductions[:, :, 2] += (
                    inducedVelZ[iBlade, :, :] / (4. * np.pi))

    return

