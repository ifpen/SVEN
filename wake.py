import numpy as np
from numba import jit, prange, njit

@njit(fastmath=True)
def addParticlesFromFilaments_Jit(leftNodes, rightNodes, circulations, particlesPerFil):

    newPosX = np.zeros(len(circulations) * particlesPerFil)
    newPosY = np.zeros(len(circulations) * particlesPerFil)
    newPosZ = np.zeros(len(circulations) * particlesPerFil)

    newVorX = np.zeros(len(circulations) * particlesPerFil)
    newVorY = np.zeros(len(circulations) * particlesPerFil)
    newVorZ = np.zeros(len(circulations) * particlesPerFil)

    newRadius = np.zeros(len(circulations) * particlesPerFil)

    ptclesCounter = 0

    for i in range(len(circulations)):
        for numParticle in range(particlesPerFil):
            rightCoef = numParticle + 1
            leftCoef = particlesPerFil - numParticle

            particlePosition = 1. / (particlesPerFil + 1.) * (leftCoef * leftNodes[i] + rightCoef * rightNodes[i])

            filamentLength = np.linalg.norm(rightNodes[i] - leftNodes[i])
            unitVector = (rightNodes[i] - leftNodes[i]) / filamentLength
            particleRadius = .5 * filamentLength
            particleVorticity = circulations[i] * filamentLength * unitVector / particlesPerFil

            newPosX[ptclesCounter] = particlePosition[0]
            newPosY[ptclesCounter] = particlePosition[1]
            newPosZ[ptclesCounter] = particlePosition[2]

            newVorX[ptclesCounter] = particleVorticity[0]
            newVorY[ptclesCounter] = particleVorticity[1]
            newVorZ[ptclesCounter] = particleVorticity[2]

            newRadius[ptclesCounter] = particleRadius

            ptclesCounter += 1

    return newPosX, newPosY, newPosZ, newVorX, newVorY, newVorZ, newRadius

@njit(fastmath=True)
def advectJit(posX, posY, posZ, vels, uInf, timeStep, length):
    for i in range(length):
        advectionDistance = np.asarray(uInf + vels[i]) * timeStep

        posX[i] += advectionDistance[0]
        posY[i] += advectionDistance[1]
        posZ[i] += advectionDistance[2]

    return posX, posY, posZ


class Wake:
    def __init__(self):
        self.particlesPositionX = np.zeros(0)
        self.particlesPositionY = np.zeros(0)
        self.particlesPositionZ = np.zeros(0)

        self.particlesVorticityX = np.zeros(0)
        self.particlesVorticityY = np.zeros(0)
        self.particlesVorticityZ = np.zeros(0)

        self.particlesRadius = np.zeros(0)
        self.particlesCoreSize = np.zeros(0)
        self.initialCoreSize = np.zeros(0)

        self.inducedVelocities = np.zeros([0, 3])

        return



    def addParticlesFromFilaments_50(self, leftNodes, rightNodes, circulations, particlesPerFil):

        # newPosX = np.zeros(len(circulations) * particlesPerFil)
        # newPosY = np.zeros(len(circulations) * particlesPerFil)
        # newPosZ = np.zeros(len(circulations) * particlesPerFil)
        #
        # newVorX = np.zeros(len(circulations) * particlesPerFil)
        # newVorY = np.zeros(len(circulations) * particlesPerFil)
        # newVorZ = np.zeros(len(circulations) * particlesPerFil)
        #
        # newRadius = np.zeros(len(circulations) * particlesPerFil)
        #
        # ptclesCounter = 0
        #
        # for i in range(len(circulations)):
        #     for numParticle in range(particlesPerFil):
        #
        #         rightCoef = numParticle + 1
        #         leftCoef = particlesPerFil - numParticle
        #
        #         particlePosition = 1. / (particlesPerFil+1.) * (leftCoef * leftNodes[i] + rightCoef * rightNodes[i])
        #
        #
        #         filamentLength =  np.linalg.norm(rightNodes[i] - leftNodes[i])
        #         unitVector = (rightNodes[i] - leftNodes[i]) / filamentLength
        #         particleRadius = .5 * filamentLength
        #         particleVorticity = circulations[i] * filamentLength * unitVector / particlesPerFil
        #
        #         newPosX[ptclesCounter] = particlePosition[0]
        #         newPosY[ptclesCounter] = particlePosition[1]
        #         newPosZ[ptclesCounter] = particlePosition[2]
        #
        #         newVorX[ptclesCounter] = particleVorticity[0]
        #         newVorY[ptclesCounter] = particleVorticity[1]
        #         newVorZ[ptclesCounter] = particleVorticity[2]
        #
        #         newRadius[ptclesCounter] = particleRadius
        #
        #         ptclesCounter += 1

        newPosX, newPosY, newPosZ, newVorX, newVorY, newVorZ, newRadius = addParticlesFromFilaments_Jit(leftNodes, rightNodes, circulations, particlesPerFil)

        self.particlesPositionX = np.concatenate((self.particlesPositionX, newPosX), axis=0)
        self.particlesPositionY = np.concatenate((self.particlesPositionY, newPosY), axis=0)
        self.particlesPositionZ = np.concatenate((self.particlesPositionZ, newPosZ), axis=0)

        self.particlesVorticityX = np.concatenate((self.particlesVorticityX, newVorX), axis=0)
        self.particlesVorticityY = np.concatenate((self.particlesVorticityY, newVorY), axis=0)
        self.particlesVorticityZ = np.concatenate((self.particlesVorticityZ, newVorZ), axis=0)

        self.particlesRadius = np.concatenate((self.particlesRadius, newRadius), axis=0)

        return




    def advectParticles(self, uInfty, timeStep):
        posX = np.asarray(self.particlesPositionX)
        posY = np.asarray(self.particlesPositionY)
        posZ = np.asarray(self.particlesPositionZ)
        vels = np.asarray(self.inducedVelocities)

        self.particlesPositionX, self.particlesPositionY, self.particlesPositionZ = advectJit(posX, posY, posZ, vels,
                                                                                              np.asarray(
                                                                                                  [uInfty, 0., 0.]),
                                                                                              timeStep, len(posX))
        return
