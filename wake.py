import numpy as np
from numba import jit, prange, njit

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

    def addParticlesFromFilaments(self, leftNodes, rightNodes, circulations):
        # Particle Position
        particlePositions = .5 * (leftNodes + rightNodes)

        for i in range(len(circulations)):
            filamentLength = np.linalg.norm(rightNodes[i] - leftNodes[i])

            unitVector = (rightNodes[i] - leftNodes[i]) / filamentLength

            particleRadius = 0.75 * filamentLength
            # particleCoreSize = filamentLength * (1.0 / (1.0 + 0.1))
            # initialCoreSize = particleCoreSize

            # print('particle Vorticity: ', circulations[i] * filamentLength * unitVector)
            particleVorticity = circulations[i] * filamentLength * unitVector

            self.particlesPositionX = np.append(self.particlesPositionX, particlePositions[i][0])
            self.particlesPositionY = np.append(self.particlesPositionY, particlePositions[i][1])
            self.particlesPositionZ = np.append(self.particlesPositionZ, particlePositions[i][2])

            self.particlesVorticityX = np.append(self.particlesVorticityX, particleVorticity[0])
            self.particlesVorticityY = np.append(self.particlesVorticityY, particleVorticity[1])
            self.particlesVorticityZ = np.append(self.particlesVorticityZ, particleVorticity[2])

            self.particlesRadius = np.append(self.particlesRadius, particleRadius)
            # self.particlesCoreSize = np.append(self.particlesCoreSize, particleCoreSize)
            # self.initialCoreSize = np.append(self.initialCoreSize, initialCoreSize)
        return

    def addParticlesFromFilaments_2(self, leftNodes, rightNodes, circulations, denom, firstRightCoef, firstLeftCoef, numPart):
        # Particle Position
        # particlePositions = .5 * (leftNodes + rightNodes)
        # particlePositions = ptcPos * (leftNodes + rightNodes)

        particlePositions_1 = 1. / denom * (firstLeftCoef * leftNodes + firstRightCoef * rightNodes)

        for i in range(len(circulations)):
            filamentLength = 1./numPart * np.linalg.norm(rightNodes[i] - leftNodes[i])

            unitVector = 1./numPart * (rightNodes[i] - leftNodes[i]) / filamentLength

            particleRadius = .5 * filamentLength
            particleCoreSize = filamentLength * (1.0 / (1.0 + 0.1))
            initialCoreSize = particleCoreSize

            particleVorticity = circulations[i] * filamentLength * unitVector

            self.particlesPositionX = np.append(self.particlesPositionX, particlePositions_1[i][0])
            self.particlesPositionY = np.append(self.particlesPositionY, particlePositions_1[i][1])
            self.particlesPositionZ = np.append(self.particlesPositionZ, particlePositions_1[i][2])

            self.particlesVorticityX = np.append(self.particlesVorticityX, particleVorticity[0])
            self.particlesVorticityY = np.append(self.particlesVorticityY, particleVorticity[1])
            self.particlesVorticityZ = np.append(self.particlesVorticityZ, particleVorticity[2])

            self.particlesRadius = np.append(self.particlesRadius, particleRadius)
            self.particlesCoreSize = np.append(self.particlesCoreSize, particleCoreSize)
            self.initialCoreSize = np.append(self.initialCoreSize, initialCoreSize)

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
