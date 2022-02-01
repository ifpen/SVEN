from wake import *
from scipy.spatial.transform import Rotation as R


class Blade:
    def __init__(self, nodes, nodeChords, airfoils, centersOrientationMatrix, nodesOrientationMatrix,
                 centersTranslationVelocity, nodesTranslationVelocity):

        self.gammaBound = np.zeros(len(nodes) - 1)
        self.newGammaBound = np.zeros(len(nodes) - 1)
        self.oldGammaBound = np.zeros(len(nodes) - 1)
        self.gammaShed = np.zeros(len(nodes) - 1)
        self.gammaTrail = np.zeros(len(nodes))
        self.attackAngle = np.zeros(len(nodes) - 1)

        self.bladeNodes = nodes
        self.trailingEdgeNode = np.zeros(np.shape(nodes))
        self.centers = .5 * (nodes[1:] + nodes[:-1])
        self.nodeChords = nodeChords
        self.centerChords = .5 * (nodeChords[1:] + nodeChords[:-1])
        self.airfoils = airfoils
        self.nodesOrientationMatrix = nodesOrientationMatrix
        self.centersOrientationMatrix = centersOrientationMatrix

        self.centersTranslationVelocity = centersTranslationVelocity
        self.nodesTranslationVelocity = nodesTranslationVelocity

        self.prevCentersTranslationVelocity = centersTranslationVelocity
        self.prevNodesTranslationVelocity = nodesTranslationVelocity

        self.inductionsFromWake = np.zeros([len(self.centers), 3])
        self.inductionsAtNodes = np.zeros([len(self.bladeNodes), 3])

        self.lift = np.zeros((len(self.centers)))
        self.effectiveVelocity = np.zeros((len(self.centers)))

        return

    def storeOldGammaBound(self, gammas):
        for i in range(len(self.centers)):
            self.oldGammaBound[i] = gammas[i]
        return

    def updateSheds(self, newGammaBound):

        self.gammaShed = self.oldGammaBound - newGammaBound

        return

    def updateTrails(self, newGammaBound):

        ghostedNewGammaBound = np.zeros(len(newGammaBound) + 2)
        ghostedNewGammaBound[1:-1] = newGammaBound
        self.gammaTrail = -(ghostedNewGammaBound[1:] - ghostedNewGammaBound[:-1])
        return

    def updateFirstWakeRow(self):

        for i in range(len(self.trailingEdgeNode)):
            dist_to_TE = [self.nodeChords[i] * 3. / 4., 0, 0]
            r = R.from_matrix(self.nodesOrientationMatrix[i])
            dist_to_TE = r.apply(dist_to_TE, inverse=False)

            self.trailingEdgeNode[i] = self.bladeNodes[i] + dist_to_TE

        return

    def updateCentersPos(self):

        for i in range(len(self.centers)):
            dist_to_3_4 = [self.centerChords[i] * 1. / 2., 0, 0]
            r = R.from_matrix(self.centersOrientationMatrix[i])
            dist_to_3_4 = r.apply(dist_to_3_4, inverse=False)

            self.centers[i] = self.centers[i] + dist_to_3_4

        return

    def estimateGammaBound(self, uInfty, nearWakeInducedVelocities):

        relax = 0.25

        newGammaBounds = np.zeros(len(self.centers))
        newGamma = np.zeros(len(self.centers))

        uWind = np.zeros(3)
        uWind[0] = uInfty

        uEffective = uWind - self.centersTranslationVelocity + nearWakeInducedVelocities + \
                     self.inductionsFromWake

        r = R.from_matrix(self.centersOrientationMatrix)
        uEffectiveInElementRef = r.apply(uEffective, inverse=True)

        # 2D assumption
        uEffectiveInElementRef[:, 1] = 0.
        self.attackAngle = np.arctan2(uEffectiveInElementRef[:, 2], uEffectiveInElementRef[:, 0])

        # I could get rid of this for loop yet...
        for i in range(len(self.centers)):
            self.lift[i] = self.airfoils[i].getLift(self.attackAngle[i])

        self.effectiveVelocity = np.linalg.norm(uEffectiveInElementRef, axis=1)

        newGamma = .5 * self.effectiveVelocity * self.centerChords * self.lift
        newGammaBounds = self.gammaBound + relax * (newGamma - self.gammaBound)

        idx = np.where(self.gammaBound == 0)
        newGammaBounds[idx] = newGamma[idx]

        self.gammaBound = newGammaBounds

        return newGammaBounds

    def getNodesAndCirculations(self, includeBoundFilaments):

        length = len(self.bladeNodes) + len(self.centers)
        if(includeBoundFilaments == True):
            length += len(self.centers)

        leftNodes = np.zeros((length,3))
        leftNodes[:len(self.bladeNodes),:] = self.bladeNodes[:,:]
        leftNodes[len(self.bladeNodes):len(self.bladeNodes)+len(self.centers),:] = self.trailingEdgeNode[0:-1,:]
        leftNodes[len(self.bladeNodes)+len(self.centers):,:] = self.bladeNodes[0:-1,:]

        rightNodes = np.zeros((length,3))
        rightNodes[:len(self.bladeNodes),:] = self.trailingEdgeNode[:,:]
        rightNodes[len(self.bladeNodes):len(self.bladeNodes)+len(self.centers),:] = self.trailingEdgeNode[1:]
        rightNodes[len(self.bladeNodes)+len(self.centers):,:] = self.bladeNodes[1:]

        circulations = np.zeros(length)
        circulations[:len(self.bladeNodes)] = self.gammaTrail
        circulations[len(self.bladeNodes):len(self.bladeNodes)+len(self.centers)] = self.gammaShed
        circulations[len(self.bladeNodes)+len(self.centers):] = self.newGammaBound

        return leftNodes, rightNodes, circulations

    def getFilamentsInfo(self, uInftyX, tStep):
        leftNodes = []
        rightNodes = []
        circulations = []

        includeShed = True

        uInfty = np.zeros(3)
        uInfty[0] = uInftyX

        # Trail filaments first
        for i in range(len(self.bladeNodes)):
            leftNodes.append(self.trailingEdgeNode[i])
            rightNodes.append(self.trailingEdgeNode[i] + (
                    uInfty - self.prevNodesTranslationVelocity[i] + self.inductionsAtNodes[i]) * tStep)
            circulations.append(self.gammaTrail[i])

        # Then shed filaments
        if (includeShed):
            for i in range(len(self.centers)):
                leftNodes.append(self.trailingEdgeNode[i] + (
                        uInfty - self.prevNodesTranslationVelocity[i] + self.inductionsAtNodes[i]) * tStep)
                rightNodes.append(self.trailingEdgeNode[i + 1] + (
                        uInfty - self.prevNodesTranslationVelocity[i + 1] + self.inductionsAtNodes[i+1]) * tStep)
                circulations.append(self.gammaShed[i])

        leftNodes = np.asarray(leftNodes)
        rightNodes = np.asarray(rightNodes)
        circulations = np.asarray(circulations)

        return leftNodes, rightNodes, circulations
