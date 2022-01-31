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
        # print(self.gammaBound, self.oldGammaBound, self.gammaBound-self.oldGammaBound)
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

        uWind = np.zeros(3)
        uWind[0] = uInfty

        # Project to blade element reference frame
        for i in range(len(self.centers)):
            uEffective = uWind - self.centersTranslationVelocity[i] + nearWakeInducedVelocities[i] + \
                         self.inductionsFromWake[i]
            #
            r = R.from_matrix(self.centersOrientationMatrix[i])
            uEffectiveInElementRef = r.apply(uEffective, inverse=True)

            # 2D assumption
            uEffectiveInElementRef[1] = 0.
            self.attackAngle[i] = np.arctan2(uEffectiveInElementRef[2], uEffectiveInElementRef[0])
            self.lift[i] = self.airfoils[i].getLift(self.attackAngle[i])
            self.effectiveVelocity[i] = np.linalg.norm(uEffectiveInElementRef)

            newGamma = .5 * np.linalg.norm(uEffectiveInElementRef) * self.centerChords[i] * self.lift[i]
            newGammaBounds[i] = self.gammaBound[i] + relax * (newGamma - self.gammaBound[i])

            if (self.gammaBound[i] == 0):
                newGammaBounds[i] = newGamma
            self.gammaBound[i] = newGammaBounds[i]

        return newGammaBounds

    def getNodesAndCirculations(self, includeBoundFilaments):
        leftNodes = []
        rightNodes = []
        circulations = []

        # Trail filaments first
        for i in range(len(self.bladeNodes)):
            leftNodes.append(self.bladeNodes[i])
            rightNodes.append(self.trailingEdgeNode[i])
            circulations.append(self.gammaTrail[i])

        # Then shed filaments
        for i in range(len(self.centers)):
            leftNodes.append(self.trailingEdgeNode[i])
            rightNodes.append(self.trailingEdgeNode[i + 1])
            circulations.append(self.gammaShed[i])

        # Maybe bound filaments also need to be considered here
        if (includeBoundFilaments):
            for i in range(len(self.centers)):
                leftNodes.append(self.bladeNodes[i])
                rightNodes.append(self.bladeNodes[i + 1])
                circulations.append(self.newGammaBound[i])

        leftNodes = np.asarray(leftNodes)
        rightNodes = np.asarray(rightNodes)
        circulations = np.asarray(circulations)

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
