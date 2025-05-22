from scipy.spatial.transform import Rotation as R
import numpy as np


class Blade:
    """Class to define a blade.

    Parameters
    ----------
    nodes : ndarray
        Array representing the blade nodes.
    nodeChords : ndarray
        Array of chord lengths at each node.
    nearWakeLength : int
        Number of kept filament rows in the wake.
    airfoils : list
        List of Airfoil objects for each section of the blade
    centersOrientationMatrix : ndarray
        Orientation matrices for the centers of each blade section.
    nodesOrientationMatrix : ndarray
        Orientation matrices for each node.
    centersTranslationVelocity : ndarray
        Translation velocity at the blade section centers.
    nodesTranslationVelocity : ndarray
        Translation velocity at each node.
    """
    def __init__(
        self, nodes, nodeChords, nearWakeLength, airfoils, 
        centersOrientationMatrix, nodesOrientationMatrix,
        centersTranslationVelocity, nodesTranslationVelocity):

        self.nearWakeLength = nearWakeLength

        self.gammaBound = np.zeros(len(nodes) - 1, dtype=np.float32)
        self.newGammaBound = np.zeros(len(nodes) - 1, dtype=np.float32)
        self.oldGammaBound = np.zeros(len(nodes) - 1, dtype=np.float32)
        self.gammaShed = np.zeros(len(nodes) - 1, dtype=np.float32)
        self.gammaTrail = np.zeros(len(nodes), dtype=np.float32)
        self.attackAngle = np.zeros(len(nodes) - 1, dtype=np.float32)

        self.bladeNodes = nodes
        self.trailingEdgeNode = np.zeros(np.shape(nodes), dtype=np.float32)
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

        self.inductionsFromWake = np.zeros(
            [len(self.centers), 3], dtype=np.float32)
        self.inductionsAtNodes = np.zeros(
            [len(self.bladeNodes), 3], dtype=np.float32)

        self.lift = np.zeros((len(self.centers)), dtype=np.float32)
        self.drag = np.zeros((len(self.centers)), dtype=np.float32)

        self.effectiveVelocity = np.zeros((len(self.centers)), dtype=np.float32)

        self.wakeNodesInductions = np.zeros(
            [len(self.bladeNodes), self.nearWakeLength,3], dtype=np.float32)
        self.trailFilamentsCirculation = np.zeros(
            [len(self.bladeNodes), self.nearWakeLength-1], dtype=np.float32)
        self.shedFilamentsCirculation = np.zeros(
            [len(self.bladeNodes)-1, self.nearWakeLength], dtype=np.float32)

        self.wakeNodes = np.zeros(
            [len(self.bladeNodes), self.nearWakeLength,3], dtype=np.float32)


        return

    def initializeWake(self):
        """
        Initialize the wake positions by setting all wake nodes to the trailing
        edge position.
        """
      
        for i in range(self.nearWakeLength):
            for j in range(len(self.bladeNodes)):
                self.wakeNodes[j,i,:] = self.trailingEdgeNode[j,:] 
        return

    def updateFilamentCirulations(self):
        """
        Update the circulation values for the first row of filaments in the wake.
        """
        self.trailFilamentsCirculation[:,0] = self.gammaTrail
        self.shedFilamentsCirculation[:,0] = self.gammaShed
        return

    def spliceNearWake(self):
        """
        Shift the circulation values associated to a wake node one step downstream
        to simulate advection.
        """
        self.wakeNodes[:,1:] = self.wakeNodes[:,:-1]
        self.trailFilamentsCirculation[:,1:] = self.trailFilamentsCirculation[:,:-1]
        self.shedFilamentsCirculation[:, 1:] = self.shedFilamentsCirculation[:, :-1]

        self.trailFilamentsCirculation[:,0] = 0.
        self.shedFilamentsCirculation[:,0] = 0.

        return

    def advectFilaments(self, uInfty, timeStep):
        """
        Advect the wake nodes based on the wind velocity and induced velocities
        using a forward Euler scheme.
        """

        wind = np.zeros(3)
        wind[0] = uInfty
        self.wakeNodes += wind*timeStep + self.wakeNodesInductions*timeStep
        return

    def storeOldGammaBound(self, gammas):
        """
        Stores the current value of bound circulation (circulation associated
        to blade nodes) into a historical variable.
        """

        self.oldGammaBound = gammas

        return

    def updateSheds(self, newGammaBound):
        """
        Updates shed filament's circulation values by calculating the difference
        between the old and new values of bound circulation (Kelvin's Theorem).
        """

        self.gammaShed = self.oldGammaBound - newGammaBound

        return

    def updateTrails(self, newGammaBound):
        """
        Update trail filament's circulation values based on the differences 
        between bound circulation values of adjacent sections.
        """

        ghostedNewGammaBound = np.zeros(len(newGammaBound) + 2)
        ghostedNewGammaBound[1:-1] = newGammaBound
        self.gammaTrail = -(ghostedNewGammaBound[1:] - ghostedNewGammaBound[:-1])
        return

    def updateFirstWakeRow(self):
        """
        Updates the positions of the first row of wake nodes based on the trailing
        edge coordinates of the blade. 
        """
        for i in range(len(self.trailingEdgeNode)):
            dist_to_TE = [self.nodeChords[i] * 3. / 4., 0, 0]
            r = R.from_matrix(self.nodesOrientationMatrix[i])
            dist_to_TE = r.apply(dist_to_TE, inverse=False)
            self.trailingEdgeNode[i] = self.bladeNodes[i] + dist_to_TE
            
        self.wakeNodes[:,0,:] = self.trailingEdgeNode
        return


    def estimateGammaBound(self, uInfty, nearWakeInducedVelocities):
        """
        Estimates bound circulation associated to each blade section based on 
        blade-element theory and Kutta-Joukowski theorem.
        """

        relax = 0.05

        uWind = np.zeros(3)
        uWind[0] = uInfty

        uEffective = (
            uWind - self.centersTranslationVelocity + nearWakeInducedVelocities + 
            self.inductionsFromWake)

        r = R.from_matrix(self.centersOrientationMatrix)
        uEffectiveInElementRef = r.apply(uEffective, inverse=True)

        # 2D assumption
        uEffectiveInElementRef[:, 1] = 0.
        self.attackAngle = np.arctan2(
            uEffectiveInElementRef[:, 2], uEffectiveInElementRef[:, 0])


        for i in range(len(self.centers)):
            self.lift[i] = self.airfoils[i].getLift(self.attackAngle[i])
            self.drag[i] = self.airfoils[i].getDrag(self.attackAngle[i])

        self.effectiveVelocity = np.linalg.norm(uEffectiveInElementRef, axis=1)


        newGamma = .5 * self.effectiveVelocity * self.centerChords * self.lift
        newGammaBounds = self.gammaBound + relax * (newGamma - self.gammaBound)

        idx = np.where(self.gammaBound == 0)
        newGammaBounds[idx] = newGamma[idx]

        self.gammaBound = newGammaBounds

        return newGammaBounds

    def getNodesAndCirculations(self, includeBoundFilaments):
        """
        Retrieves the coordinates of left and right nodes of each vortex filament
        along with the associated circulations.
        """

        length = len(self.bladeNodes) + len(self.centers)
        if(includeBoundFilaments == True):
            length += len(self.centers)

        leftNodes = np.zeros((length,3), dtype=np.float32)
        leftNodes[:len(self.bladeNodes),:] = self.bladeNodes[:,:]
        leftNodes[len(self.bladeNodes):len(self.bladeNodes)+len(self.centers),:] = (
            self.trailingEdgeNode[0:-1,:])
        leftNodes[len(self.bladeNodes)+len(self.centers):,:] = (
            self.bladeNodes[0:-1,:])

        rightNodes = np.zeros((length,3), dtype=np.float32)
        rightNodes[:len(self.bladeNodes),:] = self.trailingEdgeNode[:,:]
        rightNodes[len(self.bladeNodes):len(self.bladeNodes)+len(self.centers),:] = (
            self.trailingEdgeNode[1:])
        rightNodes[len(self.bladeNodes)+len(self.centers):,:] = (
            self.bladeNodes[1:])

        circulations = np.zeros(length, dtype=np.float32)
        circulations[:len(self.bladeNodes)] = self.gammaTrail
        circulations[len(self.bladeNodes):len(self.bladeNodes)+len(self.centers)] = (
            self.gammaShed)
        circulations[len(self.bladeNodes)+len(self.centers):] = self.newGammaBound

        return leftNodes, rightNodes, circulations

