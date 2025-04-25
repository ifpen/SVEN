from sven.blade import *

from scipy.spatial.transform import Rotation as R


class windTurbine:
    def __init__(self, nBlades, hubCenter, hubRadius, rotationalVelocity, windVelocity, bladePitch):

        self.blades = []
        self.bladeRootOrientation = []
        self.hubCenter = hubCenter
        self.hubRadius = hubRadius
        self.nBlades = nBlades
        self.rotationalVelocity = rotationalVelocity
        self.windVelocity = windVelocity
        self.bladePitch = bladePitch
        self.nNodes = 0

        self.currentAzimuth = 0.
        self.nodesRadius = 0.
        self.nodesChord = 0.
        self.centersAirfoils = 0.
        self.nodesTwistAngles = 0.

        return

    def updateTurbine(self, currentAzimuth):

        self.nNodes = len(self.nodesRadius)
        self.bladeRootOrientation = []

        # Evaluate centers twist angles
        centersRadius = .5 * (self.nodesRadius[1:] + self.nodesRadius[:-1])
        centersTwist = .5 * (self.nodesTwistAngles[1:] + self.nodesTwistAngles[:-1])

        for iBlade in range(self.nBlades):

            self.blades[iBlade].prevCentersTranslationVelocity = self.blades[iBlade].centersTranslationVelocity
            self.blades[iBlade].prevNodesTranslationVelocity = self.blades[iBlade].nodesTranslationVelocity

            # Build orientation matrices
            centersOrientationMatrix = []
            nodesOrientationMatrix = []
            centersTranslationVelocity = []
            nodesTranslationVelocity = []

            # Build first blade nodes
            r = R.from_euler('x', np.degrees(currentAzimuth) + iBlade * 360. / self.nBlades, degrees=True)
            self.bladeRootOrientation.append(r)

            nodes = []
            for i in range(self.nNodes):
                basicPosition = [0., self.nodesRadius[i], 0.]
                alongAzimuthPosition = r.apply(basicPosition)
                nodes.append(alongAzimuthPosition + self.hubCenter)

            nodes = np.asarray(nodes, dtype=np.float32)

            # Azimuth around x, pitch and twist around y
            for i in range(self.nNodes - 1):
                r1 = R.from_euler('x', np.degrees(currentAzimuth) + iBlade * 360. / self.nBlades, degrees=True)
                r2 = R.from_euler('y', 90. - np.degrees(self.bladePitch + centersTwist[i]), degrees=True)

                R1 = r1.as_matrix()
                R2 = r2.as_matrix()

                centersOrientationMatrix.append(np.dot(R1, R2))

                # Evaluate elements translation velocity
                # Velocity in hub reference frame, assuming not tilt, yaw, etc.
                centerTranslationVelocity = np.asarray([0., 0., self.rotationalVelocity * centersRadius[i]], dtype=np.float32)
                # Projection into the "global" reference frame
                centerTranslationVelocity = self.bladeRootOrientation[iBlade].apply(centerTranslationVelocity,
                                                                                    inverse=False)

                centersTranslationVelocity.append(centerTranslationVelocity)

            for i in range(self.nNodes):
                r1 = R.from_euler('x', np.degrees(currentAzimuth) + iBlade * 360. / self.nBlades, degrees=True)
                r2 = R.from_euler('y', 90. - np.degrees(self.bladePitch + self.nodesTwistAngles[i]), degrees=True)

                R1 = r1.as_matrix()
                R2 = r2.as_matrix()
                nodesOrientationMatrix.append(np.dot(R1, R2))

                nodeTranslationVelocity = np.asarray([0., 0., self.rotationalVelocity * self.nodesRadius[i]], dtype=np.float32)
                # Projection into the "global" reference frame
                nodeTranslationVelocity = self.bladeRootOrientation[iBlade].apply(nodeTranslationVelocity,
                                                                                  inverse=False)
                nodesTranslationVelocity.append(nodeTranslationVelocity)

            nodes = np.asarray(nodes)
            self.blades[iBlade].bladeNodes = nodes
            self.blades[iBlade].centersTranslationVelocity = np.asarray(centersTranslationVelocity, dtype=np.float32)
            self.blades[iBlade].nodesTranslationVelocity = np.asarray(nodesTranslationVelocity, dtype=np.float32)
            self.blades[iBlade].nodesOrientationMatrix = np.asarray(nodesOrientationMatrix, dtype=np.float32)
            self.blades[iBlade].centersOrientationMatrix = np.asarray(centersOrientationMatrix, dtype=np.float32)
            self.blades[iBlade].centers = .5 * (nodes[1:] + nodes[:-1])


        return self.blades

    def initializeTurbine(self, nodesRadius, nodesChord, nearWakeLength, centersAirfoils, nodesTwistAngles, nBlades):

        self.nodesRadius = nodesRadius
        self.nodesChord = nodesChord
        self.centersAirfoils = centersAirfoils
        self.nodesTwistAngles = nodesTwistAngles


        nbNodes = len(nodesRadius)
        for ib in range(nBlades):
            self.blades.append( Blade(np.zeros([nbNodes,3], dtype=np.float32), nodesChord, nearWakeLength, centersAirfoils, np.zeros(nbNodes, dtype=np.float32), np.zeros(nbNodes, dtype=np.float32),
                  np.zeros(nbNodes, dtype=np.float32), np.zeros(nbNodes, dtype=np.float32)))
            self.blades[ib].centerChords = .5 * (self.blades[ib].nodeChords[1:] + self.blades[ib].nodeChords[:-1])

        blades = self.updateTurbine(0.)

        for blade in blades:
            blade.updateFirstWakeRow()
            blade.initializeWake()

        return blades

    def evaluateForces(self, density):

        attackAngles = self.blades[0].attackAngle
        lift = self.blades[0].lift
        drag = []
        for i in range(len(self.blades[0].centers)):
            drag .append( self.blades[0].airfoils[i].getDrag(self.blades[0].attackAngle[i]))
        drag = np.asarray(drag)

        cn = lift * np.cos(attackAngles) + drag*np.sin(attackAngles)
        ct = lift * np.sin(attackAngles) - drag*np.cos(attackAngles)

        inScale = .5*density*self.blades[0].effectiveVelocity**2.*self.blades[0].centerChords

        return inScale*cn, inScale*ct