import numpy as np
from scipy import interpolate


class Airfoil:
    def __init__(self, dataFile, headerLength=0):
        airfoilData = np.genfromtxt(dataFile, skip_header=headerLength)
        self.splineLift = interpolate.interp1d(np.radians(airfoilData[:, 0]), airfoilData[:, 1])
        self.splineDrag = interpolate.interp1d(np.radians(airfoilData[:, 0]), airfoilData[:, 2])

    def getLift(self, aoa):
        return self.splineLift(aoa)

    def getDrag(self, aoa):
        return self.splineDrag(aoa)
