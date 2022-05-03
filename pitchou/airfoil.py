import numpy as np
from numba import jit, prange, njit

@njit(fastmath=True)
def interp_checked(altitudes, alltemps, location):
    idx = max(1, np.searchsorted(altitudes, location, side='left'))

    if(idx >= len(altitudes)-1):
        return alltemps[-1]
    else:
        x1 = altitudes[idx-1]
        x2 = altitudes[idx]

        y1 = alltemps[idx-1]
        y2 = alltemps[idx]
        yI = y1 + (y2-y1) * (location-x1) / (x2-x1)

        return yI

class Airfoil:
    def __init__(self, dataFile, headerLength=0):
        airfoilData = np.genfromtxt(dataFile, skip_header=headerLength)

        sortedIndices = np.argsort(airfoilData[:, 0])
        self.AOAs = np.radians(airfoilData[sortedIndices, 0])
        self.Lifts = airfoilData[sortedIndices,1]
        self.Drags = airfoilData[sortedIndices,2]

    def getLift(self, aoa):
        return interp_checked(self.AOAs, self.Lifts, aoa)

    def getDrag(self, aoa):
        return interp_checked(self.AOAs, self.Drags, aoa)
