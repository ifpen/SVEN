import numpy as np
from numba import jit, njit
import os



class Airfoil:
    """Class to define an airfoil.

    Parameters
    ----------
    dataFile : str
        Path to .foil file with the polars definition.
    headerLength : int, optional
        Length of the file header to skip (default is 0).
    """

    def __init__(self, dataFile, headerLength=0):

        try:
            airfoilData = np.genfromtxt(dataFile, skip_header=headerLength)
        except OSError as e:
            cwd = os.getcwd()
            raise FileNotFoundError(
                f"Could not find airfoil data file '{dataFile}'.\n"
                f"Current working directory: {cwd}\n"
                f"Please update the path to your data file."
            ) from e

        sortedIndices = np.argsort(airfoilData[:, 0])
        self.AOAs = np.radians(airfoilData[sortedIndices, 0])
        self.Lifts = airfoilData[sortedIndices, 1]
        self.Drags = airfoilData[sortedIndices, 2]

    def getLift(self, aoa):
        """
        Returns the interpolated lift coefficient for a given angle of attack.

        """
        return interp_checked(self.AOAs, self.Lifts, aoa)
        

    def getDrag(self, aoa):
        """
        Returns the interpolated drag coefficient for a given angle of attack.

        """
        return interp_checked(self.AOAs, self.Drags, aoa)



@njit(fastmath=True)
def interp_checked(altitudes, alltemps, location):
    """
        Performs a linear interpolation while handling out-of-bounds cases.
        njit is used for interpolation speed-up.

    """
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