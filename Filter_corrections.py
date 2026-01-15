import numpy as np

class AndromedaFilterCorrection:

    """Photometric data from Liverpool comes in the ugriz filter
    system. This method will convert data from the ugriz system to
    the UBVRI system of PIRATE.
    
    args: u, g
    __call__: Returns B-V colour with uncertainty."""

    def __init__(self, u = None, g = None):
        """Initialise with magnitudes from ugriz filters"""
        self.u = u
        self.g = g

    def __call__(self): 
        """To convert the u and g photometric data to its B-V colour
        using Lupton (2005) calculations."""
        B = self.u - 0.8116*(self.u - self.g) + 0.1313
        B_uncertainty = 0.0095
        V = self.g - 0.2906*(self.u - self.g) + 0.0885
        V_uncertainty = 0.0129
        colour = B-V
        colour_uncert = np.sqrt(B_uncertainty**2 + V_uncertainty**2)
        return colour, colour_uncert

    


