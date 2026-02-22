"""
author: @jp

TGP Cepheids 25-26

These are just some helper functions I made at the very start of the project. 

I ended up running with them because it is convenient but found out at the end of the project that
MJD can actually just come straight from the .fits headers (I was working in ISOT). It is not really worth
refactoring the code to account for this.

Ignore the magnitude error function.

"""

# default packages
import numpy as np
from astropy.time import Time

class Astro_Functions:

    def magnitude_error(snr: float):
        """
        Compute error on magnitudes.

        DEFUNCT!!!
        
        """
        return (2.5 / np.log(10)) * (1 / snr)

    def apparent_to_absolute(magnitude: float, distance: float):
        """
        Convert apparent magnitudes to absolute magnitudes.
        """
        return magnitude - (5 * np.log10(distance / 10))
    
    def modified_julian_date_converter(time):
        """
        Converts ISO times to MJD for ease of use.
        """
        time_array = np.array(time)
        t = Time(time_array, scale='utc', format='isot')
        return t.mjd # astronomers use modified julian date