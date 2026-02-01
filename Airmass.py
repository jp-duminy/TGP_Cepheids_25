import astropy.io.fits as fits

class Airmass:
    """Extract airmass data from fits header"""
    def __init__(self, filename):
        with fits.open(filename) as hdul:
            header = hdul[0].header 
        
        if "AIRMASS" not in header:
            raise AttributeError("AIRMASS doesn't exist in this header. Brill.")

        self.airmass = header["AIRMASS"]

        
    