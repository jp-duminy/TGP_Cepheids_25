"""Unfinished as of 05/11/25"""

import numpy as np
from photutils.aperture import CircularAperture as circ_ap, CircularAnnulus as circ_ann, \
    aperture_photometry as ap, ApertureMask as mask, SkyAperture as sky_ap, \
    RectangularAperture as rect_ap
from photutils.centroids import centroid_2dg
import photutils.psf as psf
import astropy.io.fits as fits
import astropy.wcs as wcs
import matplotlib.pyplot as plt


class Aperture_Photometry: 

    """A class to perform basic aperture photometry on a dataset, up to and
    including determining the instrumental magnitude of a target."""

    def __init__(self, filename): 

        """Initialise with dataset in form of 2D Numpy array derived from
        .fits file."""

        #Load data and header from FITS file as an HDU list 
        #Remember HDU list may have several indices
        #Perhaps ask Kenneth about HDU index

        with fits.open(filename) as hdul:
            data = hdul[0]._data
            header = hdul[0]._data

        if data.ndim != 2:
            raise ValueError(f"The image is {data.ndim}D, not 2D")
        
        data = data.astype(float)
        data = np.nan_to_num(data)

        self.data = data
        self.header = header

    def get_pixel_coords(self, WCS, RA, Dec, origin = 0):
        """Converts world coordinates of targets (RA, Dec) to pixel coordinates (x, y). 
        Requires argument of a WCS object and RA & Dec as seperate arrays
        (see wcs.WCS documentation)."""
        if isinstance(RA, float) == False or isinstance(Dec, float) == False:
            raise TypeError(f"These coordinates must be floats. If in arrays, try \
                            doing them one at a time.")
        
        x, y = WCS.wcs_world2pix(RA, Dec, origin)
        #origin is 0 for numpy array, 1 for FITS file
        return x, y
        #NB: FITS file might by upside down by the time this is used, could cause issues. 
    
    def mask_data_and_plot(self, x, y, width, plot = False):
        """Set boolean mask to cut out a square shape around the target, to remove
        other sources. Plot masked data as heatmap if plot == True, don't otherwise"""
        aperture = rect_ap((x,y), width, width)
        mask = aperture.to_mask(method = "center")
        masked_data = mask.multiply(self.data)
    
        if plot == True:
            plt.imshow(masked_data)
            plt.show()
        
        return masked_data


    def get_centroid(self, x, y, width):
        """Get Gaussian centroid of target source around which to
        centre the aperture."""
        centroid = centroid_2dg(self.mask_data(x, y, width, width))
        return centroid
    
    def get_fwhm(self, x, y, width):
        """Get the FWHM of target source centred around the centroid"""
        fwhm = psf.fit_fwhm(data = self.mask_data(x, y, width, width), 
                            xypos = self.get_centroid(x, y, width, width))
        #Function expects data to be bkgd subtracted
        #Nan/inf values automatically masked
        return fwhm
    
    def 

    



    


        

    


    





