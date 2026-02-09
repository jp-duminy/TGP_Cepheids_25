import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
import re

from Cepheid_apertures import AperturePhotometry
from Cepheid_apertures import Airmass
from AirmassInfo import AirmassInfo

# someone please double check these

# cepheids and their RA/Dec coords (I did this by hand and double checked)
cepheid_catalog = cepheid_catalog = {
    "01": {
        "ra":  "20 12 22.83",
        "dec": "+32 52 17.8",
        "e(b-v)": "0.68"
    },
    "02": {
        "ra":  "20 54 57.53",
        "dec": "+47 32 01.7",
        "e(b-v)": "0.80"
    },
    "03": {
        "ra":  "20 57 20.83",
        "dec": "+40 10 39.1",
        "e(b-v)": "0.79"
    },
    "04": {
        "ra":  "21 04 16.63",
        "dec": "+39 58 20.1",
        "e(b-v)": "0.65"
    },
    "05": {
        "ra":  "21 57 52.69",
        "dec": "+56 09 50.0",
        "e(b-v)": "0.68"
    },
    "06": {
        "ra":  "22 40 52.15",
        "dec": "+56 49 46.1",
        "e(b-v)": "0.40"
    },
    "07": {
        "ra":  "22 48 38.00",
        "dec": "+56 19 17.5",
        "e(b-v)": "0.36"
    },
    "08": {
        "ra":  "23 07 10.08",
        "dec": "+58 33 15.1",
        "e(b-v)": "0.49"
    },
    "09": {
        "ra":  "00 26 19.45",
        "dec": "+51 16 49.3",
        "e(b-v)": "0.12"
    },
    "10": {
        "ra":  "00 29 58.59",
        "dec": "+60 12 43.1",
        "e(b-v)": "0.53"
    },
    "11": {
        "ra":  "01 32 43.22",
        "dec": "+63 35 37.7",
        "e(b-v)": "0.70"
    },
}

# standard star catalog, I also did this by hand
standard_catalog = {
    "114176": {
        "ra":  "+22 43 11.0",
        "dec": "+00 21 16.0",
        "mag": "9.239",
        "e(b-v)": "0.0013"
    },
    "SA111775": {
        "ra":  "+19 37 17.0",
        "dec": "+00 11 14.0",
        "mag": "10.74",
        "e(b-v)": "	0.0009"
    },
    "F_108": {
        "ra":  "+23 16 12.0",
        "dec": "-01 50 35.0",
        "mag": "12.96",
        "e(b-v)": "0.0016"
    },
    "SA112_595": {
        "ra":  "+20 41 19.0",
        "dec": "+00 16 11.0",
        "mag": "11.35",
        "e(b-v)": "0.0016"
    },
    "GD_246": {
        "ra":  "+23 12 21.6",
        "dec": "+10 47 04.0",
        "mag": "13.09",
        "e(b-v)": "0.0015"
    },
    "G93_48": {
        "ra":  "+21 52 25.4",
        "dec": "+02 23 23.0",
        "mag": "12.74",
        "e(b-v)": "0.0012"
    },
    "G156_31": {
        "ra":  "+22 38 28.0",
        "dec": "-15 19 17.0",
        "mag": "12.36",
        "e(b-v)": "0.0049"
    }
}

# directories
input_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/2025-09-22" # change to the name of the night
output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Photometry" # where we store the results

class PhotometryDataManager:

    def __init__(self):
        """
        Store parameters for I/O.
        """