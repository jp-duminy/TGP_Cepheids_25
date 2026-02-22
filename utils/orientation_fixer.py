"""
author: @jp

TGP Cepheids 25-26

Sometimes an image is clearly flipped but the FLIPSTAT does not reflect this.

This script fixes this, simply input the night + filename and the orientation that is correct.

Make notes of flipped cepheid + night

"""

from astropy.io import fits

dir_path = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids"
file = "2025-10-23/cepheid_05_stacked.fits"

filepath = f"{dir_path}/{file}"
with fits.open(filepath, mode="update") as hdul:
    hdul[0].header["FLIPSTAT"] = "East" # make sure this is the opposite of the header
    hdul.flush()

print(f"Successfully changed header.")

# CP Cep 2025-09-22
# CP Cep 2025-10-01
# MW Cyg 2025-10-07
# CP Cep 2025-10-07
# V Lac 2025-10-08
# SW Cas 2025-10-08
# CP Cep 2025-10-13
# CP Cep 2025-10-22
# CP Cep 2025-10-23