"""

Sometimes an image is clearly flipped but the FLIPSTAT does not reflect this.

This script fixes this, simply input the night + filename and the orientation that is correct.

Make notes of flipped cepheid + night

"""

from astropy.io import fits

dir_path = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids"
file = "2025-10-01/cepheid_05_stacked.fits"

filepath = f"{dir_path}/{file}"
with fits.open(filepath, mode="update") as hdul:
    hdul[0].header["FLIPSTAT"] = "East"
    hdul.flush()

print(f"Successfully changed header.")

# CP Cep 2025-09-22
# CP Cep 2025-10-01