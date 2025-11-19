import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits
import glob
import os
import astropy.units as u
import astropy.wcs as WCS
from astropy.coordinates import SkyCoord

from astropy.visualization import ZScaleInterval

from trim import Trim
from test_dark_subtract import Bias
from test_flat_divide import Flat

def find_ceph(ceph_folder, ceph_nums):
    """
    Using glob to filter through a folder which contains the observations from one night
    this will Return all of the cepheid images taken (we only included cepheids if there
    was 5 images taken of them)
    """
    ceph_files = glob.glob(os.path.join(ceph_folder, "*.fits"))

    filtered_files = []
    for file in ceph_files:
        filename = os.path.basename(file)
        if any(f"Cepheids_{num}_" in filename for num in ceph_nums):
            filtered_files.append(file)

    return filtered_files

if __name__ == "__main__":
    
    #read in the bias frames NOTE: this is the 1x1 binning bias frames for week 1
    bias_frame1 = fits.open("/home/s2407710/Documents/tgp/bias_2025_09_20/PIRATE_161197_Bias11_0_2025_09_20_18_25_20.fits")[0].data
    bias_frame2 = fits.open("/home/s2407710/Documents/tgp/bias_2025_09_20/PIRATE_161208_Bias11_1_2025_09_20_18_36_04.fits")[0].data
    bias_frame3 = fits.open("/home/s2407710/Documents/tgp/bias_2025_09_20/PIRATE_161209_Bias11_2_2025_09_20_18_36_09.fits")[0].data
    bias_frame4 = fits.open("/home/s2407710/Documents/tgp/bias_2025_09_20/PIRATE_161210_Bias11_3_2025_09_20_18_36_15.fits")[0].data

    #read in the flat frames NOTE: this is the 1x1 binning flat frames for week 1
    flat_frame1 = fits.open("/home/s2407710/Documents/tgp/flat_V_2025_09_04/PIRATE_159244_flats_V_01_2025_09_04_06_28_52.fits")[0].data
    flat_frame2 = fits.open("/home/s2407710/Documents/tgp/flat_V_2025_09_04/PIRATE_159243_flats_V_02_2025_09_04_06_28_05.fits")[0].data
    flat_frame3 = fits.open("/home/s2407710/Documents/tgp/flat_V_2025_09_04/PIRATE_159242_flats_V_03_2025_09_04_06_27_20.fits")[0].data
    flat_frame4 = fits.open("/home/s2407710/Documents/tgp/flat_V_2025_09_04/PIRATE_159241_flats_V_04_2025_09_04_06_26_35.fits")[0].data

    #the path to the folder where the cepheid data is stored (in this case this is the path to week 1's data)
    ceph_folder_path = "/home/s2407710/Documents/tgp/22_09_cepheid1"

    #numbers of all the cepheids observed on night 1 (there were not enough observations for cepheid 1 on night 1)
    ceph_nums = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

    #a folder to store the folders of the different cepheids
    all_corrected = {}

    #filter through the different cepheids, this will allow a stacked cepheid image
    #for each cepheid to be made
    for num in ceph_nums:
        ceph_files = find_ceph(ceph_folder_path, [num])
        corrected_images = []

        total_exposure = 0
        total_gain = []
        total_read_noise = []

        with fits.open(ceph_files[0]) as hdul0:
            base_header = hdul0[0].header.copy() 

        #correct each image of a specific number, then stack the images and take the mean
        for ceph_image in ceph_files:
            with fits.open(ceph_image) as hdul:

            #opening the fits file and getting the data and header
                hdul = fits.open(ceph_image)
                ceph_data = hdul[0].data 
                header = hdul[0].header

            #getting the exposure time for each image of the cepheid, adding this to total_exposure
                exposure = header.get("EXPOSURE", 0)
                total_exposure += exposure

            #getting the gain for each cepheid image, adding to list total_gain
                gain = header.get("GAIN", 0)
                total_gain.append(gain)

            #getting the read noise for each cepheid image, adding to list total_read_noise
                read_noise = header.get("RDNOISE", 0)
                total_read_noise.append(read_noise)

            #ceph_data = fits.open(ceph_image)[0].data
            bias_correction = Bias(bias_frame1, bias_frame2, bias_frame3, bias_frame4, ceph_data)
            bias_image = bias_correction.subtraction()
            flat_correction = Flat(flat_frame1, flat_frame2, flat_frame3, flat_frame4, bias_image)
            correct_image = flat_correction.flat_divide()
            corrected_images.append(correct_image)

        stacked_cepheid = np.stack(corrected_images, axis=0)
        mean_cepheid = np.mean(stacked_cepheid, axis=0)
        all_corrected[num] = corrected_images

        base_header["TOTEXP"]  = total_exposure
        base_header["TOTGAIN"] = float(np.mean(total_gain))
        base_header["TOTRN"]   = float(np.sqrt(np.sum(np.array(total_read_noise)**2)))
        base_header["NSTACK"]  = len(ceph_files)
        base_header["COMMENT"] = f"Stacked {len(ceph_files)} images for Cepheid {num}"

    
        #this part saves the stacked cepheid images as a fits file to a folder
        out_dir = "/home/s2407710/Documents/tgp/stacked_cepheids_22_09"
        os.makedirs(out_dir, exist_ok=True)

        save_path = os.path.join(out_dir, f"cepheid_{num}_stacked.fits")
       # fits.PrimaryHDU(mean_cepheid).writeto(save_path, overwrite=True)

        hdu = fits.PrimaryHDU(mean_cepheid, header=base_header)
        hdu.writeto(save_path, overwrite=True)

        print(f"Saved stacked Cepheid {num} → {save_path}")
    """   
    #this plots the stacked images along the Z-scale
    zscale = ZScaleInterval()

    fig, axes = plt.subplots(1, len(all_corrected), figsize=(5 * len(all_corrected), 5))

    if len(all_corrected) == 1:
        axes = [axes]

    for ax, (ceph_num, images) in zip(axes, all_corrected.items()):
        mean_img = np.mean(images, axis=0)
        vmin, vmax = zscale.get_limits(mean_img)
        ax.imshow(mean_img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"Cepheid {ceph_num}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    """

    """
        hdu = fits.PrimaryHDU(mean_cepheid)
        # Add a header keyword for total exposure time, making the title for the total exposure time
        hdu.header["TOTEXP"] = total_exposure
        #taking average of the gain values, adding a header for this
        avg_gain = np.mean(total_gain)
        hdu.header["TOTGAIN"] = avg_gain
        #adding the read noises in quadrature, adding this to fits file
        read_noise_vals = np.array(total_read_noise)
        read_noise_add = np.sqrt(np.sum(read_noise_vals**2))
        hdu.header["TOTRN"] = read_noise_add
        """