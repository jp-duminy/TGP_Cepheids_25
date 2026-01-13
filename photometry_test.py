from Cepheid_apertures import Aperture_Photometry

def main():
    Test = Aperture_Photometry(filename="c:/Users/drcla/OneDrive/Telescope Group Project/TGP_Cepheids_25/test_data_for_phot.fits")
    data = Test.mask_data_and_plot(50, 50, 60, plot = False)
    centroid, fwhm = Test.get_centroid_and_fwhm(data)
    print(centroid)
    print(fwhm)
    flux, sky_per_pix, ap_size, ann_size = Test.aperture_photometry(data, centroid, ap_rad = 6, plot = False, ceph_name = "woo", date = "hoo")
    #Test.curve_of_growth(data, savefig = True)
    mag = Test.instrumental_magnitude(flux)
    print(mag)
    dmag = 1.086 / Test.get_snr(flux, sky_per_pix, ap_size, ann_size, gain = 1, exp_time = 1,
                                read_noise = 0, stack_size = 1)
    print(dmag)



    

main()

"""Notes: Test data is 101x101 array. 
get_centroid_and_fwhm returns parameters in pixel coordinates
of masked data, where origin is shifted
Don't make masked data array so small, it cuts off the apertures,
but probs not issue with real data"""