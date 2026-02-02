import numpy as np
import astropy.io.fits as fits
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from Cepheid_apertures import AperturePhotometry

class Reduction_To_Photometry:
    """
    A class which will take in the files from reduction_pipeline and pass them on to
    Cepheid_apertures in the correct format
    """
    def __init__(self, file_dir):
        """
        Initialising the class with where to find the reduced files and
        the pattern to use to sort the Cepheid files
        """
        self.file_dir = Path(file_dir)
        self.file_pattern = re.compile(r'cepheid_(\d+)_.*\.fits$', re.IGNORECASE)
        self.dir_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})') # 4 digit year, 2 digit month, 2 digit day

    def date_dirs(self):
        # where are the date files stored
        base = self.file_dir
        date_dirs = [d for d in base.iterdir() if d.is_dir() and self.dir_pattern.match(d.name)]
        return date_dirs
    
    def sort_files(self, directory: Path):
        """
        sort the files in the directory created by the reduction pipeline
        """
        sorted_files = defaultdict(list)

        #for file in Path(file_dir).glob("*.fits"):
        for file in directory.glob("*.fits"):
            pattern_match = self.file_pattern.search(file.name)
            if pattern_match:
                ceph_num = int(pattern_match.group(1))
                sorted_files[ceph_num].append(file)
        return dict(sorted_files)

    def do_photometry(self, file_path):
        """
        Going to call the Cepheid_aperture class and do photometry
        on each file from the reduction pipeline
        """
        do_photometry = AperturePhotometry(file_path)
        return do_photometry
    
    def save_photometry(self, photometry, source_file: Path, date: str, output_base: str):
        """
        How to save the photometry results
        """
        ceph_id = self.get_cepheid_id(source_file)
        if ceph_id is None:
            raise ValueError(f"Cannot determine Cepheid ID from {source_file.name}")

        out_dir = Path(output_base) / date / f"cepheid_{ceph_id:02d}"
        # if directory doesn't exist, make it
        out_dir.mkdir(parents=True, exist_ok=True)

        # naming the output files, adding the _phot to make it easier to identify
        out_name = f"{date}_cepheid_{ceph_id:02d}_{source_file.stem}_phot.fits"
        out_path = out_dir / out_name
        photometry.writeto(out_path, overwrite=True)
        return out_path
    
    def get_cepheid_id(self, file_path: Path):
        m = self.file_pattern.search(file_path.name)
        return int(m.group(1)) if m else None
    
    def photometry_to_row(self, ap_obj, source_file: Path, date: str, gain, read_noise, stack_size):
        """
        Run aperture photometry and return results as a dict
        suitable for a pandas DataFrame row.
        """
        # identify cepheid
        ceph_name = source_file.stem

        # get observation time from header (ISOT string)
        time_isot = ap_obj.header.get("DATE-OBS")

        # photometry
        centroid, fwhm = ap_obj.get_centroid_and_fwhm()
        ap_rad = 2.0 * fwhm

        target_flux, sky_per_pix, ap_area, ann_area = ap_obj.aperture_photometry(
            centroid=centroid,
            ap_rad=ap_rad,
            ceph_name=ceph_name,
            date=date,
            plot=False,
            subpixels=1
        ) #should this be 5

        # instrumental magnitude
        mag = ap_obj.instrumental_magnitude(target_flux)

        # magnitude error
        mag_err = ap_obj.get_inst_mag_error(
            target_counts=target_flux,
            aperture_area=ap_area,
            sky_counts=sky_per_pix,
            sky_ann_area=ann_area,
            gain=gain,
            exp_time=ap_obj.header["EXPTIME"],
            read_noise=read_noise,
            stack_size=stack_size,
        )

        return {
            "name": ceph_name,
            "time": time_isot,          # keep ISOT; convert later
            "magnitude": mag,
            "magnitude_error": mag_err,
        }


if __name__ == "__main__":

    ceph_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_test"
    output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_Photometry"

    ceph_coords = {"Ceph_num": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
                   "RA": ["20 12 22.83", "20 54 57.53", "20 57 20.83", "21 04 16.63", "21 57 52.69", "22 40 52.15", "22 48 38.00", "23 07 10.08", "00 26 19.45", "00 29 58.59", "01 32 43.22"],
                   "Dec": ["+32 52 17.8", "+47 32 01.7", "+40 10 39.1", "+39 58 20.1", "+56 09 50.0", "+56 49 46.1", "+56 19 17.5", "+58 33 15.1", "+51 16 49.3", "+60 12 43.1", "+63 35 37.7"]}
    
    ceph_skycoords = {}

    for num, ra_str, dec_str in zip(
        ceph_coords["Ceph_num"],
        ceph_coords["RA"],
        ceph_coords["Dec"]
    ):
        coord = SkyCoord(
            ra_str,
            dec_str,
            unit=(u.hourangle, u.deg),
            frame="icrs"
        )
        ceph_skycoords[num] = coord

    for num, coord in ceph_skycoords.items():
        print(num, coord.ra.deg, coord.dec.deg)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialise reducer
    reducer = Reduction_To_Photometry(ceph_dir)

    all_rows = []

    # Main loop: dates to files to photometry
    for date_dir in reducer.date_dirs():

        date = date_dir.name
        date_output_dir = Path(output_dir) / date
        date_output_dir.mkdir(parents=True, exist_ok=True)

        night_rows = []

        for file in date_dir.glob("*.fits"):

            try:
                # Run aperture photometry object
                ap_obj = reducer.do_photometry(file)
                header = ap_obj.header
                data = ap_obj.data
                wcs = ap_obj.wcs

                ny, nx = data.shape

                # Get metadata from stacked FITS headers
                exptime = header.get("EXPTIME", header.get("MEANEXP", 1.0))
                gain = header.get("GAIN", header.get("MEANGAIN", 1.0))
                read_noise = header.get("RDNOISE", header.get("TOTRN", 5.0))
                stack_size = header.get("NSTACK", 1)
                time_isot = header.get("DATE-OBS")
                if time_isot is None:
                    raise KeyError("DATE-OBS not found in FITS header")
                """
                # Photometry
                # Before calling get_centroid_and_fwhm()
                """
                # not sure about this?
                data = ap_obj.data
                ny, nx = data.shape
                # Find which Cepheids are on this image (WCS)
                for ceph_num, coord in ceph_skycoords.items():

                    #x_guess, y_guess = wcs.world_to_pixel_values(coord.ra.deg, coord.dec.deg)
                    x_guess, y_guess = wcs.world_to_pixel_values(coord)
                    print(f"Checking Cepheid {ceph_num} at pixel ({x_guess:.1f}, {y_guess:.1f})")

                    if not (0 <= x_guess < nx and 0 <= y_guess < ny):
                        continue  # Cepheid not on this CCD

                    # Cutout around expected position
                    x_guess = float(x_guess)
                    y_guess = float(y_guess)
                    
                    x0 = int(round(x_guess))
                    y0 = int(round(y_guess))

                    half_size = 25  # pixels
                    x1 = max(0, x0 - half_size)
                    x2 = min(nx, x0 + half_size)
                    y1 = max(0, y0 - half_size)
                    y2 = min(ny, y0 + half_size)

                    cutout = data[y1:y2, x1:x2]

                    ny_c, nx_c = cutout.shape
                    if ny_c < 7 or nx_c < 7:
                        continue

                    # Force odd dimensions
                    if ny_c % 2 == 0:
                        cutout = cutout[:-1, :]
                        y2 -= 1

                    if nx_c % 2 == 0:
                        cutout = cutout[:, :-1]
                        x2 -= 1

                    if cutout.size == 0:
                        continue

                    # Centroid + FWHM
                    centroid_local, fwhm = ap_obj.get_centroid_and_fwhm(cutout)

                    centroid = (
                        centroid_local[0] + x1,
                        centroid_local[1] + y1,
                    )

                    ap_rad = 2.0 * fwhm

                    # Aperture photometry
                    target_flux, sky_per_pix, ap_area, ann_area = ap_obj.aperture_photometry(
                        centroid=centroid,
                        ap_rad=ap_rad,
                        ceph_name=f"cepheid_{ceph_num}",
                        date=date,
                        plot=False,
                    )

                    magnitude = ap_obj.instrumental_magnitude(target_flux)

                    magnitude_error = ap_obj.get_inst_mag_error(
                        target_counts=target_flux,
                        aperture_area=ap_area,
                        sky_counts=sky_per_pix,
                        sky_ann_area=ann_area,
                        gain=gain,
                        exp_time=exptime,
                        read_noise=read_noise,
                        stack_size=stack_size,
                    )

                    row = {
                        "name": f"cepheid_{ceph_num}",
                        "time": time_isot,
                        "magnitude": magnitude,
                        "magnitude_error": magnitude_error,
                    }

                    night_rows.append(row)
                    all_rows.append(row)

            except Exception as e:
                print(f"Skipping {file.name}: {e}")

        # Save per-night CSV
        if night_rows:
            df_night = pd.DataFrame(night_rows)
            df_night = df_night.sort_values(by=["name", "time"]).reset_index(drop=True)
            night_csv = date_output_dir / f"{date}_photometry.csv"
            df_night.to_csv(night_csv, index=False)
            print(f"Saved per-night CSV: {night_csv}")

    # Save combined CSV
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all = df_all.sort_values(by=["name", "time"]).reset_index(drop=True)
        combined_csv = Path(output_dir) / "all_nights_photometry.csv"
        df_all.to_csv(combined_csv, index=False)
        print(f"Saved combined CSV: {combined_csv}")
        """

                centroid, fwhm = ap_obj.get_centroid_and_fwhm(ap_obj.data, plot=True)
                ap_rad = 2.0 * fwhm


                target_flux, sky_per_pix, ap_area, ann_area = ap_obj.aperture_photometry(
                    centroid=centroid,
                    ap_rad=ap_rad,
                    ceph_name=file.stem,
                    date=date,
                    plot=False,
                )

                magnitude = ap_obj.instrumental_magnitude(target_flux)

                magnitude_error = ap_obj.get_inst_mag_error(
                    target_counts=target_flux,
                    aperture_area=ap_area,
                    sky_counts=sky_per_pix,
                    sky_ann_area=ann_area,
                    gain=gain,
                    exp_time=exptime,
                    read_noise=read_noise,
                    stack_size=stack_size,
                )

                # Store row
                row = {
                    "name": file.stem,
                    "time": time_isot,
                    "magnitude": magnitude,
                    "magnitude_error": magnitude_error,
                }

                night_rows.append(row)
                all_rows.append(row)

            except Exception as e:
                print(f"Skipping {file.name}: {e}")

        # Save per-night CSV
        if night_rows:
            df_night = pd.DataFrame(night_rows)
            df_night = df_night.sort_values(by=["name", "time"]).reset_index(drop=True)
            night_csv_path = date_output_dir / f"{date}_photometry.csv"
            df_night.to_csv(night_csv_path, index=False)
            print(f"Saved per-night CSV: {night_csv_path}")       

    # Save combined CSV for all nights
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all = df_all.sort_values(by=["name", "time"]).reset_index(drop=True)
        combined_csv_path = Path(output_dir) / "all_nights_photometry.csv"
        df_all.to_csv(combined_csv_path, index=False)
        print(f"Saved combined CSV: {combined_csv_path}")
        """