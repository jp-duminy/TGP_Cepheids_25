"""
Full differential photometry pipeline for Andromeda CV1 across all LT stacked images.
Uses WCS to locate CV1 and reference stars, calibrates against Pan-STARRS g-band.
@author: jp
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt

from Cepheid_apertures import AperturePhotometry, DustExtinction
from photometry_per_night import SinglePhotometry

import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False

# ========================
# DIRECTORIES
# ========================
stacked_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Andromeda"
output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Andromeda/Photometry"

# ========================
# CATALOGUES
# ========================
cv1_coords = {
    "ra": 10.36375,
    "dec": 41.16956,
    "e(b-v)": 0.06,
}

# Pan-STARRS g-band reference stars
andromeda_reference_catalogue = {
    "ref1": {
        "ra": 10.325845706611698,
        "dec": 41.19885332428335,
        "g_true": 13.6372
    },
    "ref2": {
        "ra": 10.398563205499425,
        "dec": 41.1278041684018,
        "g_true": 15.0174
    },
    "ref3": {
        "ra": 10.414045181371485,
        "dec": 41.226639052836326,
        "g_true": 14.5635
    },
    "ref4": {
        "ra": 10.315504008634187,
        "dec": 41.119531147713005,
        "g_true": 16.9177
    },
    "ref5": {
        "ra": 10.423218417492418,
        "dec": 41.1789170971879,
        "g_true": 14.1134
    }
}


def run_andromeda_photometry(plot=False):
    """
    Perform differential aperture photometry on CV1 across all LT stacked images.
    
    For each stacked image:
        1. Use WCS to convert RA/Dec -> pixel coordinates
        2. Run SinglePhotometry.raw_photometry on CV1 and 5 reference stars
        3. Compute zero-point offset from reference stars
        4. Apply calibration and dust correction
    
    Returns
    -------
    df : pd.DataFrame
        Columns: MJD, ISOT, g_calibrated, g_err, zp, n_refs
    """
    stacked_path = Path(stacked_dir)
    stacked_files = sorted(stacked_path.glob("h_e_*_stacked.fits"))

    if len(stacked_files) == 0:
        raise FileNotFoundError(f"No stacked files found in {stacked_dir}")

    print(f"Found {len(stacked_files)} stacked LT images\n")

    results = []

    for fits_path in stacked_files:
        print(f"\n{'='*60}")
        print(f"Processing: {fits_path.name}")
        print(f"{'='*60}")

        hdr = fits.getheader(str(fits_path))
        mjd = hdr['MJD']
        isot = hdr['DATE-OBS']
        wcs = WCS(hdr)

        # --- CV1 photometry ---
        cv1_x, cv1_y = wcs.all_world2pix(cv1_coords['ra'], cv1_coords['dec'], 0)

        try:
            cv1_phot = SinglePhotometry(
                fits_path=str(fits_path),
                x_rough=float(cv1_x),
                y_rough=float(cv1_y),
                name="CV1",
                ebv=str(cv1_coords['e(b-v)']),
            )
            m_cv1, m_cv1_err = cv1_phot.raw_photometry(width=150, plot=plot)
        except Exception as e:
            print(f"  CV1 FAILED: {e}")
            continue

        # --- Reference star photometry ---
        offsets = []
        for ref_id, ref_data in andromeda_reference_catalogue.items():
            try:
                ref_x, ref_y = wcs.all_world2pix(ref_data['ra'], ref_data['dec'], 0)

                ref_phot = SinglePhotometry(
                    fits_path=str(fits_path),
                    x_rough=float(ref_x),
                    y_rough=float(ref_y),
                    name=ref_id,
                    ebv="0.0",
                )
                m_ref, m_ref_err = ref_phot.raw_photometry(width=150, plot=False)
                offset = ref_data['g_true'] - m_ref
                offsets.append(offset)
                print(f"  {ref_id}: m_inst = {m_ref:.3f}, offset = {offset:.3f}")

            except Exception as e:
                print(f"  {ref_id} FAILED: {e}")
                continue

        if len(offsets) < 3:
            print(f"  SKIPPING: only {len(offsets)} refs succeeded (need >= 3)")
            continue

        # --- Calibrate via differential zero-point ---
        offsets = np.array(offsets)
        zp = np.median(offsets)
        zp_err = np.std(offsets) / np.sqrt(len(offsets))

        g_cal = m_cv1 + zp
        g_cal_err = np.sqrt(m_cv1_err**2 + zp_err**2)

        # --- Convert SDSS-g to Johnson V using Jester et al. (2005) ---
        # V = g - 0.58*(g-r) - 0.01   (RMS = 0.02 mag)
        # g-r derived from literature B-V via inverse Jester:
        #   B-V = 0.98*(g-r) + 0.22  =>  g-r = (B-V - 0.22) / 0.98
        # Literature mean B-V ~ 1.0 for a P=31.4d Cepheid (Massey et al.)
        BV_literature = 1.28 # https://www.aavso.org/index.php/aavso-alert-notice-422
        g_r = (BV_literature - 0.22) / 0.98  # ~ 0.796
        jester_offset = 0.58 * g_r + 0.01     # ~ 0.472
        jester_rms = 0.02  # mag, from Jester et al. (2005)

        V_cal = g_cal - jester_offset
        V_cal_err = np.sqrt(g_cal_err**2 + jester_rms**2)

        # --- Dust correction in V-band ---
        # A_V = 3.1 * E(B-V) (standard R_V = 3.1)
        A_V = 3.1 * cv1_coords['e(b-v)']
        V_corrected = V_cal - A_V

        results.append({
            'MJD': mjd,
            'ISOT': isot,
            'V_mag': V_corrected,
            'V_err': V_cal_err,
            'g_calibrated': g_cal,
            'g_err': g_cal_err,
            'zp': zp,
            'n_refs': len(offsets),
        })

        print(f"\n  >> V = {V_corrected:.3f} +/- {V_cal_err:.3f}  (g = {g_cal:.3f}, ZP = {zp:.3f}, {len(offsets)} refs)")

    # --- Compile results ---
    df = pd.DataFrame(results)
    df = df.sort_values('MJD').reset_index(drop=True)

    # --- Save ---
    """
    
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "andromeda_CV1_lightcurve.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nLight curve saved to: {csv_path}")
    print(f"Total epochs: {len(df)}")
    """
    out_path = Path(output_dir)
    period_fit_df = df[['ISOT', 'V_mag', 'V_err']].copy()
    period_fit_df.columns = ['ISOT', 'm_differential', 'm_differential_err']
    period_fit_df['Name'] = 'Andromeda_CV1'
    period_fit_path = out_path / "andromeda_CV1_period_fit.csv"
    period_fit_df.to_csv(period_fit_path, index=False)
    print(f"Period-fit-ready CSV saved to: {period_fit_path}")

    return df

if __name__ == "__main__":
    df = run_andromeda_photometry(plot=False)
    plot_light_curve(df)
    plot_zp_stability(df)