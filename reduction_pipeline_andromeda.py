import numpy as np
from astropy.io import fits
from pathlib import Path
from collections import defaultdict

def extract_lt_date(filename):
    """
    Extract observing-night date from LT filename:
    h_e_YYYYMMDD_XXX_*.fits
    This works by picking the third (0,1,2) block that is seperated by _ which is the date
    """
    return filename.split("_")[2]

def stack_lt_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_by_date = defaultdict(list)

    # Group by observing night (filename date)
    for f in sorted(input_dir.glob("h_e_*.fits")):
        date = extract_lt_date(f.name)
        files_by_date[date].append(f)

    # Stack per night
    for date, files in files_by_date.items():
        print(f"\nObserving night {date}: {len(files)} files")

        images = []
        headers = []

        for f in files:
            with fits.open(f) as hdul:
                images.append(hdul[0].data.astype(float))
                headers.append(hdul[0].header)

        stack = np.mean(np.stack(images, axis=0), axis=0)

        # Build stacked header
        base = headers[0].copy()

        base["NSTACK"]  = 3
        base["TOTEXP"]  = sum(h["EXPTIME"] for h in headers)
        base["MEANEXP"] = base["TOTEXP"] / 3
        base["COMMENT"] = f"Mean stack of 3 LT images from night {date}"
        base["OBSDATE"] = date

         # Mean gain (if available)
        gains = [h.get("GAIN", 0) for h in headers if "GAIN" in h]
        if gains:
            base["MEANGAIN"] = float(np.mean(gains))

        # Correct read noise for mean stack
        read_noises = [h.get('RDNOIS', 0) for h in headers]
        if read_noises:
            base['TOTRN'] = float(np.sqrt(np.sum(np.array(read_noises)**2)))


        outname = f"h_e_{date}_stacked.fits"
        fits.PrimaryHDU(stack, base).writeto(
            output_dir / outname, overwrite=True
        )

        print(f"Saved {outname}")
        
def main():
    input_dir = "/storage/teaching/TelescopeGroupProject/Andromeda_LT_Data"
    output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/andromeda_LT"

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    stack_lt_images(input_dir, output_dir)

    # Summary of stacked files
    output_path = Path(output_dir)
    stacked_files = sorted(output_path.glob("h_e_*_stacked.fits"))
    print(f"\nStacking complete. {len(stacked_files)} stacked files created:")
    for f in stacked_files:
        print(f"  {f.name}")

if __name__ == "__main__":
    main()