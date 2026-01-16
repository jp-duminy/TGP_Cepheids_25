import numpy as np
from astropy.io import fits
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re
import glob

class Calibration_Set:
    """
    Returns a master flat and bias for a given week+day configuration.
    """
    def __init__(self, name, bias_dir, flat_dir):
        self.name = name
        self.bias_dir = Path(bias_dir) if bias_dir else None
        self.flat_dir = Path(flat_dir) if flat_dir else None
        
        self.master_bias = None
        self.master_flat = None

    @staticmethod
    def img_trim(img):
        """
        Trims image by desired number of pixels.
        """
        trim = 50 # decided on 50 pixels
        return img[trim:-trim, trim:-trim]
    
    def img_stacker(self, imgs):
        """
        Stack images and return stack + median of stack.
        """
        stack = np.stack(imgs, axis=0)
        median = np.median(stack, axis=0)
        return stack, median
    
    def create_master_bias(self):
        """
        Create master bias image in a specified directory.
        """
        bias_tag = "*.fits"

        if self.master_bias is not None: # if this already exists
            return self.master_bias
        
        bias_files = sorted(self.bias_dir.glob(bias_tag))
        if len(bias_files) == 0:
            raise FileNotFoundError(f"No bias files in {self.bias_dir}") # diagnostic
        print(f"{len(bias_files)} found in d{self.bias_dir}.")

        bias_frames = [self.img_trim(fits.getdata(f)) for f in bias_files]
        _, self.master_bias = self.img_stacker(bias_frames)
        print(f"Master bias shape: {self.master_bias.shape}")

        return self.master_bias

    def create_master_flat(self):
        """
        Create master bias image in a specified directory.
        """
        flat_tag = "*.fits"

        if self.master_flat is not None: # if this already exists
            return self.master_flat
        
        flat_files = sorted(self.bias_dir.glob(flat_tag))
        if len(flat_files) == 0:
            raise FileNotFoundError(f"No flat files in {self.flat_dir}") # diagnostic
        print(f"{len(flat_files)} found in d{self.flat_dir}.")

        flat_frames = [self.img_trim(fits.getdata(f)) for f in flat_files]
        stack, normalisation = self.img_stacker(flat_frames)
        self.master_flat = stack / normalisation
        print(f"Master flat shape: {self.master_flat.shape}")

        return self.master_flat
        
    def prepare(self):
        """
        Prepare master bias and flat frames.
        """
        self.create_master_bias()
        self.create_master_flat()
        return self
    
class CepheidDataOrganiser:

    """Organises Cepheid files by number and night."""

    def __init__(self, cepheids_directory):
        """Create path to Cepheid directory and find patterns in files of "Cepheid_(#)"."""
        self.cepheids_directory = Path(cepheids_directory)
        self.cepheid_pattern = re.compile(r'Cepheids?_(\d+)', re.IGNORECASE) 
    
    def list_observation_nights(self):
        """Sort all directories for nights in the Cepheids directory
        in alphabetical order"""
        nights = sorted([night for night in self.cepheids_directory.iterdir() if night.is_dir()])
        return nights

    def organise_night(self, night_directory):
        """Organise each night's files based on Cepheid number"""
        cepheid_files = defaultdict(list)
        
        for file in sorted(Path(night_directory).glob("*.fits")):
            pattern_presence = self.cepheid_pattern.search(file.name)
            if pattern_presence:
                ceph_num = int(pattern_presence.group(1))
                cepheid_files[ceph_num].append(file)
        
        return dict(cepheid_files)
    
    def organise_all_nights(self):
        """Organise all Cepheid files firstly by night, and then by Cepheid number. Returns
        a list of files that correspond to each night+Cepheid."""
        all_data = {}
        nights = self.list_observation_nights()
        
        for night in nights:
            night_name = night.name
            cepheid_files = self.organise_night(night)
            if cepheid_files:
                all_data[night_name] = cepheid_files
        
        return all_data
    
    def filter_useful_images(file_list, min_sequence=5, method='last_n'):
        """Static method""" 
    
    




