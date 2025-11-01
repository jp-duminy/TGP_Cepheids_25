# -*- coding: utf-8 -*-
"""
Period fitting Cepheids TGP 25/26
@author: jp
"""

import numpy as np
import scipy
import emcee as mc
from matplotlib import pyplot as plt
import astropy
from cepheid_analysis import Cepheid_Chi_Error_Analysis


class Cepheid_Period_Finder:
    
    def __init__(self, name, time, magnitude, snr):
        """
        Initialise cepheid parameters relevant to period fitting.
        """
        self.name = name
        self.time = [np.array(t) for t in time] # time in days
        self.magnitude = [np.array(m) for m in magnitude]
        self.snr = [np.array(s) for s in snr]
        
    def julian_date_converter(self):
        """
        Converts ISO times to MJD for ease of use.
        """
        t = astropy.time.Time(self.time, scale='utc', format='iso')
        self.julian_date = t.mjd # modified julian date, simply for the plot
        
    def light_curve(self):
        """
        Plot the light curve of each cepheid variable.
        """
        
        ncols = 3
        nrows = (len(self.time) + ncols - 1) // ncols
        
        fig, ax = plt.subplots(nrows, ncols)
        
        for i in range(len(self.time)):
            ax[i].plot(self.julian_date[i], self.magnitude[i])
            ax[i].set_xlabel('Modified Julian Date')
            ax[i].set_ylabel('Corrected Magnitude')
            ax[i].set_title(self.name[i])
            ax[i].invert_yaxis()
            ax[i].grid(True)
            
        for j in range(len(self.time), len(ax)):
            ax[j].axis('off')
            
        ax.set_title('Light Curve for Cepheids')
        plt.show()
        
        
    # unfinished code
    def lomb_scargle_2(self, julian_date, magnitude):
        """
        Employ Lomb-Scargle periodogram to locate period and then optimise this value.
        Needs error update.
        """
        frequency, power = astropy.timeseries.LombScargle(julian_date, magnitude).autopower()
        best_freq = frequency[np.argmax(power)]
        
        
    def lomb_scargle_period(self):
        """
        Employ Lomb-Scargle periodogram to locate period and then optimise this value. 
        Need update for errors.
        """
        ncols = 3
        nrows = (len(self.time) + ncols - 1) // ncols
        fig, ax = plt.subplots(nrows, ncols)
        
        self.best_frequencies = []
        
        for i in range(len(self.time)):
            frequency, power = astropy.timeseries.LombScargle(self.julian_date[i], self.magnitude[i]).autopower()
            best_freq = frequency[np.argmax(power)]
            self.best_frequencies.append(best_freq)
            
            ax[i].plot(frequency, power)
            ax[i].set_xlabel('Frequency')
            ax[i].set_ylabel('Lomb-Scargle Power')
            ax[i].set_title(self.name[i])
            ax[i].invert_yaxis()
            ax[i].grid(True)
            
        for j in range(len(self.time), len(ax)):
            ax[j].axis('off')
        
        ax.set_title('Power Spectrum from Lomb-Scargle Periodogram')
        plt.show()
        
        
        
