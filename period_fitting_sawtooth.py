"""
Sawtooth period fitting Cepheids TGP 25/26
@author: jp
"""

# under construction !

import numpy as np
import scipy
import emcee as mc
from matplotlib import pyplot as plt
from period_fitting_sinusoid import Sinusoid_Period_Finder
import corner

class Sawtooth_Period_Finder(Sinusoid_Period_Finder):

    def __init__(self, name, time, magnitude, magnitude_error):
        super().__init__(name, time, magnitude, magnitude_error)

    def sawtooth_model(self, t, amplitude, phase, offset, width, frequency):
        """
        Sawtooth fitting function.
        """
        return amplitude * scipy.signal.sawtooth(2 * np.pi * frequency * (t - phase), width) + offset
    
    def saw_chi_sq(self, theta, period):
        """
        Chi-square function taking period as an argument so period may be iterated over.
        """
        a, p, o, w = theta
        f = 1 / period
        M_model = self.sawtooth_model(self.time, a, p, o, w, f)

        chisq = np.sum(((self.magnitude - M_model) / self.magnitude_error)**2)

        return chisq

    def fit_sawtooth(self):
        """
        Iterates over the literature range of classical cepheid periods, fixing periods whilst fitting
        other free parameters. Returns chisqu plot for period range and best-fit parameters.
        """
        p_min = 1.49107 # days, Breger (1980)
        p_max = 78.14 # days, Soszyński et al. (2024)
        period_range = np.linspace(p_min, p_max, 100)

        chisqu_vals = []
        param_vals = []
        param_uncertainties = []

        for period in period_range:
            # define a rough initial guess
            a0 = (np.max(self.magnitude) - np.min(self.magnitude)) / 2 
            p0 = 0.0
            o0 = np.median(self.magnitude)
            theta = [a0, p0, o0]
            frequency = 1 / period
                
            parameters, cov = scipy.optimize.curve_fit(lambda t, a, p, o: self.sinusoid_model(t, a, p, o, frequency), 
                                                   self.time, self.magnitude, theta, 
                                                   sigma=self.magnitude_error, absolute_sigma=True)
            
            param_vals.append(parameters)
            uncertainties = np.sqrt(np.diag(cov))
            param_uncertainties.append(uncertainties)

            chisqu = self.chi_sq(parameters, period)
            chisqu_vals.append(chisqu)

        fig, ax = plt.subplots()
        ax.plot(period_range, chisqu_vals)
        ax.set_xlabel('Period [days]')
        ax.set_ylabel(f"$\chi^2$")
        ax.set_title(f"$\chi^2$ value for period range")
        plt.show()

        best_period_index = np.argmin(chisqu_vals)
        best_chisqu_period = period_range[best_period_index]
        best_chisqu_params = param_vals[best_period_index]
        best_chisqu_uncertainties = param_uncertainties[best_period_index]
        best_chisqu = chisqu_vals[best_period_index]

        print(f"Best period: {best_chisqu_period} \n with $\chi^2$ value of {best_chisqu}")

        print(f"Best-fit parameters:")
        for i in range(len(best_chisqu_params)):
            print(f"{best_chisqu_params[i]} \u00B1 {best_chisqu_uncertainties[i]}")

        self.a0, self.p0, self.o0 = best_chisqu_params
        self.f0 = 1 / best_chisqu_period

        return best_chisqu_period, best_chisqu_params
    
    def plot_sawtooth_fit(self):
        """
        Overplot the modelled data from curve_fit onto the data.
        """
        fig, ax = plt.subplots()
        y = self.sinusoid_model(self.time, self.a0, self.p0, self.o0, self.f0)
        ax.errorbar(self.time, self.magnitude, yerr=self.magnitude_error, fmt='o', label='Data')
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.set_title(self.name)
        ax.plot(self.time, y, label='Modelled Data')
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude
        ax.set_title(f"Sinusoidal $\chi^2$ Fit for Cepheid {self.name}")

        plt.show()

    
    def saw_ln_likelihood(self, theta):
        """
        Log-likelihood function that takes a parameter vector as its input and returns the log-likelihood.
        The likelihood function is the probability of the dataset given the parameters.
        """
        a, f, p, o, w = theta
        modelled_magnitude = self.sawtooth_model(self.time, a, f, p, o, w)
        residuals = self.magnitude - modelled_magnitude
        constant = np.log(2 * np.pi * self.magnitude_error**2)

        return -0.5 * np.sum((residuals / self.magnitude_error)**2 + constant)

    def saw_ln_prior(self, theta):
        """
        Generate priors for the posterior distribution that will be sampled from.
        """
        a_min, a_max = -2 * abs(self.a0), 2 * abs(self.a0)
        f_min, f_max = 0.5 * self.f0, 2 * self.f0 # frequency can't be negative
        p_min, p_max = self.p0 - 2*abs(self.p0), self.p0 + 2*abs(self.p0)
        o_min, o_max = self.o0 - 2*abs(self.o0), self.o0 + 2*abs(self.o0)
        w_min, w_max = 0.0, 1.0  # width must be in range 0-1

        a, f, p, o, w = theta
        
        if (a_min < a < a_max and
            f_min < f < f_max and
            p_min < p < p_max and
            o_min < o < o_max and
            w_min <= w <= w_max):
            return 0.0
        else:
            return -np.inf
        
    def saw_ln_prob(self, theta):
        """
        Full log-probability function combining the priors and the likelihood.
        """
        lp = self.saw_ln_prior(theta)
        if not np.isfinite(lp): # if prior is outside of expected range
            return -np.inf # declare it impossible
        return lp + self.saw_ln_likelihood(theta) # new probability in log space

    def saw_walker_initialisation(self, nwalkers, ndim):
        """
        Uses chi-squared result to determine best initial position for walkers.
        """
        w0 = 0.5
        pos = np.array([self.a0, self.f0, self.p0, self.o0, w0]) # stitch into parameter vector
        
        starting_position = pos + 1e-2 * np.random.randn(nwalkers, ndim) # add gaussian noise
        
        return starting_position
    
    def saw_run_mcmc(self):
        """
        Run the emcee Monte Carlo Markov Chain.
        """
        ndim = 5 # number of dimensions
        nwalkers = 100 # numbers of walkers to explore parameter space

        pos = self.saw_walker_initialisation(nwalkers, ndim)

        sampler = mc.EnsembleSampler(nwalkers, ndim, self.saw_ln_prob) # sample the distribution

        print(f"Running burn-in for cepheid...")
        pos = sampler.run_mcmc(pos, 100) # let walkers explore parameter space
        print(f"Burn-in complete.")
        sampler.reset() # reset sampler before main chain


        print(f"Running production...")
        sampler.run_mcmc(pos, 5000, progress=True) # run main chain

        self.sampler = sampler # keep sampler for analysis of quality of chain
        return sampler

    def saw_parameter_time_series(self):
        """
        Plots the parameter time series to analyse how long it takes walkers to explore
        parameter space.
        """
        ndim = 5
        fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        labels = ["Amplitude", "Frequency", "Phase", "Offset", "Width"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("Step Number")

    def saw_plot_corner(self, truths):
        """
        First diagnostic plot: corner plot to analyse quality of fit.
        """
        tau = self.sampler.get_autocorr_time()
        print(f"Autocorrelation times: {tau}")
        thin  = int(np.mean(tau) / 2)
        self.flat_samples = self.sampler.get_chain(thin=thin, flat=True)
        
        labels = ["Amplitude", "Frequency", "Phase", "Offset", "Width"]
        fig = corner.corner(
            self.flat_samples, labels=labels, truths=truths
            )
        plt.show()
    
    def plot_sawtooth_emcee_fit(self):
        """
        Plot the model returned by emcee.
        """
        a0, f0, p0, o0, w0 = np.mean(self.flat_samples)

        fig, ax = plt.subplots()
        y = self.sawtooth_model(self.time, a0, f0, p0, o0, w0)
        ax.errorbar(self.time, self.magnitude, yerr=self.magnitude_error, fmt='o', label='Data')
        ax.set_xlabel('Time [Days]')
        ax.set_ylabel('Corrected Magnitude')
        ax.set_title(self.name)
        ax.plot(self.time, y, label='Modelled Data')
        ax.legend()
        ax.invert_yaxis() # brighter -> lower magnitude
        ax.set_title(f"Emcee Fit for Cepheid {self.name}")

        plt.show()


