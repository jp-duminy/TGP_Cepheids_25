# Cepheids Telescope Group Project 2025-26

Welcome to the repository; this is for the University of Edinburgh senior honours (4th year) cepheids telescope group project (2025-26). This was a five-person project, officially titled 'The Cepheid Period-Luminosity Relation and the Distance to the Andromeda Galaxy'. The project ran from September 2025 to February 2026.

## Introduction

The goal of the project was to derive a fundamental distance measurement to the Andromeda Galaxy. This was done by observing eleven classical cepheid variable stars in the Milky Way along with a twelfth classical cepheid in the Andromeda Galaxy. Cepheid variable stars pulsate, and their period of pulsation is related to their intrinsic luminosity, known as the period-luminosity (P-L) relation. If we know a cepheid's intrinsic luminosity (absolute magnitude) and measure the apparent magnitude, the distance modulus equation tells us how far away it is. Cepheids are thus crucial distance indicators in astronomy. 

From the eleven classical cepheid variable stars, which have literature parallax measurements (Biller-Jones et al. 2021), we can calibrate a P-L relation. Each rung on the cosmic distance ladder calibrates the next, in this case the parallax measurements allow us to generate a P-L relation as we know what each cepheid's absolute magnitude is already.

## Observations & Data Collection

We conducted eight observation nights with the remotely-operated PIRATE telescope at the Teide Observatory in Tenerife; our observations were reinforced by extra exposures queued by other telescope users on other nights. 

Standard astronomical reduction was applied to our images, taken in the V-filter. Due to time and availability constraints, there was insufficient time to conduct our own V and B-filter observations of the Andromeda cepheid. Thus, images from the Liverpool Telescope in the SDSS g-band were provided.

We then performed differential photometry to account for night-to-night offsets in the cepheid brightness. This was done by also measuring the magnitudes of five bright non-variable reference stars for each cepheid across nights. This allows us to normalise our light curves, meaning we are measuring the actual change in brightness of a cepheid by accounting for systematic variations.

## Data Analysis & Model Fitting

Cepheid light curves are usually modelled with a sawtooth, though a sinusoid can serve as a first-order approximation. Cepheids were modelled by first conducting a chi-squares grid search over a fixed literature period range for both sawtooth and sinusoid models. This provided both an initial guess for the model parameters and period, as well as allowing the trimming of any outlying data points. This was then fed into emcee, the MCMC modelling package. It is easiest to understand how this works by examining the code (I explain it in the docstrings). 

We store the MCMC chains of each galactic cepheid and resample from them in a hierarchical ln-likelihood model to create our P-L relation. This fully propagates the uncertainties from each cepheid into the final fit and marginalises over their individual parameters. Again, it is probably easiest to look at my docstrings and code comments to understand this.

## Andromeda Data Analysis

Andromeda required a filter transformation and a different approach to data management, but otherwise followed the same procedure. M31 CV1 was fitted and then its chain was sampled with the P-L chain to compute its distance modulus measurement. 

This represented a measurement of the distance to the Andromeda galaxy. 

## Code

All code should work if you fancy running it. I moved everything into folders and changed some import logic to make the repository easier to navigate (and hopefully more understandable): if I forgot to do this anywhere, it should be a straightforward fix.

Enjoy,

- JP