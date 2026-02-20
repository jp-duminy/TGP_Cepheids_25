<<<<<<< HEAD
# Todo for Reduction

Sort out fits file headers
Do standard stars
Look through other observing nights data
If any major issue are found try find why they are a problem or write them down to ask Kenneth
Add gain and read noise headers if not already there
Need to sort out the work flow, 
Test for git

Update 15/01

File Structure:
Cepheids: /storage/teaching/TelescopeGroupProject/2025-26/Cepheids/ -> night dates
Calibration Frames: /storage/teaching/TelescopeGroupProject/2026-26/Calibrations/week#/binning1x1 -> files

Binning is always the same.
~~Code from reduction_pipeline_skeleton needs to be reviewed and ported into reduction_pipeline.~~ 17/01
~~JP: CalibrationManager~~  16/01
~~Don: CepheidDataOrganiser~~ 16/01

## Update 17/01
Pipeline works but requires refinement and extra features need to be added.
Runtime: ~5-10mins
A directory with reduced images now exists for each night. All cepheids have at l;east 8 raw data points.

**Argumentative Cepheids**
Cepheid 2 (wind shear?)
Cepheid 3 (wind shear?)
Cepheid 5 (flipping issues)

**Targets**
- Add standard star pathing (should be fairly straightforward)
- ~~Adjust cepheid analysis to only take V-filter images~~ Mimi and Amy 22/01
- Double check whether the pipeline follows proper protocol (is correction applied in correct order etc.)
- Incorporate sigma clipping to remove cosmic rays
- Find a method to deal with wind shear?
- Potentially incorporate more .fits headers?



=======
## Update 27/01

### Small Tasks:

- ~~run_period_fit must be updated to compare the two models and pick the best one (implement Bayesian Information Criterion)~~ 30/01
- ~~Store chain~~ 31/01
- Need to write functions to parse data from photometry
- ~~Update plots to incorporate errors (can extract from chains!)~~ 31/01
- Start thinking about applying the data analysis to Andromeda CV1

### Big Ones:

- ~~BIC is useful but also compare to literature & eyeball the fit; allow flexibility (reduced chi-squares)~~ 30/01
- ~~Once best model is picked, run emcee with desired model~~ 30/01
- Store chains (as dictionaries) and check whether they converged
- ~~Sample randomly from cepheid chains; period and magnitude (midline)~~ 31/01
- ~~Marginalised likelihood model for P-L relation (use literature values)~~ 31/01
- ~~Run emcee on P-L in a similar way to cepheid~~ 31/01
>>>>>>> Period-Fitting-Beta
