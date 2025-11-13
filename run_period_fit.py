import pandas as pd
print(f"Script is running")
import os
print(os.getcwd())
import period_fitting_sinusoid
print(dir(period_fitting_sinusoid))
print(period_fitting_sinusoid.__file__)
from period_fitting_sinusoid import Sinusoid_Period_Finder
from matplotlib import pyplot as plt

filename = r"C:\Users\jp\OneDrive\Documents\1 Edinburgh University\Year 4\Telescope Group Project\Cepheids Data 3.csv"
df = pd.read_csv(filename)

time_list = df["Time"].dropna().astype(str).str.strip().tolist()


finder = Sinusoid_Period_Finder(
    name="Test",
    time=time_list,
    magnitude=df["Magnitude"].values,
    magnitude_error=df["Magnitude Error"].values,
)


finder.light_curve()

best_period, best_params = finder.fit_sinusoid()
print(f"Test 2")
print(f"\nBest-fit period: {best_period:.5f} days")
print("Best-fit parameters:", best_params)

print("About to plot light curve")
finder.light_curve()
print("Light curve plotted")

print("About to plot sinusoid fit")
finder.plot_sinusoid_fit()
print("Sinusoid fit plotted")

print(f"Test 3")

finder.sine_run_mcmc()
finder.sine_parameter_time_series()
finder.sine_plot_corner()
best_period = finder.sine_plot_emcee_fit()
print(f"Best period is {best_period}")

