"""
Fit with Data in a pandas DataFrame
===================================

Simple example demonstrating how to read in the data using pandas and supply
the elements of the DataFrame from lmfit.

"""
import matplotlib.pyplot as plt
import pandas as pd

from lmfit.models import LorentzianModel

###############################################################################
# read the data into a pandas DataFrame, and use the 'x' and 'y' columns:
dframe = pd.read_csv('peak.csv')

model = LorentzianModel()
params = model.guess(dframe['y'], x=dframe['x'])

result = model.fit(dframe['y'], params, x=dframe['x'])

###############################################################################
# and gives the plot and fitting results below:
result.plot_fit()
plt.show()

print(result.fit_report())
