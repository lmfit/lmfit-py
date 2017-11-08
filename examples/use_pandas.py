from lmfit.models import LorentzianModel
import matplotlib.pyplot as plt
import pandas as pd

dframe = pd.read_csv('peak.csv')

model = LorentzianModel()
params = model.guess(dframe['y'], x=dframe['x'])

result = model.fit(dframe['y'], params, x=dframe['x'])

print(result.fit_report())
result.plot_fit()
plt.show()
