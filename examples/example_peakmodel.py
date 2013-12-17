"""
Example using the built-in Peak-like models
"""
import numpy as np
from lmfit.models1d import GaussianModel, LorentzianModel, VoigtModel
import matplotlib.pyplot as plt

x  = np.linspace(0, 10, 101)

sca = 1./(2.0*np.sqrt(2*np.pi))
noise =  5e-2*np.random.randn(len(x))
dat = 2.60 -0.04*x + 7.5 * np.exp(-(x-4.0)**2 / (2*0.35)**2) + noise

mod = GaussianModel(background='linear')
# mod = VoigtModel(background='linear')
# mod = LorentzianModel(background='linear')

mod.guess_starting_values(dat, x)


plt.plot(x, dat)

# initial guess
plt.plot(x, mod.model(x=x) + mod.calc_background(x), 'r+')

mod.fit(dat, x=x)

print mod.fit_report()

# best fit
plt.plot(x, mod.model(x=x) + mod.calc_background(x))
plt.show()

