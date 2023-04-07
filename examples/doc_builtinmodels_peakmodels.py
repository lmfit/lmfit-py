# <examples/doc_builtinmodels_peakmodels.py>
import matplotlib.pyplot as plt
from numpy import loadtxt

from lmfit.models import GaussianModel, LorentzianModel, VoigtModel

data = loadtxt('test_peak.dat')
x = data[:, 0]
y = data[:, 1]


# Gaussian model
mod = GaussianModel()
pars = mod.guess(y, x=x)
out = mod.fit(y, pars, x=x)

print(out.fit_report(correl_mode='table'))

plt.plot(x, y)
plt.plot(x, out.best_fit, '-', label='Gaussian Model')
plt.legend()
plt.show()


# Lorentzian model
mod = LorentzianModel()
pars = mod.guess(y, x=x)
out = mod.fit(y, pars, x=x)

print(out.fit_report(correl_mode='table'))

plt.figure()
plt.plot(x, y, '-')
plt.plot(x, out.best_fit, '-', label='Lorentzian Model')
plt.legend()
plt.show()


# Voigt model
mod = VoigtModel()
pars = mod.guess(y, x=x)
out = mod.fit(y, pars, x=x)

print(out.fit_report(correl_mode='table'))

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

axes[0].plot(x, y, '-')
axes[0].plot(x, out.best_fit, '-', label='Voigt Model\ngamma constrained')
axes[0].legend()

# allow the gamma parameter to vary in the fit
pars['gamma'].vary = True
out_gamma = mod.fit(y, pars, x=x)
print(out.fit_report(correl_mode='table'))

axes[1].plot(x, y, '-')
axes[1].plot(x, out_gamma.best_fit, '-', label='Voigt Model\ngamma unconstrained')
axes[1].legend()

plt.show()
# <end examples/doc_builtinmodels_peakmodels.py>
