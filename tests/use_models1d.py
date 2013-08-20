import numpy as np
from lmfit.models1d import LinearModel, QuadraticModel, ExponentialModel, GaussianModel
import matplotlib.pyplot as plt


x  = np.linspace(0, 10, 101)
# dat = 118.0 + 10.0*np.exp(-x/7.0) + 5e-2*np.random.randn(len(x))
# dat = 18.0 + 1.5*x  + 5.6*np.random.randn(len(x))

sca = 1./(2.0*np.sqrt(2*np.pi))
noise =  5e-2*np.random.randn(len(x))
dat = 2.60 -0.04*x + 7.5 * np.exp(-(x-4.0)**2 / (2*0.35)**2) + noise

# mod = ExponentialModel(background='linear')
mod = GaussianModel(background='linear')
# mod = LinearModel()
mod.guess_starting_values(dat, x)
mod.params['bkg_offset'].value=2.0

init = mod.model(x=x)+mod.calc_background(x)
mod.fit(dat, x=x)

print mod.fit_report()

fit = mod.model(x=x)+mod.calc_background(x)

plt.plot(x, dat)
plt.plot(x, init)
plt.plot(x, fit)
plt.show()

