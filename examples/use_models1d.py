import numpy as np
from lmfit.old_models1d import LinearModel, QuadraticModel, ExponentialModel
from lmfit.old_models1d import  LorenztianModel, GaussianModel, VoigtModel
import matplotlib.pyplot as plt


x  = np.linspace(0, 10, 101)
# dat = 118.0 + 10.0*np.exp(-x/7.0) + 5e-2*np.random.randn(len(x))
# dat = 18.0 + 1.5*x  + 5.6*np.random.randn(len(x))

sig = 0.47
amp = 12.00
cen = 5.66
eps = 0.15
off = 9
slo = 0.0012
sca = 1./(2.0*np.sqrt(2*np.pi))/sig

noise =  eps*np.random.randn(len(x))

dat = off +slo*x + amp*sca* np.exp(-(x-cen)**2 / (2*sig)**2) + noise

# mod = ExponentialModel(background='linear')
# mod = LinearModel()

mod = GaussianModel(background='quad')
mod = VoigtModel(background='quad')
mod = LorenztianModel(background='quad')
mod.guess_starting_values(dat, x, negative=False)
mod.params['bkg_offset'].value=min(dat)

init = mod.model(x=x)+mod.calc_background(x)
mod.fit(dat, x=x)


print mod.fit_report()

fit = mod.model(x=x)+mod.calc_background(x)

plt.plot(x, dat)
plt.plot(x, init)
plt.plot(x, fit)
plt.show()

