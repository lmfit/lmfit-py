import numpy as np
from lmfit.old_models1d import StepModel

import matplotlib.pyplot as plt

x  = np.linspace(0, 10, 201)
dat = np.ones_like(x)
dat[:48] = 0.0
dat[48:77] = np.arange(77-48)/(77.0-48)
dat = dat +  5e-2*np.random.randn(len(x))
dat = 110.2 * dat + 12.0

mod = StepModel(background='constant', form='erf') # linear') # 'atan')

mod.guess_starting_values(dat, x)

init = mod.model(x=x)+mod.calc_background(x)
mod.fit(dat, x=x)

print mod.fit_report()

fit = mod.model(x=x)+mod.calc_background(x)

plt.plot(x, dat)
plt.plot(x, init, 'r+')
plt.plot(x, fit)
plt.show()

