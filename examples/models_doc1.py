from numpy import loadtxt
from lmfit import fit_report
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt

data = loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]

model = GaussianModel(['x'])
pars  = model.params()

pars['amplitude'].value = y.max()
pars['amplitude'].min   = 0
pars['center'].value = x.mean()
pars['sigma'].value  = (x.max()-x.min())/5.0
pars['sigma'].min    = 0.

result = model.fit(y, x=x, params=pars)

print fit_report(result.params, min_correl=0.25)

plt.plot(x, y,         'bo')
plt.plot(x, result.init_fit, 'r-')
plt.plot(x, result.best_fit, 'k--')
plt.show()
