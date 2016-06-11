#!/usr/bin/env python
#<examples/doc_with_itercb.py>

from numpy import linspace, random
import matplotlib.pylab as pylab

from lmfit.models import LinearModel, GaussianModel
from lmfit.lineshapes import gaussian

def per_iteration(pars, iter, resid, *args, **kws):
    print(" ITER ", iter, ["%.5f" % p for p in pars.values()])


x = linspace(0., 20, 401)
y = gaussian(x, amplitude=24.56, center=7.6543, sigma=1.23)
y = y  - .20*x + 3.333 + random.normal(scale=0.23,  size=len(x))

mod = GaussianModel(prefix='peak_') + LinearModel(prefix='bkg_')

pars = mod.make_params()
pars['peak_amplitude'].value = 3.0
pars['peak_center'].value = 6.0
pars['peak_sigma'].value = 2.0
pars['bkg_intercept'].value = 0.0
pars['bkg_slope'].value = 0.0


out = mod.fit(y, pars, x=x, iter_cb=per_iteration)

pylab.plot(x, y, 'b--')

print(' Nfev = ', out.nfev)
print( out.fit_report())

pylab.plot(x, out.best_fit, 'k-')
pylab.show()

#<end examples/doc_with_itercb.py>
