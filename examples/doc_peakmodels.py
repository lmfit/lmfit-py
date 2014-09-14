#!/usr/bin/env python
#<examples/doc_peakmodels.py>
from numpy import loadtxt
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel
import matplotlib.pyplot as plt

data = loadtxt('test_peak.dat')
x = data[:, 0]
y = data[:, 1]

MODEL = 'gauss'
MODEL = 'loren'
MODEL = 'voigt'
# gamma_free = False
gamma_free = True

if MODEL.lower().startswith('g'):
    mod = GaussianModel()
elif MODEL.lower().startswith('l'):
    mod = LorentzianModel()
elif MODEL.lower().startswith('v'):
    mod = VoigtModel()

pars = mod.guess(y, x=x)

if gamma_free:
    pars['gamma'].set(value=0.7, vary=True, expr='')

out = mod.fit(y, pars, x=x)
print(out.fit_report(min_correl=0.25))

plt.plot(x, y,  'b-')
plt.plot(x, out.best_fit,  'r-')
plt.show()


#<end examples/doc_peakmodels.py>
