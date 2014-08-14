#!/usr/bin/env python
#<examples/models_doc1.py>
from numpy import loadtxt
from lmfit import fit_report
from lmfit.models import GaussianModel, VoigtModel
import matplotlib.pyplot as plt


data = loadtxt('test_peak.dat')
x = data[:, 0]
y = data[:, 1]

gmodel = GaussianModel()
gmodel.guess_starting_values(y, x=x)
gresult = gmodel.fit(y, x=x)

print 'With Gaussian: '
print fit_report(gresult.params, min_correl=0.25)
print 'Chi-square = %.3f, Reduced Chi-square = %.3f' % (gresult.chisqr, gresult.redchi)
plt.plot(x, y,         'ko')
plt.plot(x, gresult.best_fit, 'r-')


vmodel = VoigtModel()
vmodel.guess_starting_values(y, x=x)
vresult = vmodel.fit(y, x=x)

print 'With Voigt: '
print fit_report(vresult.params, min_correl=0.25)
print 'Chi-square = %.3f, Reduced Chi-square = %.3f' % (vresult.chisqr, vresult.redchi)

plt.plot(x, vresult.best_fit, 'b-')


vmodel.params['gamma'].vary = True
vmodel.params['gamma'].expr = None

vresult2 = vmodel.fit(y, x=x)

print 'With Voigt, varying gamma: '
print fit_report(vresult2.params, min_correl=0.25)
print 'Chi-square = %.3f, Reduced Chi-square = %.3f' % (vresult2.chisqr, vresult2.redchi)
plt.plot(x, vresult.best_fit, 'g-')

plt.show()

#<end examples/models_doc1.py>
