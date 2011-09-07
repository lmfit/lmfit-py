import sys
try:
    import matplotlib
    matplotlib.use('WXAgg')
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

from lmfit import Parameters, Minimizer
from testutils import report_errors

from NISTModels import Models, ReadNistData

NISTData = ReadNistData('Gauss2')
resid, npar, dimx = Models['Gauss2']

p_cert = Parameters()
params = Parameters()
for i in range(npar):
    pname = 'b%i' % (i+1)
    cval  = NISTData['cert_values'][i]
    cerr  = NISTData['cert_stderr'][i]
    pval1 = NISTData['start1'][i]
    p_cert.add(pname, value=cval)
    p_cert.stderr = cerr
    params.add(pname, value=pval1)
    print pname, cval, cerr,  pval1
y = NISTData['y']
x = NISTData['x']

if HASPYLAB:
    pylab.plot(x, y, 'r+')


myfit = Minimizer(resid, params,
                  fcn_args=(x,), fcn_kws={'y':y},
                  scale_covar=True)

myfit.prepare_fit()
init = resid(params, x)

if HASPYLAB:
    pylab.plot(x, init, 'b--')

myfit.leastsq()
print ' Nfev = ', myfit.nfev
print myfit.chisqr, myfit.redchi, myfit.nfree
report_errors(params, modelpars=p_cert, show_correl=False)

fit = -resid(params, x, )

if HASPYLAB:
    pylab.plot(x, fit, 'k-')

pylab.show()
#




