import math
from optparse import OptionParser

from lmfit import Parameters, minimize
from NISTModels import Models, ReadNistData

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False


def ndig(a, b):
    """Precision for NIST values."""
    return round(-math.log10((abs(abs(a)-abs(b)) + 1.e-15) / abs(b)))


def Compare_NIST_Results(DataSet, myfit, NISTdata):
    print(' ======================================')
    print(' %s: ' % DataSet)
    print(' | Parameter Name |  Value Found   |  Certified Value | # Matching Digits |')
    print(' |----------------+----------------+------------------+-------------------|')
    params = myfit.params
    val_dig_min = 200
    err_dig_min = 200
    for i in range(NISTdata['nparams']):
        parname = 'b%i' % (i+1)
        par = params[parname]
        thisval = par.value
        certval = NISTdata['cert_values'][i]
        vdig = ndig(thisval, certval)
        pname = (parname + ' value ' + ' '*14)[:14]
        print(' | %s | % -.7e | % -.7e   | %2i                |' % (pname, thisval, certval, vdig))
        val_dig_min = min(val_dig_min, vdig)

        thiserr = par.stderr
        certerr = NISTdata['cert_stderr'][i]
        if thiserr is not None and myfit.errorbars:
            edig = ndig(thiserr, certerr)
            ename = (parname + ' stderr' + ' '*14)[:14]
            print(' | %s | % -.7e | % -.7e   | %2i                |' % (ename, thiserr, certerr, edig))
            err_dig_min = min(err_dig_min, edig)

    print(' |----------------+----------------+------------------+-------------------|')
    sumsq = NISTdata['sum_squares']
    try:
        chi2 = myfit.chisqr
        print(' | Sum of Squares | %.7e  | %.7e    |  %2i               |' % (chi2, sumsq,
                                                                              ndig(chi2, sumsq)))
    except:
        pass
    print(' |----------------+----------------+------------------+-------------------|')
    if not myfit.errorbars:
        print(' |          * * * * COULD NOT ESTIMATE UNCERTAINTIES * * * *              |')
        err_dig_min = 0
    if err_dig_min < 199:
        print(' Worst agreement: %i digits for value, %i digits for error ' % (val_dig_min, err_dig_min))
    else:
        print(' Worst agreement: %i digits' % (val_dig_min))
    return val_dig_min


def NIST_Test(DataSet, method='leastsq', start='start2', plot=True):

    NISTdata = ReadNistData(DataSet)
    resid, npar, dimx = Models[DataSet]
    y = NISTdata['y']
    x = NISTdata['x']

    params = Parameters()
    for i in range(npar):
        pname = 'b%i' % (i+1)
        cval = NISTdata['cert_values'][i]
        cerr = NISTdata['cert_stderr'][i]
        pval1 = NISTdata[start][i]
        params.add(pname, value=pval1)

    myfit = minimize(resid, params, method=method, args=(x,), kws={'y': y})
    digs = Compare_NIST_Results(DataSet, myfit, NISTdata)

    if plot and HASPYLAB:
        fit = -resid(myfit.params, x)
        plt.plot(x, y, 'ro')
        plt.plot(x, fit, 'k+-')
        plt.show()

    return digs > 2


modelnames = []
ms = ''
for d in sorted(Models.keys()):
    ms = ms + ' %s ' % d
    if len(ms) > 55:
        modelnames.append(ms)
        ms = '    '
modelnames.append(ms)
modelnames = '\n'.join(modelnames)

usage = """
 === Test Fit to NIST StRD Models ===

usage:
------
    python fit_NIST_lmfit.py [options] Model Start

where Start is one of 'start1','start2' or 'cert', for different
starting values, and Model is one of

    %s

if Model = 'all', all models and starting values will be run.

options:
--------
  -m  name of fitting method.  One of:
          leastsq, nelder, powell, lbfgsb, bfgs,
          tnc, cobyla, slsqp, cg, newto-cg
      leastsq (Levenberg-Marquardt) is the default
""" % modelnames

############################
parser = OptionParser(usage=usage, prog="fit-NIST.py")

parser.add_option("-m", "--method", dest="method", metavar='METH',
                  default='leastsq', help="set method name, default = 'leastsq'")

(opts, args) = parser.parse_args()
dset = ''
start = 'start2'
if len(args) > 0:
    dset = args[0]
if len(args) > 1:
    start = args[1]

if dset.lower() == 'all':
    tpass = 0
    tfail = 0
    failures = []
    dsets = sorted(Models.keys())
    for dset in dsets:
        for start in ('start1', 'start2', 'cert'):
            if NIST_Test(dset, method=opts.method, start=start, plot=False):
                tpass += 1
            else:
                tfail += 1
                failures.append("   %s (starting at '%s')" % (dset, start))

    print('--------------------------------------')
    print(' Fit Method: %s ' % opts.method)
    print(' Final Results: %i pass, %i fail.' % (tpass, tfail))
    print(' Tests Failed for:\n %s' % '\n '.join(failures))
    print('--------------------------------------')
elif dset not in Models:
    print(usage)
else:
    NIST_Test(dset, method=opts.method, start=start, plot=True)
