#!/usr/bin/env python
"""This runs the same fits for the NIST StRD models but using
plain old scipy.optimize and relying on no code from lmfit.
In fact, it goes right down to using
  scipy.optimize._minpack._lmdif()

The tests only check best-fit value, not estimated uncertainties.
Currently, not all tests pass.


"""

from __future__ import print_function
import sys
import math
import numpy as np

from scipy.optimize import _minpack

from NISTModels import Models, ReadNistData

try:
    import matplotlib
    matplotlib.use('WXAgg')
    import pylab
    HASPYLAB = True
except IOError:
    HASPYLAB = False


def ndig(a, b):
    return int(0.5-math.log10(abs(abs(a)-abs(b))/abs(b)))

def Compare_NIST_Results(DataSet, vals, NISTdata):
    #print(' ======================================')
    print(' %s: ' % DataSet)
    print(' | Parameter Name |  Value Found   |  Certified Value | # Matching Digits |')
    print(' |----------------+----------------+------------------+-------------------|')

    val_dig_min = 1000
    err_dig_min = 1000
    for i in range(NISTdata['nparams']):
        parname = 'b%i' % (i+1)
        thisval = vals[i]
        certval = NISTdata['cert_values'][i]

        vdig   = ndig(thisval, certval)

        pname = (parname + ' value ' + ' '*14)[:14]
        print(' | %s | % -.7e | % -.7e   | %2i                |' % (pname, thisval, certval, vdig))
        val_dig_min = min(val_dig_min, vdig)

    print(' |----------------+----------------+------------------+-------------------|')
    print(' Worst agreement: %i digits for value ' % (val_dig_min))
    return val_dig_min

def NIST_Test(DataSet, start='start2', plot=True):

    NISTdata = ReadNistData(DataSet)
    resid, npar, dimx = Models[DataSet]
    y = NISTdata['y']
    x = NISTdata['x']

    vals = []
    for i in range(npar):
        pname = 'b%i' % (i+1)
        cval  = NISTdata['cert_values'][i]
        cerr  = NISTdata['cert_stderr'][i]
        pval1 = NISTdata[start][i]
        vals.append(pval1)

    maxfev = 2500 * (npar + 1)
    factor  = 100
    xtol = 1.e-14
    ftol = 1.e-14
    epsfcn  = 1.e-13
    gtol = 1.e-14
    diag =  None
    print(" Fit with: ", factor, xtol, ftol, gtol, epsfcn, diag)

    _best, out, ier = _minpack._lmdif(resid, vals, (x, y), 1,
                                      ftol, xtol, gtol,
                                      maxfev, epsfcn, factor, diag)

    digs = Compare_NIST_Results(DataSet, _best, NISTdata)

    if plot and HASPYLAB:
        fit = -resid(_best, x, )
        pylab.plot(x, y, 'ro')
        pylab.plot(x, fit, 'k+-')
        pylab.show()

    return digs > 2

msg1 = """
----- NIST StRD Models -----
Select one of the Models listed below:
and a starting point of 'start1' or 'start2'
"""

msg2 = """
That is, use
    python fit_NIST.py Bennett5 start1
or go through all models and starting points with:
    python fit_NIST.py all
"""

if __name__  == '__main__':
    dset = 'Bennett5'
    start = 'start2'
    if len(sys.argv) < 2:
        print(msg1)
        out = ''
        for d in sorted(Models.keys()):
            out = out + ' %s ' % d
            if len(out) > 55:
                print( out)
                out = ''
        print(out)
        print(msg2)

        sys.exit()

    if len(sys.argv) > 1:
        dset = sys.argv[1]
    if len(sys.argv) > 2:
        start = sys.argv[2]
    if dset.lower() == 'all':
        tpass = 0
        tfail = 0
        failures = []
        dsets = sorted(Models.keys())
        for dset in dsets:
            for start in ('start1', 'start2'):
                if NIST_Test(dset, start=start, plot=False):
                    tpass += 1
                else:
                    tfail += 1
                    failures.append("   %s (starting at '%s')" % (dset, start))

        print('--------------------------------------')
        print(' Final Results: %i pass, %i fail.' % (tpass, tfail))
        print(' Tests Failed for:\n %s' % '\n '.join(failures))
        print('--------------------------------------')
    else:
        NIST_Test(dset, start=start, plot=True)

