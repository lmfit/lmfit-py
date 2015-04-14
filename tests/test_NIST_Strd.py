from __future__ import print_function
import sys
import math
from optparse import OptionParser

from lmfit import Parameters, minimize

from NISTModels import Models, ReadNistData

HASPYLAB = False
for arg in sys.argv:
    if 'nose' in arg:
        HASPYLAB = False

if HASPYLAB:
    try:
        import matplotlib
        import pylab
        HASPYLAB = True
    except ImportError:
        HASPYLAB = False

def ndig(a, b):
    "precision for NIST values"
    return round(-math.log10((abs(abs(a)-abs(b)) +1.e-15)/ abs(b)))

ABAR = ' |----------------+----------------+------------------+-------------------|'
def Compare_NIST_Results(DataSet, myfit, params, NISTdata):
    buff = [' ======================================',
 ' %s: ' % DataSet,
 ' | Parameter Name |  Value Found   |  Certified Value | # Matching Digits |']
    buff.append(ABAR)

    val_dig_min = 200
    err_dig_min = 200
    fmt = ' | %s | % -.7e | % -.7e   | %2i                |'
    for i in range(NISTdata['nparams']):
        parname = 'b%i' % (i+1)
        par = params[parname]
        thisval = par.value
        certval = NISTdata['cert_values'][i]
        vdig    = ndig(thisval, certval)
        pname   = (parname + ' value ' + ' '*14)[:14]
        buff.append(fmt % (pname, thisval, certval, vdig))
        val_dig_min = min(val_dig_min, vdig)

        thiserr = par.stderr
        certerr = NISTdata['cert_stderr'][i]
        if thiserr is not None and myfit.errorbars:
            edig   = ndig(thiserr, certerr)
            ename = (parname + ' stderr' + ' '*14)[:14]
            buff.append(fmt % (ename, thiserr, certerr, edig))
            err_dig_min = min(err_dig_min, edig)

    buff.append(ABAR)
    sumsq = NISTdata['sum_squares']
    try:
        chi2 = myfit.chisqr
        buff.append(' | Sum of Squares | %.7e  | %.7e    |  %2i               |'
                    % (chi2, sumsq, ndig(chi2, sumsq)))
    except:
        pass
    buff.append(ABAR)
    if not myfit.errorbars:
        buff.append(' |          * * * * COULD NOT ESTIMATE UNCERTAINTIES * * * *              |')
        err_dig_min = 0
    if err_dig_min < 199:
        buff.append(' Worst agreement: %i digits for value, %i digits for error '
                    % (val_dig_min, err_dig_min))
    else:
        buff.append(' Worst agreement: %i digits' % (val_dig_min))
    return val_dig_min, '\n'.join(buff)

def NIST_Dataset(DataSet, method='leastsq', start='start2',
                 plot=True, verbose=False):

    NISTdata = ReadNistData(DataSet)
    resid, npar, dimx = Models[DataSet]
    y = NISTdata['y']
    x = NISTdata['x']

    params = Parameters()
    for i in range(npar):
        pname = 'b%i' % (i+1)
        cval  = NISTdata['cert_values'][i]
        cerr  = NISTdata['cert_stderr'][i]
        pval1 = NISTdata[start][i]
        params.add(pname, value=pval1)

    myfit = minimize(resid, params, method=method, args=(x,), kws={'y':y})
    digs, buff = Compare_NIST_Results(DataSet, myfit, myfit.params, NISTdata)
    if verbose:
        print(buff)
    if plot and HASPYLAB:
        fit = -resid(myfit.params, x, )
        pylab.plot(x, y, 'ro')
        pylab.plot(x, fit, 'k+-')
        pylab.show()

    return digs > 1

def build_usage():
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
    python fit_NIST.py [options] Model Start

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
    return usage

############################
def run_interactive():
    usage = build_usage()
    parser = OptionParser(usage=usage, prog="fit-NIST.py")

    parser.add_option("-m", "--method", dest="method",
                      metavar='METH',
                      default='leastsq',
                      help="set method name, default = 'leastsq'")

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
                if NIST_Dataset(dset, method=opts.method, start=start,
                                plot=False, verbose=True):
                    tpass += 1
                else:
                    tfail += 1
                    failures.append("   %s (starting at '%s')" % (dset, start))
        print('--------------------------------------')
        print(' Fit Method: %s ' %  opts.method)
        print(' Final Results: %i pass, %i fail.' % (tpass, tfail))
        print(' Tests Failed for:\n %s' % '\n '.join(failures))
        print('--------------------------------------')
    elif dset not in Models:
        print(usage)
    else:
        return NIST_Dataset(dset, method=opts.method,
                            start=start, plot=True, verbose=True)

def RunNIST_Model(model):
    out1 = NIST_Dataset(model, start='start1', plot=False, verbose=False)
    out2 = NIST_Dataset(model, start='start2', plot=False, verbose=False)
    print("NIST Test" , model, out1, out2)
    assert(out1 or out2)
    return out1 or out2

def test_Bennett5():
    return RunNIST_Model('Bennett5')

def test_BoxBOD():
    return RunNIST_Model('BoxBOD')

def test_Chwirut1():
    return RunNIST_Model('Chwirut1')

def test_Chwirut2():
    return RunNIST_Model('Chwirut2')

def test_DanWood():
    return RunNIST_Model('DanWood')

def test_ENSO():
    return RunNIST_Model('ENSO')

def test_Eckerle4():
    return RunNIST_Model('Eckerle4')

def test_Gauss1():
    return RunNIST_Model('Gauss1')

def test_Gauss2():
    return RunNIST_Model('Gauss2')

def test_Gauss3():
    return RunNIST_Model('Gauss3')

def test_Hahn1():
    return RunNIST_Model('Hahn1')

def test_Kirby2():
    return RunNIST_Model('Kirby2')

def test_Lanczos1():
    return RunNIST_Model('Lanczos1')

def test_Lanczos2():
    return RunNIST_Model('Lanczos2')

def test_Lanczos3():
    return RunNIST_Model('Lanczos3')

def test_MGH09():
    return RunNIST_Model('MGH09')

def test_MGH10():
    return RunNIST_Model('MGH10')

def test_MGH17():
    return RunNIST_Model('MGH17')

def test_Misra1a():
    return RunNIST_Model('Misra1a')

def test_Misra1b():
    return RunNIST_Model('Misra1b')

def test_Misra1c():
    return RunNIST_Model('Misra1c')

def test_Misra1d():
    return RunNIST_Model('Misra1d')

def test_Nelson():
    return RunNIST_Model('Nelson')

def test_Rat42():
    return RunNIST_Model('Rat42')

def test_Rat43():
    return RunNIST_Model('Rat43')

def test_Roszman1():
    return RunNIST_Model('Roszman1')

def test_Thurber():
    return RunNIST_Model('Thurber')

if __name__ == '__main__':
    run_interactive()
