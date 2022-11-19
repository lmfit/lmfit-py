import math
from optparse import OptionParser

from numpy.testing import assert_allclose

from lmfit import Parameters, minimize

from .NISTModels import Models, ReadNistData


def ndig(a, b):
    """Precision for NIST values."""
    return round(-math.log10((abs(abs(a)-abs(b)) + 1.e-15) / abs(b)))


ABAR = ' |----------------+----------------+------------------+-------------------|'


def Compare_NIST_Results(DataSet, myfit, params, NISTdata):
    buff = [' ======================================',
            f' {DataSet}: ',
            ' | Parameter Name |  Value Found   |  Certified Value | # Matching Digits |']
    buff.append(ABAR)

    val_dig_min = 200
    err_dig_min = 200
    fmt = ' | %s | % -.7e | % -.7e   | %2i                |'
    for i in range(NISTdata['nparams']):
        parname = f'b{i+1}'
        par = params[parname]
        thisval = par.value
        certval = NISTdata['cert_values'][i]
        vdig = ndig(thisval, certval)
        pname = (parname + ' value ' + ' '*14)[:14]
        buff.append(fmt % (pname, thisval, certval, vdig))
        val_dig_min = min(val_dig_min, vdig)

        thiserr = par.stderr
        certerr = NISTdata['cert_stderr'][i]
        if thiserr is not None and myfit.errorbars:
            edig = ndig(thiserr, certerr)
            ename = (parname + ' stderr' + ' '*14)[:14]
            buff.append(fmt % (ename, thiserr, certerr, edig))
            err_dig_min = min(err_dig_min, edig)

    buff.append(ABAR)
    sumsq = NISTdata['sum_squares']
    try:
        chi2 = myfit.chisqr
        buff.append(' | Sum of Squares | %.7e  | %.7e    |  %2i               |'
                    % (chi2, sumsq, ndig(chi2, sumsq)))
    except Exception:
        pass
    buff.append(ABAR)
    if not myfit.errorbars:
        buff.append(' |          * * * * COULD NOT ESTIMATE UNCERTAINTIES * * * *              |')
        err_dig_min = 0
    if err_dig_min < 199:
        buff.append(f' Worst agreement: {val_dig_min} digits for value, '
                    f'{err_dig_min} digits for error ')
    else:
        buff.append(f' Worst agreement: {val_dig_min} digits')
    return val_dig_min, '\n'.join(buff)


def NIST_Dataset(DataSet, method='leastsq', start='start2',
                 plot=False, verbose=False):

    NISTdata = ReadNistData(DataSet)
    resid, npar, dimx = Models[DataSet]
    y = NISTdata['y']
    x = NISTdata['x']

    params = Parameters()
    for i in range(npar):
        pname = f'b{i+1}'
        pval1 = NISTdata[start][i]
        params.add(pname, value=pval1)
    try:
        myfit = minimize(resid, params, method=method, args=(x,), kws={'y': y}, nan_policy='raise')
    except ValueError:
        if verbose:
            print("Fit failed... nans?")
        return False
    digs, buff = Compare_NIST_Results(DataSet, myfit, myfit.params, NISTdata)
    if verbose:
        print(buff)

    return digs > 2


def build_usage():
    modelnames = []
    ms = ''
    for d in sorted(Models.keys()):
        ms = ms + f' {d} '
        if len(ms) > 55:
            modelnames.append(ms)
            ms = '    '
    modelnames.append(ms)
    modelnames = '\n'.join(modelnames)

    usage = f"""
 === Test Fit to NIST StRD Models ===

usage:
------
    python fit_NIST.py [options] Model Start

where Start is one of 'start1','start2' or 'cert', for different
starting values, and Model is one of

    {modelnames}

if Model = 'all', all models and starting values will be run.

options:
--------
  -m  name of fitting method.  One of:
          leastsq, nelder, powell, lbfgsb, bfgs,
          tnc, cobyla, slsqp, cg, newton-cg
      leastsq (Levenberg-Marquardt) is the default
"""
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
                    failures.append(f"   {dset} (starting at '{start}')")
        print('--------------------------------------')
        print(f' Fit Method: {opts.method} ')
        print(f' Final Results: {tpass} pass, {tfail} fail.')
        print(' Tests Failed for:\n %s' % '\n '.join(failures))
        print('--------------------------------------')
    elif dset not in Models:
        print(usage)
    else:
        return NIST_Dataset(dset, method=opts.method,
                            start=start, plot=False, verbose=True)


def RunNIST_Model(model):

    dset = ReadNistData(model)
    func, npar, dimx = Models[model]
    rss = (func(dset['cert_values'], x=dset['x'], y=dset['y'])**2).sum()
    tiny_rss = 1.e-16
    print(rss, dset['sum_squares'], tiny_rss)

    if dset['sum_squares'] < tiny_rss:
        assert rss < tiny_rss
    else:
        assert_allclose(rss, dset['sum_squares'])

    out1 = NIST_Dataset(model, start='start1', plot=False, verbose=False)
    out2 = NIST_Dataset(model, start='start2', plot=False, verbose=False)
    assert (out1 or out2)


def test_Bennett5():
    RunNIST_Model('Bennett5')


def test_BoxBOD():
    RunNIST_Model('BoxBOD')


def test_Chwirut1():
    RunNIST_Model('Chwirut1')


def test_Chwirut2():
    RunNIST_Model('Chwirut2')


def test_DanWood():
    RunNIST_Model('DanWood')


def test_ENSO():
    RunNIST_Model('ENSO')


def test_Eckerle4():
    RunNIST_Model('Eckerle4')


def test_Gauss1():
    RunNIST_Model('Gauss1')


def test_Gauss2():
    RunNIST_Model('Gauss2')


def test_Gauss3():
    RunNIST_Model('Gauss3')


def test_Hahn1():
    RunNIST_Model('Hahn1')


def test_Kirby2():
    RunNIST_Model('Kirby2')


def test_Lanczos1():
    RunNIST_Model('Lanczos1')


def test_Lanczos2():
    RunNIST_Model('Lanczos2')


def test_Lanczos3():
    RunNIST_Model('Lanczos3')


def test_MGH09():
    RunNIST_Model('MGH09')


def test_MGH10():
    RunNIST_Model('MGH10')


def test_MGH17():
    RunNIST_Model('MGH17')


def test_Misra1a():
    RunNIST_Model('Misra1a')


def test_Misra1b():
    RunNIST_Model('Misra1b')


def test_Misra1c():
    RunNIST_Model('Misra1c')


def test_Misra1d():
    RunNIST_Model('Misra1d')


def test_Nelson():
    RunNIST_Model('Nelson')


def test_Rat42():
    RunNIST_Model('Rat42')


def test_Rat43():
    RunNIST_Model('Rat43')


def test_Roszman1():
    RunNIST_Model('Roszman1')


def test_Thurber():
    RunNIST_Model('Thurber')


if __name__ == '__main__':
    run_interactive()
