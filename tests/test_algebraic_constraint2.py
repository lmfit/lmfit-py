from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from lmfit import Parameters, Parameter, Minimizer
from lmfit.utilfuncs import gaussian, loren, pvoigt
from lmfit.printfuncs import report_fit
import sys


HASPYLAB = False
# Turn off plotting if run by nosetests.
if not sys.argv[0].endswith('nosetests'):
    try:
        import matplotlib
        import pylab
        HASPYLAB = True
    except ImportError:
        pass

# Turn off plotting if run by nosetests.
if sys.argv[0].endswith('nosetests'):
    HASPYLAB = False 

def test_constraints(with_plot=True):
    with_plot = with_plot and HASPYLAB

    def residual(pars, x, sigma=None, data=None):
        yg = gaussian(x, pars['amp_g'].value,
                   pars['cen_g'].value, pars['wid_g'].value)
        yl = loren(x, pars['amp_l'].value,
                   pars['cen_l'].value, pars['wid_l'].value)

        slope = pars['line_slope'].value
        offset = pars['line_off'].value
        model =  yg +  yl + offset + x * slope
        if data is None:
            return model
        if sigma is None:
            return (model - data)
        return (model - data)/sigma


    n = 201
    xmin = 0.
    xmax = 20.0
    x = linspace(xmin, xmax, n)

    data = (gaussian(x, 21, 8.1, 1.2) +
            loren(x, 10, 9.6, 2.4) +
            random.normal(scale=0.23,  size=n) +
            x*0.5)

    if with_plot:
        pylab.plot(x, data, 'r+')

    pfit = [Parameter(name='amp_g',  value=10),
            Parameter(name='cen_g',  value=9),
            Parameter(name='wid_g',  value=1),

            Parameter(name='amp_tot',  value=20),
            Parameter(name='amp_l',  expr='amp_tot - amp_g'),
            Parameter(name='cen_l',  expr='1.5+cen_g'),
            Parameter(name='wid_l',  expr='2*wid_g'),

            Parameter(name='line_slope', value=0.0),
            Parameter(name='line_off', value=0.0)]

    sigma = 0.021  # estimate of data error (for all data points)

    myfit = Minimizer(residual, pfit,
                      fcn_args=(x,), fcn_kws={'sigma':sigma, 'data':data},
                      scale_covar=True)

    myfit.prepare_fit()
    init = residual(myfit.params, x)

    myfit.leastsq()

    print(' Nfev = ', myfit.nfev)
    print( myfit.chisqr, myfit.redchi, myfit.nfree)

    report_fit(myfit.params, min_correl=0.3)

    fit = residual(myfit.params, x)
    if with_plot:
        pylab.plot(x, fit, 'b-')
    assert(myfit.params['cen_l'].value == 1.5 + myfit.params['cen_g'].value)
    assert(myfit.params['amp_l'].value == myfit.params['amp_tot'].value - myfit.params['amp_g'].value)
    assert(myfit.params['wid_l'].value == 2 * myfit.params['wid_g'].value)

    # now, change fit slightly and re-run
    myfit.params['wid_l'].expr = '1.25*wid_g'
    myfit.leastsq()
    report_fit(myfit.params, min_correl=0.4)
    fit2 = residual(myfit.params, x)
    if with_plot:
        pylab.plot(x, fit2, 'k')
        pylab.show()

    assert(myfit.params['cen_l'].value == 1.5 + myfit.params['cen_g'].value)
    assert(myfit.params['amp_l'].value == myfit.params['amp_tot'].value - myfit.params['amp_g'].value)
    assert(myfit.params['wid_l'].value == 1.25 * myfit.params['wid_g'].value)

test_constraints()
