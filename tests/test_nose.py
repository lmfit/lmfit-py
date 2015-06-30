# -*- coding: utf-8 -*-
from __future__ import print_function
from lmfit import minimize, Parameters, Parameter, report_fit, Minimizer
from lmfit.minimizer import SCALAR_METHODS
from lmfit.lineshapes import gaussian
import numpy as np
from numpy import pi
from numpy.testing import assert_
import unittest
import nose
from nose import SkipTest

def check(para, real_val, sig=3):
    err = abs(para.value - real_val)
    print( para.name, para.value, real_val, para.stderr)
    assert(err < sig * para.stderr)

def check_wo_stderr(para, real_val, sig=0.1):
    err = abs(para.value - real_val)
    print (para.name, para.value, real_val)
    assert(err < sig)

def check_paras(para_fit, para_real):
    for i in para_fit:
        check(para_fit[i], para_real[i].value)

def test_simple():
    # create data to be fitted
    np.random.seed(1)
    x = np.linspace(0, 15, 301)
    data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
            np.random.normal(size=len(x), scale=0.2) )

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        amp = params['amp'].value
        shift = params['shift'].value
        omega = params['omega'].value
        decay = params['decay'].value

        model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
        return model - data

    # create a set of Parameters
    params = Parameters()
    params.add('amp',   value= 10,  min=0)
    params.add('decay', value= 0.1)
    params.add('shift', value= 0.0, min=-pi / 2., max=pi / 2)
    params.add('omega', value= 3.0)

    # do fit, here with leastsq model
    result = minimize(fcn2min, params, args=(x, data))

    # calculate final result
    final = data + result.residual

    # write error report
    print(" --> SIMPLE --> ")
    print(result.params)
    report_fit(result.params)

    #assert that the real parameters are found

    for para, val in zip(result.params.values(), [5, 0.025, -.1, 2]):
        
        check(para, val)

def test_lbfgsb():
    p_true = Parameters()
    p_true.add('amp', value=14.0)
    p_true.add('period', value=5.33)
    p_true.add('shift', value=0.123)
    p_true.add('decay', value=0.010)

    def residual(pars, x, data=None):
        amp = pars['amp'].value
        per = pars['period'].value
        shift = pars['shift'].value
        decay = pars['decay'].value

        if abs(shift) > pi/2:
            shift = shift - np.sign(shift) * pi
        model = amp * np.sin(shift + x / per) * np.exp(-x * x * decay * decay)
        if data is None:
            return model
        return (model - data)

    n = 2500
    xmin = 0.
    xmax = 250.0
    noise = np.random.normal(scale=0.7215, size=n)
    x     = np.linspace(xmin, xmax, n)
    data  = residual(p_true, x) + noise

    fit_params = Parameters()
    fit_params.add('amp', value=11.0, min=5, max=20)
    fit_params.add('period', value=5., min=1., max=7)
    fit_params.add('shift', value=.10,  min=0.0, max=0.2)
    fit_params.add('decay', value=6.e-3, min=0, max=0.1)

    init = residual(fit_params, x)

    out = minimize(residual, fit_params, method='lbfgsb', args=(x,), kws={'data':data})

    fit = residual(fit_params, x)

    for name, par in out.params.items():
        nout = "%s:%s" % (name, ' '*(20-len(name)))
        print("%s: %s (%s) " % (nout, par.value, p_true[name].value))

    for para, true_para in zip(out.params.values(), p_true.values()):
        check_wo_stderr(para, true_para.value)

def test_derive():
    def func(pars, x, data=None):
        a = pars['a'].value
        b = pars['b'].value
        c = pars['c'].value

        model=a * np.exp(-b * x)+c
        if data is None:
            return model
        return (model - data)

    def dfunc(pars, x, data=None):
        a = pars['a'].value
        b = pars['b'].value
        c = pars['c'].value
        v = np.exp(-b*x)
        return np.array([v, -a*x*v, np.ones(len(x))])

    def f(var, x):
        return var[0]* np.exp(-var[1] * x)+var[2]

    params1 = Parameters()
    params1.add('a', value=10)
    params1.add('b', value=10)
    params1.add('c', value=10)

    params2 = Parameters()
    params2.add('a', value=10)
    params2.add('b', value=10)
    params2.add('c', value=10)

    a, b, c = 2.5, 1.3, 0.8
    x = np.linspace(0,4,50)
    y = f([a, b, c], x)
    data = y + 0.15*np.random.normal(size=len(x))

    # fit without analytic derivative
    min1 = Minimizer(func, params1, fcn_args=(x,), fcn_kws={'data':data})
    out1 = min1.leastsq()
    fit1 = func(out1.params, x)

    # fit with analytic derivative
    min2 = Minimizer(func, params2, fcn_args=(x,), fcn_kws={'data':data})
    out2 = min2.leastsq(Dfun=dfunc, col_deriv=1)
    fit2 = func(out2.params, x)

    
    print ('''Comparison of fit to exponential decay
    with and without analytic derivatives, to
       model = a*exp(-b*x) + c
    for a = %.2f, b = %.2f, c = %.2f
    ==============================================
    Statistic/Parameter|   Without   | With      |
    ----------------------------------------------
    N Function Calls   |   %3i       |   %3i     |
    Chi-square         |   %.4f    |   %.4f  |
       a               |   %.4f    |   %.4f  |
       b               |   %.4f    |   %.4f  |
       c               |   %.4f    |   %.4f  |
    ----------------------------------------------
    ''' %  (a, b, c,
            out1.nfev,   out2.nfev,
            out1.chisqr, out2.chisqr,
            out1.params['a'].value, out2.params['a'].value,
            out1.params['b'].value, out2.params['b'].value,
            out1.params['c'].value, out2.params['c'].value ))

    check_wo_stderr(out1.params['a'], out2.params['a'].value, 0.00005)
    check_wo_stderr(out1.params['b'], out2.params['b'].value, 0.00005)
    check_wo_stderr(out1.params['c'], out2.params['c'].value, 0.00005)

def test_peakfit():
    def residual(pars, x, data=None):
        g1 = gaussian(x, pars['a1'].value, pars['c1'].value, pars['w1'].value)
        g2 = gaussian(x, pars['a2'].value, pars['c2'].value, pars['w2'].value)
        model = g1 + g2
        if data is None:
            return model
        return (model - data)

    n    = 601
    xmin = 0.
    xmax = 15.0
    noise = np.random.normal(scale=.65, size=n)
    x = np.linspace(xmin, xmax, n)

    org_params = Parameters()
    org_params.add_many(('a1', 12.0, True, None, None, None),
                        ('c1',  5.3, True, None, None, None),
                        ('w1',  1.0, True, None, None, None),
                        ('a2',  9.1, True, None, None, None),
                        ('c2',  8.1, True, None, None, None),
                        ('w2',  2.5, True, None, None, None))

    data  = residual(org_params, x) + noise


    fit_params = Parameters()
    fit_params.add_many(('a1',  8.0, True, None, 14., None),
                        ('c1',  5.0, True, None, None, None),
                        ('w1',  0.7, True, None, None, None),
                        ('a2',  3.1, True, None, None, None),
                        ('c2',  8.8, True, None, None, None))

    fit_params.add('w2', expr='2.5*w1')

    myfit = Minimizer(residual, fit_params,
                      fcn_args=(x,), fcn_kws={'data':data})

    myfit.prepare_fit()

    init = residual(fit_params, x)


    out = myfit.leastsq()

    # print(' N fev = ', myfit.nfev)
    # print(myfit.chisqr, myfit.redchi, myfit.nfree)

    report_fit(out.params)

    fit = residual(out.params, x)
    check_paras(out.params, org_params)


def test_scalar_minimize_has_no_uncertainties():
    # scalar_minimize doesn't calculate uncertainties.
    # when a scalar_minimize is run the stderr and correl for each parameter
    # should be None. (stderr and correl are set to None when a Parameter is
    # initialised).
    # This requires a reset after a leastsq fit has been done.
    # Only when scalar_minimize calculates stderr and correl can this test
    # be removed.

    np.random.seed(1)
    x = np.linspace(0, 15, 301)
    data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
            np.random.normal(size=len(x), scale=0.2) )

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        amp = params['amp'].value
        shift = params['shift'].value
        omega = params['omega'].value
        decay = params['decay'].value

        model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
        return model - data

    # create a set of Parameters
    params = Parameters()
    params.add('amp',   value= 10,  min=0)
    params.add('decay', value= 0.1)
    params.add('shift', value= 0.0, min=-pi / 2., max=pi / 2)
    params.add('omega', value= 3.0)

    mini = Minimizer(fcn2min, params, fcn_args=(x, data))
    out = mini.minimize()
    assert_(np.isfinite(out.params['amp'].stderr))
    print(out.errorbars)
    assert_(out.errorbars == True)
    out2 = mini.minimize(method='nelder-mead')
    assert_(out2.params['amp'].stderr is None)
    assert_(out2.params['decay'].stderr is None)
    assert_(out2.params['shift'].stderr is None)
    assert_(out2.params['omega'].stderr is None)
    assert_(out2.params['amp'].correl is None)
    assert_(out2.params['decay'].correl is None)
    assert_(out2.params['shift'].correl is None)
    assert_(out2.params['omega'].correl is None)
    assert_(out2.errorbars == False)


def test_multidimensional_fit_GH205():
    # test that you don't need to flatten the output from the objective
    # function. Tests regression for GH205.
    pos = np.linspace(0, 99, 100)
    xv, yv = np.meshgrid(pos, pos)
    f = lambda xv, yv, lambda1, lambda2: (np.sin(xv * lambda1)
                                             + np.cos(yv * lambda2))

    data = f(xv, yv, 0.3, 3)
    assert_(data.ndim, 2)

    def fcn2min(params, xv, yv, data):
        """ model decaying sine wave, subtract data"""
        lambda1 = params['lambda1'].value
        lambda2 = params['lambda2'].value
        model = f(xv, yv, lambda1, lambda2)
        return model - data

    # create a set of Parameters
    params = Parameters()
    params.add('lambda1', value=0.4)
    params.add('lambda2', value=3.2)

    mini = Minimizer(fcn2min, params, fcn_args=(xv, yv, data))
    res = mini.minimize()

class CommonMinimizerTest(unittest.TestCase):

    def setUp(self):
        """
        test scale minimizers except newton-cg (needs jacobian) and
        anneal (doesn't work out of the box).
        """
        p_true = Parameters()
        p_true.add('amp', value=14.0)
        p_true.add('period', value=5.33)
        p_true.add('shift', value=0.123)
        p_true.add('decay', value=0.010)
        self.p_true = p_true

        n = 2500
        xmin = 0.
        xmax = 250.0
        noise = np.random.normal(scale=0.7215, size=n)
        self.x     = np.linspace(xmin, xmax, n)
        data  = self.residual(p_true, self.x) + noise

        fit_params = Parameters()
        fit_params.add('amp', value=11.0, min=5, max=20)
        fit_params.add('period', value=5., min=1., max=7)
        fit_params.add('shift', value=.10,  min=0.0, max=0.2)
        fit_params.add('decay', value=6.e-3, min=0, max=0.1)
        self.fit_params = fit_params

        init = self.residual(fit_params, self.x)
        self.mini = Minimizer(self.residual, fit_params, [self.x, data])

    def residual(self, pars, x, data=None):
        amp = pars['amp'].value
        per = pars['period'].value
        shift = pars['shift'].value
        decay = pars['decay'].value

        if abs(shift) > pi/2:
            shift = shift - np.sign(shift) * pi
        model = amp*np.sin(shift + x/per) * np.exp(-x*x*decay*decay)
        if data is None:
            return model
        return (model - data)
        
    def test_diffev_bounds_check(self):
        # You need finite (min, max) for each parameter if you're using
        # differential_evolution.
        self.fit_params['decay'].min = None
        self.minimizer = 'differential_evolution'
        np.testing.assert_raises(ValueError, self.scalar_minimizer)

    def test_scalar_minimizers(self):
        # test all the scalar minimizers
        for method in SCALAR_METHODS:
            if method in ['newton', 'dogleg', 'trust-ncg']:
                continue
            self.minimizer = SCALAR_METHODS[method]
            if method == 'Nelder-Mead':
                sig = 0.2
            else:
                sig = 0.15
            self.scalar_minimizer(sig=sig)
        
    def scalar_minimizer(self, sig=0.15):
        try:
            from scipy.optimize import minimize as scipy_minimize
        except ImportError:
            raise SkipTest

        print(self.minimizer)
        out = self.mini.scalar_minimize(method=self.minimizer)

        fit = self.residual(out.params, self.x)

        for name, par in out.params.items():
            nout = "%s:%s" % (name, ' '*(20-len(name)))
            print("%s: %s (%s) " % (nout, par.value, self.p_true[name].value))

        for para, true_para in zip(out.params.values(),
                                   self.p_true.values()):
            check_wo_stderr(para, true_para.value, sig=sig)


if __name__ == '__main__':
    nose.main()
