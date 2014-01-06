# -*- coding: utf-8 -*-
from __future__ import print_function
from lmfit import minimize, Parameters, Parameter, report_fit, Minimizer
import numpy as np
pi = np.pi
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
    params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
    params.add('omega', value= 3.0)


    # do fit, here with leastsq model
    result = minimize(fcn2min, params, args=(x, data))

    # calculate final result
    final = data + result.residual

    # write error report
    report_fit(params)

    #assert that the real parameters are found

    for para, val in zip(params.values(), [5, 0.025, -.1, 2]):
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
            shift = shift - np.sign(shift)*pi
        model = amp*np.sin(shift + x/per) * np.exp(-x*x*decay*decay)
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

    for name, par in fit_params.items():
        nout = "%s:%s" % (name, ' '*(20-len(name)))
        print("%s: %s (%s) " % (nout, par.value, p_true[name].value))

    for para, true_para in zip(fit_params.values(), p_true.values()):
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
        return [v, -a*x*v, np.ones(len(x))]

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
    min1.leastsq()
    fit1 = func(params1, x)

    # fit with analytic derivative
    min2 = Minimizer(func, params2, fcn_args=(x,), fcn_kws={'data':data})
    min2.leastsq(Dfun=dfunc, col_deriv=1)
    fit2 = func(params2, x)

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
            min1.nfev,   min2.nfev,
            min1.chisqr, min2.chisqr,
            params1['a'].value, params2['a'].value,
            params1['b'].value, params2['b'].value,
            params1['c'].value, params2['c'].value ))

    check_wo_stderr(min1.params['a'], min2.params['a'].value, 0.000001)
    check_wo_stderr(min1.params['b'], min2.params['b'].value, 0.000001)
    check_wo_stderr(min1.params['c'], min2.params['c'].value, 0.000001)

def test_peakfit():
    from lmfit.utilfuncs import gaussian
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


    myfit.leastsq()

    print(' N fev = ', myfit.nfev)
    print(myfit.chisqr, myfit.redchi, myfit.nfree)

    report_fit(fit_params)

    fit = residual(fit_params, x)
    check_paras(fit_params, org_params)


class CommonMinimizerTest(object):

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
            shift = shift - np.sign(shift)*pi
        model = amp*np.sin(shift + x/per) * np.exp(-x*x*decay*decay)
        if data is None:
            return model
        return (model - data)

    def test_scalar_minimizer(self):
        try:
            from scipy.optimize import minimize as scipy_minimize
        except ImportError:
            raise SkipTest

        print(self.minimizer)
        self.mini.scalar_minimize(method=self.minimizer)

        fit = self.residual(self.fit_params, self.x)

        for name, par in self.fit_params.items():
            nout = "%s:%s" % (name, ' '*(20-len(name)))
            print("%s: %s (%s) " % (nout, par.value, self.p_true[name].value))

        for para, true_para in zip(self.fit_params.values(),
                                   self.p_true.values()):
            check_wo_stderr(para, true_para.value)

class TestNelder_Mead(CommonMinimizerTest, unittest.TestCase):

    def setUp(self):
        self.minimizer = 'Nelder-Mead'
        super(TestNelder_Mead, self).setUp()

    # override default because Nelder-Mead is less precise
    def test_scalar_minimizer(self):
        try:
            from scipy.optimize import minimize as scipy_minimize
        except ImportError:
            raise SkipTest

        print(self.minimizer)
        self.mini.scalar_minimize(method=self.minimizer)

        fit = self.residual(self.fit_params, self.x)

        for name, par in self.fit_params.items():
            nout = "%s:%s" % (name, ' '*(20-len(name)))
            print("%s: %s (%s) " % (nout, par.value, self.p_true[name].value))

        for para, true_para in zip(self.fit_params.values(),
                                   self.p_true.values()):
            check_wo_stderr(para, true_para.value, sig=0.2)


class TestL_BFGS_B(CommonMinimizerTest, unittest.TestCase):

    def setUp(self):
        self.minimizer = 'L-BFGS-B'
        super(TestL_BFGS_B, self).setUp()


class TestTNC(CommonMinimizerTest, unittest.TestCase):

    def setUp(self):
        self.minimizer = 'TNC'
        super(TestTNC, self).setUp()


class TestCOBYLA(CommonMinimizerTest, unittest.TestCase):

    def setUp(self):
        self.minimizer = 'COBYLA'
        super(TestCOBYLA, self).setUp()


class TestSLSQP(CommonMinimizerTest, unittest.TestCase):

    def setUp(self):
        self.minimizer = 'SLSQP'
        super(TestSLSQP, self).setUp()


class TestBFGS(CommonMinimizerTest, unittest.TestCase):

    def setUp(self):
        self.minimizer = 'BFGS'
        super(TestBFGS, self).setUp()


class TestCG(CommonMinimizerTest, unittest.TestCase):

    def setUp(self):
        self.minimizer = 'CG'
        super(TestCG, self).setUp()


class TestPowell(CommonMinimizerTest, unittest.TestCase):

    def setUp(self):
        self.minimizer = 'Powell'
        super(TestPowell, self).setUp()


if __name__ == '__main__':
    nose.main()
