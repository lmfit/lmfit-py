# -*- coding: utf-8 -*-
from __future__ import print_function
from lmfit import minimize, Parameters, Parameter, report_fit, Minimizer
from lmfit.minimizer import (SCALAR_METHODS, HAS_EMCEE,
                             MinimizerResult, _lnpost, _nan_policy)
from lmfit.lineshapes import gaussian
import numpy as np
from numpy import pi
from numpy.testing import (assert_, decorators, assert_raises,
                           assert_almost_equal, assert_equal)
import unittest
import nose
from nose import SkipTest


def check(para, real_val, sig=3):
    err = abs(para.value - real_val)
    print('Check Param w/ stderr: ',  para.name, para.value, real_val, para.stderr)
    assert(err < sig * para.stderr)

def check_wo_stderr(para, real_val, sig=0.1):
    err = abs(para.value - real_val)
    print('Check Param w/o stderr: ', para.name, para.value, real_val, sig)
    assert(err < sig)

def check_paras(para_fit, para_real, sig=3):
    for i in para_fit:
        check(para_fit[i], para_real[i].value, sig=sig)

def test_simple():
    # create data to be fitted
    np.random.seed(1)
    x = np.linspace(0, 15, 301)
    data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
            np.random.normal(size=len(x), scale=0.2))

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        amp = params['amp']
        shift = params['shift']
        omega = params['omega']
        decay = params['decay']

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
        amp = pars['amp']
        per = pars['period']
        shift = pars['shift']
        decay = pars['decay']

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
        model= pars['a'] * np.exp(-pars['b'] * x) + pars['c']
        if data is None:
            return model
        return model - data

    def dfunc(pars, x, data=None):
        v = np.exp(-pars['b']*x)
        return np.array([v, -pars['a']*x*v, np.ones(len(x))])

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
        g1 = gaussian(x, pars['a1'], pars['c1'], pars['w1'])
        g2 = gaussian(x, pars['a2'], pars['c2'], pars['w2'])
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
                      fcn_args=(x,), fcn_kws={'data': data})

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
        amp = params['amp']
        shift = params['shift']
        omega = params['omega']
        decay = params['decay']

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
        model = f(xv, yv, params['lambda1'], params['lambda2'])
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
        self.x = np.linspace(xmin, xmax, n)
        self.data = self.residual(p_true, self.x) + noise

        fit_params = Parameters()
        fit_params.add('amp', value=11.0, min=5, max=20)
        fit_params.add('period', value=5., min=1., max=7)
        fit_params.add('shift', value=.10,  min=0.0, max=0.2)
        fit_params.add('decay', value=6.e-3, min=0, max=0.1)
        self.fit_params = fit_params

        self.mini = Minimizer(self.residual, fit_params, [self.x, self.data])

    def residual(self, pars, x, data=None):
        amp = pars['amp']
        per = pars['period']
        shift = pars['shift']
        decay = pars['decay']

        if abs(shift) > pi/2:
            shift = shift - np.sign(shift) * pi
        model = amp*np.sin(shift + x/per) * np.exp(-x*x*decay*decay)
        if data is None:
            return model
        return model - data

    def test_diffev_bounds_check(self):
        # You need finite (min, max) for each parameter if you're using
        # differential_evolution.
        self.fit_params['decay'].min = -np.inf
        self.fit_params['decay'].vary = True
        self.minimizer = 'differential_evolution'
        np.testing.assert_raises(ValueError, self.scalar_minimizer)

        # but only if a parameter is not fixed
        self.fit_params['decay'].vary = False
        self.mini.scalar_minimize(method='differential_evolution', maxiter=1)

    def test_scalar_minimizers(self):
        # test all the scalar minimizers
        for method in SCALAR_METHODS:
            if method in ['newton', 'dogleg', 'trust-ncg', 'cg']:
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

        self.residual(out.params, self.x)

        for name, par in out.params.items():
            nout = "%s:%s" % (name, ' '*(20-len(name)))
            print("%s: %s (%s) " % (nout, par.value, self.p_true[name].value))

        for para, true_para in zip(out.params.values(),
                                   self.p_true.values()):
            check_wo_stderr(para, true_para.value, sig=sig)

    def test_nan_policy(self):
        # check that an error is raised if there are nan in
        # the data returned by userfcn
        self.data[0] = np.nan

        for method in SCALAR_METHODS:
            assert_raises(ValueError,
                          self.mini.scalar_minimize,
                          SCALAR_METHODS[method])

        assert_raises(ValueError, self.mini.minimize)

        # now check that the fit proceeds if nan_policy is 'omit'
        self.mini.nan_policy = 'omit'
        res = self.mini.minimize()
        assert_equal(res.ndata, np.size(self.data, 0) - 1)

        for para, true_para in zip(res.params.values(),
                                   self.p_true.values()):
            check_wo_stderr(para, true_para.value, sig=0.15)

    def test_nan_policy_function(self):
        a = np.array([0, 1, 2, 3, np.nan])
        assert_raises(ValueError, _nan_policy, a)
        assert_(np.isnan(_nan_policy(a, nan_policy='propagate')[-1]))
        assert_equal(_nan_policy(a, nan_policy='omit'), [0, 1, 2, 3])

        a[-1] = np.inf
        assert_raises(ValueError, _nan_policy, a)
        assert_(np.isposinf(_nan_policy(a, nan_policy='propagate')[-1]))
        assert_equal(_nan_policy(a, nan_policy='omit'), [0, 1, 2, 3])
        assert_equal(_nan_policy(a, handle_inf=False), a)

    @decorators.slow
    def test_emcee(self):
        # test emcee
        if not HAS_EMCEE:
            return True

        np.random.seed(123456)
        out = self.mini.emcee(nwalkers=100, steps=200,
                                      burn=50, thin=10)

        check_paras(out.params, self.p_true, sig=3)

    @decorators.slow
    def test_emcee_PT(self):
        # test emcee with parallel tempering
        if not HAS_EMCEE:
            return True

        np.random.seed(123456)
        self.mini.userfcn = residual_for_multiprocessing
        out = self.mini.emcee(ntemps=4, nwalkers=50, steps=200,
                              burn=100, thin=10, workers=2)

        check_paras(out.params, self.p_true, sig=3)

    @decorators.slow
    def test_emcee_multiprocessing(self):
        # test multiprocessing runs
        if not HAS_EMCEE:
            return True

        np.random.seed(123456)
        self.mini.userfcn = residual_for_multiprocessing
        out = self.mini.emcee(steps=10, workers=4)

    def test_emcee_bounds_length(self):
        # the log-probability functions check if the parameters are
        # inside the bounds. Check that the bounds and parameters
        # are the right lengths for comparison. This can be done
        # if nvarys != nparams
        if not HAS_EMCEE:
            return True
        self.mini.params['amp'].vary=False
        self.mini.params['period'].vary=False
        self.mini.params['shift'].vary=False

        out = self.mini.emcee(steps=10)

    @decorators.slow
    def test_emcee_partial_bounds(self):
        # mcmc with partial bounds
        if not HAS_EMCEE:
            return True

        np.random.seed(123456)
        # test mcmc output vs lm, some parameters not bounded
        self.fit_params['amp'].max = np.inf
        # self.fit_params['amp'].min = -np.inf
        out = self.mini.emcee(nwalkers=100, steps=300,
                                      burn=100, thin=10)

        check_paras(out.params, self.p_true, sig=3)

    def test_emcee_init_with_chain(self):
        # can you initialise with a previous chain
        if not HAS_EMCEE:
            return True

        out = self.mini.emcee(nwalkers=100, steps=5)
        # can initialise with a chain
        out2 = self.mini.emcee(nwalkers=100, steps=1, pos=out.chain)

        # can initialise with a correct subset of a chain
        out3 = self.mini.emcee(nwalkers=100,
                               steps=1,
                               pos=out.chain[..., -1, :])

        # but you can't initialise if the shape is wrong.
        assert_raises(ValueError,
                      self.mini.emcee,
                      nwalkers=100,
                      steps=1,
                      pos=out.chain[..., -1, :-1])

    def test_emcee_reuse_sampler(self):
        if not HAS_EMCEE:
            return True

        self.mini.emcee(nwalkers=100, steps=5)

        # if you've run the sampler the Minimizer object should have a _lastpos
        # attribute
        assert_(hasattr(self.mini, '_lastpos'))

        # now try and re-use sampler
        out2 = self.mini.emcee(steps=10, reuse_sampler=True)
        assert_(out2.chain.shape[1] == 15)

        # you shouldn't be able to reuse the sampler if nvarys has changed.
        self.mini.params['amp'].vary = False
        assert_raises(ValueError, self.mini.emcee, reuse_sampler=True)

    def test_emcee_lnpost(self):
        # check ln likelihood is calculated correctly. It should be
        # -0.5 * chi**2.
        result = self.mini.minimize()

        # obtain the numeric values
        # note - in this example all the parameters are varied
        fvars = np.array([par.value for par in result.params.values()])

        # calculate the cost function with scaled values (parameters all have
        # lower and upper bounds.
        scaled_fvars = []
        for par, fvar in zip(result.params.values(), fvars):
            par.value = fvar
            scaled_fvars.append(par.setup_bounds())

        val = self.mini.penalty(np.array(scaled_fvars))

        # calculate the log-likelihood value
        bounds = np.array([(par.min, par.max)
                           for par in result.params.values()])
        val2 = _lnpost(fvars,
                       self.residual,
                       result.params,
                       result.var_names,
                       bounds,
                       userargs=(self.x, self.data))

        assert_almost_equal(-0.5 * val, val2)

    def test_emcee_output(self):
        # test mcmc output
        if not HAS_EMCEE:
            return True
        try:
            from pandas import DataFrame
        except ImportError:
            return True
        out = self.mini.emcee(nwalkers=10, steps=20, burn=5, thin=2)
        assert_(isinstance(out, MinimizerResult))
        assert_(isinstance(out.flatchain, DataFrame))

        # check that we can access the chains via parameter name
        assert_(out.flatchain['amp'].shape[0] == 80)
        assert_(out.errorbars is True)
        assert_(np.isfinite(out.params['amp'].correl['period']))

        # the lnprob array should be the same as the chain size
        assert_(np.size(out.chain)//4 == np.size(out.lnprob))

    @decorators.slow
    def test_emcee_float(self):
        # test that it works if the residuals returns a float, not a vector
        if not HAS_EMCEE:
            return True

        def resid(pars, x, data=None):
            return -0.5 * np.sum(self.residual(pars, x, data=data)**2)

        # just return chi2
        def resid2(pars, x, data=None):
            return np.sum(self.residual(pars, x, data=data)**2)

        self.mini.userfcn = resid
        np.random.seed(123456)
        out = self.mini.emcee(nwalkers=100, steps=200,
                                      burn=50, thin=10)
        check_paras(out.params, self.p_true, sig=3)

        self.mini.userfcn = resid2
        np.random.seed(123456)
        out = self.mini.emcee(nwalkers=100, steps=200,
                              burn=50, thin=10, float_behavior='chi2')
        check_paras(out.params, self.p_true, sig=3)

    @decorators.slow
    def test_emcee_seed(self):
        # test emcee seeding can reproduce a sampling run
        if not HAS_EMCEE:
            return True

        out = self.mini.emcee(params=self.fit_params,
                              nwalkers=100,
                              steps=1, seed=1)
        out2 = self.mini.emcee(params=self.fit_params,
                               nwalkers=100,
                               steps=1, seed=1)

        assert_almost_equal(out.chain, out2.chain)


def residual_for_multiprocessing(pars, x, data=None):
    # a residual function defined in the top level is needed for
    # multiprocessing. bound methods don't work.
    amp = pars['amp']
    per = pars['period']
    shift = pars['shift']
    decay = pars['decay']

    if abs(shift) > pi/2:
        shift = shift - np.sign(shift) * pi
    model = amp*np.sin(shift + x/per) * np.exp(-x*x*decay*decay)
    if data is None:
        return model
    return (model - data)


if __name__ == '__main__':
    nose.main()
