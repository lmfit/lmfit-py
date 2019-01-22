# Benchmarking scripts for lmfit

from copy import deepcopy

import numpy as np

from lmfit import Minimizer, Parameters, conf_interval, minimize


def obj_func(params, x, data):
    """ decaying sine wave, subtract data"""
    amp = params['amp'].value
    shift = params['shift'].value
    omega = params['omega'].value
    decay = params['decay'].value
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
    return model - data


class MinimizeSuite:
    """
    Benchmarks using minimize() and least-squares
    """
    def setup(self):
        pass

    def time_minimize(self):
        np.random.seed(201)
        x = np.linspace(0, 15, 601)

        data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
                np.random.normal(size=len(x), scale=0.3))
        params = Parameters()
        params.add('amp', value=1, min=0, max=100)
        params.add('decay', value=0.0, min=0, max=10)
        params.add('shift', value=0.0, min=-np.pi/2., max=np.pi/2)
        params.add('omega', value=1.0, min=0, max=10)

        return minimize(obj_func, params, args=(x, data))

    def time_minimize_withnan(self):
        np.random.seed(201)
        x = np.linspace(0, 15, 601)
        x[53] = np.nan

        data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
                np.random.normal(size=len(x), scale=0.3))
        params = Parameters()
        params.add('amp', value=1, min=0, max=100)
        params.add('decay', value=0.0, min=0, max=10)
        params.add('shift', value=0.0, min=-np.pi/2., max=np.pi/2)
        params.add('omega', value=1.0, min=0, max=10)

        return minimize(obj_func, params, args=(x, data), nan_policy='omit')

    def time_minimize_large(self):
        np.random.seed(201)
        x = np.linspace(0, 19, 70001)
        data = (5. * np.sin(0.6*x - 0.1) * np.exp(-x*x*0.0165) +
                np.random.normal(size=len(x), scale=0.3))
        params = Parameters()
        params.add('amp', value=1, min=0, max=100)
        params.add('decay', value=0.0, min=0, max=10)
        params.add('shift', value=0.0, min=-np.pi/2., max=np.pi/2)
        params.add('omega', value=0.40, min=0, max=10)

        return minimize(obj_func, params, args=(x, data))

    def time_confinterval(self):
        np.random.seed(0)
        x = np.linspace(0.3, 10, 100)
        y = 1/(0.1*x)+2+0.1*np.random.randn(x.size)

        p = Parameters()
        p.add_many(('a', 0.1), ('b', 1))

        def residual(p):
            a = p['a'].value
            b = p['b'].value

            return 1/(a*x)+b-y

        minimizer = Minimizer(residual, p)
        out = minimizer.leastsq()
        return conf_interval(minimizer, out)


class MinimizerClassSuite:
    """
    Benchmarks for the Minimizer class
    """
    def setup(self):
        self.x = np.linspace(1, 10, 250)
        np.random.seed(0)
        self.y = (3.0 * np.exp(-self.x / 2)
                  - 5.0 * np.exp(-(self.x - 0.1) / 10.)
                  + 0.1 * np.random.randn(len(self.x)))

        self.p = Parameters()
        self.p.add_many(('a1', 4., True, 0., 10.),
                        ('a2', 4., True, -10., 10.),
                        ('t1', 3., True, 0.01, 10.),
                        ('t2', 3., True, 0.01, 20.))

        self.p_emcee = deepcopy(self.p)
        self.p_emcee.add('noise', 0.2, True, 0.001, 1.)

        self.mini_de = Minimizer(Minimizer_Residual,
                                 self.p,
                                 fcn_args=(self.x, self.y),
                                 kws={'seed': 1,
                                      'polish': False,
                                      'maxiter': 100})

        self.mini_emcee = Minimizer(Minimizer_lnprob,
                                    self.p_emcee,
                                    fcn_args=(self.x, self.y))

    def time_differential_evolution(self):
        return self.mini_de.minimize(method='differential_evolution')

    def time_emcee(self):
        return self.mini_emcee.emcee(self.p_emcee, steps=100, seed=1)


def Minimizer_Residual(p, x, y):
    v = p.valuesdict()
    return (v['a1'] * np.exp(-x / v['t1'])
            + v['a2'] * np.exp(-(x - 0.1) / v['t2'])
            - y)


def Minimizer_lnprob(p, x, y):
    noise = p['noise'].value
    return -0.5 * np.sum((Minimizer_Residual(p, x, y) / noise)**2
                         + np.log(2 * np.pi * noise**2))
