import numpy as np
from scipy.special import gamma, gammaln, beta, betaln, erf, erfc, wofz
from numpy import pi
from lmfit import Model
from lmfit.utilfuncs import (gaussian, normalized_gaussian, exponential,
                             powerlaw, linear, parabolic)


class DimensionalError(Exception):
    pass


def _validate_1d(independent_vars):
    if len(independent_vars) != 1:
        raise DimensionalError(
            "This model requires exactly one independent variable.")


def _suffixer(suffix, coded_param_names):
    """Return a dictionary relating parmeters' hard-coded names to their
    (possibly) suffixed names."""
    if suffix is None:
        param_names = coded_param_names
    else:
        param_names = map(lambda p: p + suffix, coded_param_names)
    return dict(zip(coded_param_names, param_names))


class BaseModel(Model):
    """Whereas Model takes a user-provided function, BaseModel should be
    subclassed with a hard-coded function."""

    def _parse_params(self):
        # overrides method of Model that inspects func
        param_names = _suffixer(self.suffix, self._param_names) 
        self.param_names = set(param_names.values())  # used by Model
        return param_names  # a lookup dictionary


class Parabolic(BaseModel):
    def __init__(self, independent_vars, missing='none', suffix=None):
        _validate_1d(independent_vars)
        var_name, = independent_vars
        self.suffix = suffix
        self._param_names = ['a', 'b', 'c']
        p = self._parse_params()
        def func(**kwargs):
            a = kwargs[p['a']]
            b = kwargs[p['b']]
            c = kwargs[p['c']]
            var = kwargs[var_name]
            return parabolic(var, a, b, c)
        super(Parabolic, self).__init__(func, independent_vars, missing)


Quadratic = Parabolic  # synonym


class Linear(BaseModel):
    def __init__(self, independent_vars, missing='none', suffix=None):
        _validate_1d(independent_vars)
        var_name, = independent_vars
        self.suffix = suffix
        self._param_names = ['slope', 'intercept']
        p = self._parse_params()
        def func(**kwargs):
            slope = kwargs[p['slope']]
            intercept = kwargs[p['intercept']]
            var = kwargs[var_name]
            return linear(var, slope, intercept)
        super(Linear, self).__init__(func, independent_vars, missing)


class Constant(BaseModel):
    def __init__(self, independent_vars=[], missing='none', suffix=None):
        # special case with default []
        self.suffix = suffix
        self._param_names = ['c']
        p = self._parse_params()
        def func(**kwargs):
            c = kwargs[p['c']]
            return c
        super(Constant, self).__init__(func, independent_vars, missing)


class Polynomial(BaseModel):
    def __init__(self, order, independent_vars, missing='none', suffix=None):
        if not isinstance(order, int):
            raise TypeError("order must be an integer.")
        _validate_1d(independent_vars)
        var_name, = independent_vars
        self.suffix = suffix
        self._param_names = ['c' + str(i) for i in range(order + 1)]
        p = self._parse_params()
        def func(**kwargs):
            var = kwargs[var_name]
            return np.sum([kwargs[p[name]]*var**i for 
                           i, name in enumerate(self._param_names)], 0)
        super(Polynomial, self).__init__(func, independent_vars, missing)


class Exponential(BaseModel):
    def __init__(self, independent_vars, missing='none', suffix=None):
        _validate_1d(independent_vars)
        var_name, = independent_vars
        self.suffix = suffix
        self._param_names = ['amplitude', 'decay']
        p = self._parse_params()
        def func(**kwargs):
            amplitude = kwargs[p['amplitude']]
            decay = kwargs[p['decay']]
            var = kwargs[var_name]
            return exponential(var, amplitude, decay)
        super(Exponential, self).__init__(func, independent_vars, missing)


class NormalizedGaussian(BaseModel):
    def __init__(self, independent_vars, missing='none', suffix=None):
        self.dim = len(independent_vars)

        if self.dim == 1:
            var_name, = independent_vars
            self.suffix = suffix
            self._param_names = ['center', 'sigma']
            p = self._parse_params()
            def func(**kwargs):
                center = kwargs[p['center']]
                sigma = kwargs[p['sigma']]
                var = kwargs[var_name]
                return normalized_gaussian(var, center, sigma)
        else:
            raise NotImplementedError("I only do 1d gaussians for now.")
            # TODO: Detect dimensionality from number of independent vars
        super(NormalizedGaussian, self).__init__(
            func, independent_vars, missing)


class Gaussian(BaseModel):
    def __init__(self, independent_vars, missing='none', suffix=None):
        self.dim = len(independent_vars)

        if self.dim == 1:
            var_name, = independent_vars
            self.suffix = suffix
            self._param_names = ['height', 'center', 'sigma']
            p = self._parse_params()
            def func(**kwargs):
                height = kwargs[p['height']]
                center = kwargs[p['center']]
                sigma = kwargs[p['sigma']]
                var = kwargs[var_name]
                return gaussian(var, height, center, sigma)
        else:
            raise NotImplementedError("I only do 1d gaussians for now.")
            # TODO: Detect dimensionality from number of independent vars
        super(Gaussian, self).__init__(func, independent_vars, missing)


class PowerLaw(BaseModel):
    def __init__(self, independent_vars, missing='none', suffix=None):
        _validate_1d(independent_vars)
        var_name, = independent_vars
        self.suffix = suffix
        self._param_names = ['coefficient', 'exponent']
        p = self._parse_params()
        def func(**kwargs):
            coefficient = kwargs[p['coefficient']]
            exponent = kwargs[p['exponent']]
            var = kwargs[var_name]
            return powerlaw(var, coefficient, exponent)
        super(PowerLaw, self).__init__(func, independent_vars, missing)
