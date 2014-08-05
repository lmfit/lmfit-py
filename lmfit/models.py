import numpy as np
from .model import Model
from .lineshapes import (gaussian, normalized_gaussian, exponential,
                         powerlaw, linear, parabolic)

class DimensionalError(Exception):
    pass

def _validate_1d(independent_vars):
    if len(independent_vars) != 1:
        raise DimensionalError(
            "This model requires exactly one independent variable.")

COMMON_DOC = """

Parameters
----------
independent_vars: list of strings to be set as variable names
missing: None, 'drop', or 'raise'
    None: Do not check for null or missing values.
    'drop': Drop null or missing observations in data.
        Use pandas.isnull if pandas is available; otherwise,
        silently fall back to numpy.isnan.
    'raise': Raise a (more helpful) exception when data contains null
        or missing values.
suffix: string to append to paramter names, needed to add two Models that
    have parameter names in common. None by default.
"""

class QuadraticModel(Model):
    __doc__ = parabolic.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(QuadraticModel, self).__init__(parabolic, **kwargs)


ParabolicModel = QuadraticModel

class LinearModel(Model):
    __doc__ = linear.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(LinearModel, self).__init__(linear, **kwargs)

class ConstantModel(Model):
    __doc__ = "x -> c" + COMMON_DOC
    def __init__(self, **kwargs):
        def func(x, c):
            return c
        super(ConstantModel, self).__init__(func, **kwargs)

class PolynomialModel(Model):
    __doc__ = "x -> c0 + c1 * x + c2 * x**2 + ... c7 * x**7" + COMMON_DOC
    def __init__(self, order, **kwargs):
        if not isinstance(order, int)  or order > 7:
            raise TypeError("order must be an integer less than 7.")
        kwargs['param_names'] = ['c%i' % i for i in range(order + 1)]
        def polynomial(x, c0=1,  c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0):
            out = np.zeros_like(x)
            args = dict(c0=c0,  c1=c1, c2=c2, c3=c3, c4=c4,
                        c5=c5, c6=c6, c7=c7)
            for i in range(order+1):
                out += x**i * args.get('c%i' % i, 0)
            return out
        super(PolynomialModel, self).__init__(polynomial, **kwargs)

class ExponentialModel(Model):
    __doc__ = exponential.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(ExponentialModel, self).__init__(exponential, **kwargs)

class GaussianModel(Model):
    __doc__ = gaussian.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(GaussianModel, self).__init__(gaussian, **kwargs)

class PowerLawModel(Model):
    __doc__ = powerlaw.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(PowerLawModel, self).__init__(powerlaw, **kwargs)

