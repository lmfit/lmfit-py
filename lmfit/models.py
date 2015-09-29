import numpy as np
from .model import Model

from .lineshapes import (gaussian, lorentzian, voigt, pvoigt, pearson7,
                         step, rectangle, breit_wigner, logistic,
                         students_t, lognormal, damped_oscillator,
                         expgaussian, skewed_gaussian, donaich,
                         skewed_voigt, exponential, powerlaw, linear,
                         parabolic)

from . import lineshapes

from .asteval import Interpreter
from .astutils import get_ast_names

class DimensionalError(Exception):
    pass

def _validate_1d(independent_vars):
    if len(independent_vars) != 1:
        raise DimensionalError(
            "This model requires exactly one independent variable.")

def index_of(arr, val):
    """return index of array nearest to a value
    """
    if val < min(arr):
        return 0
    return np.abs(arr-val).argmin()

def fwhm_expr(model):
    "return constraint expression for fwhm"
    return "%.7f*%ssigma" % (model.fwhm_factor, model.prefix)

def guess_from_peak(model, y, x, negative, ampscale=1.0, sigscale=1.0):
    "estimate amp, cen, sigma for a peak, create params"
    if x is None:
        return 1.0, 0.0, 1.0
    maxy, miny = max(y), min(y)
    maxx, minx = max(x), min(x)
    imaxy = index_of(y, maxy)
    cen = x[imaxy]
    amp = (maxy - miny)*2.0
    sig = (maxx-minx)/6.0

    halfmax_vals = np.where(y > (maxy+miny)/2.0)[0]
    if negative:
        imaxy = index_of(y, miny)
        amp = -(maxy - miny)*2.0
        halfmax_vals = np.where(y < (maxy+miny)/2.0)[0]
    if len(halfmax_vals) > 2:
        sig = (x[halfmax_vals[-1]] - x[halfmax_vals[0]])/2.0
        cen = x[halfmax_vals].mean()
    amp = amp*sig*ampscale
    sig = sig*sigscale

    pars = model.make_params(amplitude=amp, center=cen, sigma=sig)
    pars['%ssigma' % model.prefix].set(min=0.0)
    return pars

def update_param_vals(pars, prefix, **kwargs):
    """convenience function to update parameter values
    with keyword arguments"""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars

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
prefix: string to prepend to paramter names, needed to add two Models that
    have parameter names in common. None by default.
"""

class ConstantModel(Model):
    __doc__ = "x -> c" + COMMON_DOC
    def __init__(self, *args, **kwargs):
        def constant(x, c):
            return c
        super(ConstantModel, self).__init__(constant, *args, **kwargs)

    def guess(self, data, **kwargs):
        pars = self.make_params()
        pars['%sc' % self.prefix].set(value=data.mean())
        return update_param_vals(pars, self.prefix, **kwargs)


class LinearModel(Model):
    __doc__ = linear.__doc__ + COMMON_DOC if linear.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(LinearModel, self).__init__(linear, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        sval, oval = 0., 0.
        if x is not None:
            sval, oval = np.polyfit(x, data, 1)
        pars = self.make_params(intercept=oval, slope=sval)
        return update_param_vals(pars, self.prefix, **kwargs)


class QuadraticModel(Model):
    __doc__ = parabolic.__doc__ + COMMON_DOC if parabolic.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(QuadraticModel, self).__init__(parabolic, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        a, b, c = 0., 0., 0.
        if x is not None:
            a, b, c = np.polyfit(x, data, 2)
        pars = self.make_params(a=a, b=b, c=c)
        return update_param_vals(pars, self.prefix, **kwargs)

ParabolicModel = QuadraticModel

class PolynomialModel(Model):
    __doc__ = "x -> c0 + c1 * x + c2 * x**2 + ... c7 * x**7" + COMMON_DOC
    MAX_DEGREE=7
    DEGREE_ERR = "degree must be an integer less than %d."
    def __init__(self, degree, *args, **kwargs):
        if not isinstance(degree, int)  or degree > self.MAX_DEGREE:
            raise TypeError(self.DEGREE_ERR % self.MAX_DEGREE)

        self.poly_degree = degree
        pnames = ['c%i' % (i) for i in range(degree + 1)]
        kwargs['param_names'] = pnames

        def polynomial(x, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0):
            return np.polyval([c7, c6, c5, c4, c3, c2, c1, c0], x)

        super(PolynomialModel, self).__init__(polynomial, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        if x is not None:
            out = np.polyfit(x, data, self.poly_degree)
            for i, coef in enumerate(out[::-1]):
                pars['%sc%i'% (self.prefix, i)].set(value=coef)
        return update_param_vals(pars, self.prefix, **kwargs)


class GaussianModel(Model):
    __doc__ = gaussian.__doc__ + COMMON_DOC if gaussian.__doc__ else ""
    fwhm_factor = 2.354820
    def __init__(self, *args, **kwargs):
        super(GaussianModel, self).__init__(gaussian, *args, **kwargs)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)


class LorentzianModel(Model):
    __doc__ = lorentzian.__doc__ + COMMON_DOC if lorentzian.__doc__ else ""
    fwhm_factor = 2.0
    def __init__(self, *args, **kwargs):
        super(LorentzianModel, self).__init__(lorentzian, *args, **kwargs)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative, ampscale=1.25)
        return update_param_vals(pars, self.prefix, **kwargs)


class VoigtModel(Model):
    __doc__ = voigt.__doc__ + COMMON_DOC if voigt.__doc__ else ""
    fwhm_factor = 3.60131
    def __init__(self, *args, **kwargs):
        super(VoigtModel, self).__init__(voigt, *args, **kwargs)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('gamma', expr='%ssigma' % self.prefix)
        self.set_param_hint('fwhm',  expr=fwhm_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative,
                               ampscale=1.5, sigscale=0.65)
        return update_param_vals(pars, self.prefix, **kwargs)


class PseudoVoigtModel(Model):
    __doc__ = pvoigt.__doc__ + COMMON_DOC if pvoigt.__doc__ else ""
    fwhm_factor = 2.0
    def __init__(self, *args, **kwargs):
        super(PseudoVoigtModel, self).__init__(pvoigt, *args, **kwargs)
        self.set_param_hint('fraction', value=0.5)
        self.set_param_hint('fwhm',  expr=fwhm_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative, ampscale=1.25)
        pars['%sfraction' % self.prefix].set(value=0.5)
        return update_param_vals(pars, self.prefix, **kwargs)


class Pearson7Model(Model):
    __doc__ = pearson7.__doc__ + COMMON_DOC if pearson7.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(Pearson7Model, self).__init__(pearson7, *args, **kwargs)
        self.set_param_hint('expon',  value=1.5)

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative)
        pars['%sexpon' % self.prefix].set(value=1.5)
        return update_param_vals(pars, self.prefix, **kwargs)


class StudentsTModel(Model):
    __doc__ = students_t.__doc__ + COMMON_DOC if students_t.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(StudentsTModel, self).__init__(students_t, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)


class BreitWignerModel(Model):
    __doc__ = breit_wigner.__doc__ + COMMON_DOC if breit_wigner.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(BreitWignerModel, self).__init__(breit_wigner, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative)
        pars['%sq' % self.prefix].set(value=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class LognormalModel(Model):
    __doc__ = lognormal.__doc__ + COMMON_DOC if lognormal.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(LognormalModel, self).__init__(lognormal, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = self.make_params(amplitude=1.0, center=0.0, sigma=0.25)
        pars['%ssigma' % self.prefix].set(min=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class DampedOscillatorModel(Model):
    __doc__ = damped_oscillator.__doc__ + COMMON_DOC if damped_oscillator.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(DampedOscillatorModel, self).__init__(damped_oscillator, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        pars =guess_from_peak(self, data, x, negative,
                              ampscale=0.1, sigscale=0.1)
        return update_param_vals(pars, self.prefix, **kwargs)

class ExponentialGaussianModel(Model):
    __doc__ = expgaussian.__doc__ + COMMON_DOC if expgaussian.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(ExponentialGaussianModel, self).__init__(expgaussian, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

class SkewedGaussianModel(Model):
    __doc__ = skewed_gaussian.__doc__ + COMMON_DOC if skewed_gaussian.__doc__ else ""
    fwhm_factor = 2.354820
    def __init__(self, *args, **kwargs):
        super(SkewedGaussianModel, self).__init__(skewed_gaussian, *args, **kwargs)
        self.set_param_hint('sigma', min=0)

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

class DonaichModel(Model):
    __doc__ = donaich.__doc__ + COMMON_DOC if donaich.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(DonaichModel, self).__init__(donaich, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5)
        return update_param_vals(pars, self.prefix, **kwargs)


class PowerLawModel(Model):
    __doc__ = powerlaw.__doc__ + COMMON_DOC if powerlaw.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(PowerLawModel, self).__init__(powerlaw, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        try:
            expon, amp = np.polyfit(np.log(x+1.e-14), np.log(data+1.e-14), 1)
        except:
            expon, amp = 1, np.log(abs(max(data)+1.e-9))

        pars = self.make_params(amplitude=np.exp(amp), exponent=expon)
        return update_param_vals(pars, self.prefix, **kwargs)


class ExponentialModel(Model):
    __doc__ = exponential.__doc__ + COMMON_DOC if exponential.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(ExponentialModel, self).__init__(exponential, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        try:
            sval, oval = np.polyfit(x, np.log(abs(data)+1.e-15), 1)
        except:
            sval, oval = 1., np.log(abs(max(data)+1.e-9))
        pars = self.make_params(amplitude=np.exp(oval), decay=-1.0/sval)
        return update_param_vals(pars, self.prefix, **kwargs)


class StepModel(Model):
    __doc__ = step.__doc__ + COMMON_DOC if step.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(StepModel, self).__init__(step, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        if x is None:
            return
        ymin, ymax = min(data), max(data)
        xmin, xmax = min(x), max(x)
        pars = self.make_params(amplitude=(ymax-ymin),
                                center=(xmax+xmin)/2.0)
        pars['%ssigma' % self.prefix].set(value=(xmax-xmin)/7.0, min=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class RectangleModel(Model):
    __doc__ = rectangle.__doc__ + COMMON_DOC if rectangle.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(RectangleModel, self).__init__(rectangle, *args, **kwargs)
        self.set_param_hint('midpoint',
                            expr='(%scenter1+%scenter2)/2.0' % (self.prefix,
                                                                self.prefix))
    def guess(self, data, x=None, **kwargs):
        if x is None:
            return
        ymin, ymax = min(data), max(data)
        xmin, xmax = min(x), max(x)
        pars = self.make_params(amplitude=(ymax-ymin),
                                center1=(xmax+xmin)/4.0,
                                center2=3*(xmax+xmin)/4.0)
        pars['%ssigma1' % self.prefix].set(value=(xmax-xmin)/7.0, min=0.0)
        pars['%ssigma2' % self.prefix].set(value=(xmax-xmin)/7.0, min=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class ExpressionModel(Model):
    """Model from User-supplied expression

Parameters
----------
expr:    string of mathematical expression for model.
independent_vars: list of strings to be set as variable names
missing: None, 'drop', or 'raise'
    None: Do not check for null or missing values.
    'drop': Drop null or missing observations in data.
        Use pandas.isnull if pandas is available; otherwise,
        silently fall back to numpy.isnan.
    'raise': Raise a (more helpful) exception when data contains null
        or missing values.
prefix: NOT supported for ExpressionModel
"""

    idvar_missing  = "No independent variable found in\n %s"
    idvar_notfound = "Cannot find independent variables '%s' in\n %s"
    no_prefix      = "ExpressionModel does not support `prefix` argument"
    def __init__(self, expr, independent_vars=None, init_script=None,
                 *args, **kwargs):

        # create ast evaluator, load custom functions
        self.asteval = Interpreter()
        for name in lineshapes.functions:
            self.asteval.symtable[name] = getattr(lineshapes, name, None)
        if init_script is not None:
            self.asteval.eval(init_script)

        # save expr as text, parse to ast, save for later use
        self.expr = expr.strip()
        self.astcode = self.asteval.parse(self.expr)

        # find all symbol names found in expression
        sym_names = get_ast_names(self.astcode)

        if independent_vars is None and 'x' in sym_names:
            independent_vars = ['x']
        if independent_vars is None:
            raise ValueError(self.idvar_missing % (self.expr))

        # determine which named symbols are parameter names,
        # try to find all independent variables
        idvar_found = [False]*len(independent_vars)
        param_names = []
        for name in sym_names:
            if name in independent_vars:
                idvar_found[independent_vars.index(name)] = True
            elif name not in self.asteval.symtable:
                param_names.append(name)

        # make sure we have all independent parameters
        if not all(idvar_found):
            lost = []
            for ix, found in enumerate(idvar_found):
                if not found:
                    lost.append(independent_vars[ix])
            lost = ', '.join(lost)
            raise ValueError(self.idvar_notfound % (lost, self.expr))

        kwargs['independent_vars'] = independent_vars
        if 'prefix' in kwargs:
            raise Warning(self.no_prefix)

        def _eval(**kwargs):
            for name, val in kwargs.items():
                self.asteval.symtable[name] = val
            return self.asteval.run(self.astcode)

        super(ExpressionModel, self).__init__(_eval, *args, **kwargs)

        # set param names here, and other things normally
        # set in _parse_params(), which will be short-circuited.
        self.independent_vars = independent_vars
        self._func_allargs = independent_vars + param_names
        self._param_names = set(param_names)
        self._func_haskeywords = True
        self.def_vals = {}

    def __repr__(self):
        return  "<lmfit.ExpressionModel('%s')>" % (self.expr)

    def _parse_params(self):
        """ExpressionModel._parse_params is over-written (as `pass`)
        to prevent normal parsing of function for parameter names
        """
        pass
