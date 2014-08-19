import numpy as np
from .model import Model
from .parameter import Parameter

from .lineshapes import (gaussian, lorentzian, voigt, pvoigt, pearson7,
                         step, rectangle, breit_wigner, logistic,
                         students_t, lognormal, damped_oscillator,
                         expgaussian, donaich, skewed_voigt, exponential,
                         powerlaw, linear, parabolic)

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

def estimate_peak(y, x, negative):
    "estimate amp, cen, sigma for a peak"
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
    return amp*sig, cen, sig

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
class ConstantModel(Model):
    __doc__ = "x -> c" + COMMON_DOC
    def __init__(self, **kwargs):
        def func(x, c):
            return c
        super(ConstantModel, self).__init__(func, **kwargs)

    def guess_starting_values(self, data, **kwargs):
        self.set_paramval('c', data.mean())
        self.has_initial_guess = True

class LinearModel(Model):
    __doc__ = linear.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(LinearModel, self).__init__(linear, **kwargs)

    def guess_starting_values(self, data, x=None, **kwargs):
        sval, oval = 0., 0.
        if x is not None:
            sval, oval = np.polyfit(x, data, 1)
        self.set_paramval('intercept', oval)
        self.set_paramval('sslope' , sval)
        self.has_initial_guess = True

class QuadraticModel(Model):
    __doc__ = parabolic.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(QuadraticModel, self).__init__(parabolic, **kwargs)

    def guess_starting_values(self, data, x=None, **kwargs):
        a, b, c = 0., 0., 0.
        if x is not None:
            a, b, c = np.polyfit(x, data, 2)
        self.set_paramval('a', a)
        self.set_paramval('b', b)
        self.set_paramval('c', c)
        self.has_initial_guess = True

ParabolicModel = QuadraticModel

class PolynomialModel(Model):
    __doc__ = "x -> c0 + c1 * x + c2 * x**2 + ... c7 * x**7" + COMMON_DOC
    MAX_DEGREE=7
    DEGREE_ERR = "degree must be an integer less than %d."
    def __init__(self, degree, **kwargs):
        if not isinstance(degree, int)  or degree > self.MAX_DEGREE:
            raise TypeError(self.DEGREE_ERR % self.MAX_DEGREE)

        self.poly_degree = degree
        pnames = ['c%i' % (i) for i in range(degree + 1)]
        kwargs['param_names'] = pnames

        def polynomial(x, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0):
            out = np.zeros_like(x)
            args = dict(c0=c0, c1=c1, c2=c2, c3=c3,
                        c4=c4, c5=c5, c6=c6, c7=c7)
            for i in range(self.poly_degree+1):
                out += x**i * args.get('c%i' % i, 0)
            return out
        super(PolynomialModel, self).__init__(polynomial, **kwargs)

    def guess_starting_values(self, data, x=None, **kws):
        coefs = np.zeros(self.MAX_DEGREE+1)
        if x is not None:
            out = np.polyfit(x, data, self.poly_degree)
            for i, coef in enumerate(out[::-1]):
                coefs[i] = coef
        for i in range(self.poly_degree+1):
            self.set_paramval('c%i' % (i), coefs[i])
        self.has_initial_guess = True

class GaussianModel(Model):
    __doc__ = gaussian.__doc__ + COMMON_DOC
    fwhm_factor = 2.354820
    def __init__(self, **kwargs):
        super(GaussianModel, self).__init__(gaussian, **kwargs)
        self.params.add('%sfwhm' % self.prefix,
                        expr='%.6f*%ssigma' % (self.fwhm_factor, self.prefix))

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.set_paramval('amplitude', amp)
        self.set_paramval('center', cen)
        self.set_paramval('sigma', sig)
        self.has_initial_guess = True

class LorentzianModel(Model):
    __doc__ = lorentzian.__doc__ + COMMON_DOC
    fwhm_factor = 2.0
    def __init__(self, **kwargs):
        super(LorentzianModel, self).__init__(lorentzian, **kwargs)
        self.params.add('%sfwhm' % self.prefix,
                        expr='%.7f*%ssigma' % (self.fwhm_factor,
                                               self.prefix))

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.set_paramval('amplitude', amp)
        self.set_paramval('center', cen)
        self.set_paramval('sigma', sig)
        self.has_initial_guess = True

class VoigtModel(Model):
    __doc__ = voigt.__doc__ + COMMON_DOC
    fwhm_factor = 3.60131
    def __init__(self, **kwargs):
        super(VoigtModel, self).__init__(voigt, **kwargs)
        self.params.add('%sfwhm' % self.prefix,
                        expr='%.7f*%ssigma' % (self.fwhm_factor,
                                               self.prefix))

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.set_paramval('amplitude', amp)
        self.set_paramval('center', cen)
        self.set_paramval('sigma', sig)
        self.params['%sgamma' % self.prefix] = \
                              Parameter(expr = '%ssigma' % self.prefix)
        self.has_initial_guess = True

class PseudoVoigtModel(Model):
    __doc__ = pvoigt.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(PseudoVoigtModel, self).__init__(pvoigt, **kwargs)

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.set_paramval('amplitude', amp)
        self.set_paramval('center', cen)
        self.set_paramval('sigma', sig)
        self.set_paramval('fraction', 0.5)
        self.has_initial_guess = True


class Pearson7Model(Model):
    __doc__ = pearson7.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(Pearson7Model, self).__init__(pearson7, **kwargs)

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.set_paramval('amplitude', amp)
        self.set_paramval('center', cen)
        self.set_paramval('sigma', sig)
        self.set_paramval('exponent', 0.5)
        self.has_initial_guess = True


class StudentsTModel(Model):
    __doc__ = students_t.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(StudentsTModel, self).__init__(students_t, **kwargs)

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.params['%samplitude' % self.prefix].value = amp
        self.params['%scenter' % self.prefix].value = cen
        self.params['%ssigma' % self.prefix].value = sig
        self.has_initial_guess = True

class BrietWignerModel(Model):
    __doc__ = breit_wigner.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(BreitWignerModel, self).__init__(breit_wigner, **kwargs)

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.params['%samplitude' % self.prefix].value = amp
        self.params['%scenter' % self.prefix].value = cen
        self.params['%ssigma' % self.prefix].value = sig
        self.params['%sq' % self.prefix].value = 1.0
        self.has_initial_guess = True

class DampedOscillatorModel(Model):
    __doc__ = damped_oscillator.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(DampedOscillatorModel, self).__init__(damped_oscillator, **kwargs)

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.params['%samplitude' % self.prefix].value = amp
        self.params['%scenter' % self.prefix].value = cen
        self.params['%ssigma' % self.prefix].value = sig
        self.has_initial_guess = True

class ExponentialGaussianModel(Model):
    __doc__ = expgaussian.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(ExponentialGaussianModel, self).__init__(expgaussian, **kwargs)

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.params['%samplitude' % self.prefix].value = amp
        self.params['%scenter' % self.prefix].value = cen
        self.params['%ssigma' % self.prefix].value = sig
        self.has_initial_guess = True


class DonaichModel(Model):
    __doc__ = donaich.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(DonaichModel, self).__init__(donaich, **kwargs)

    def guess_starting_values(self, data, x=None, negative=False, **kwargs):
        amp, cen, sig = estimate_peak(data, x, negative)
        self.params['%samplitude' % self.prefix].value = amp
        self.params['%scenter' % self.prefix].value = cen
        self.params['%ssigma' % self.prefix].value = sig
        self.has_initial_guess = True


class PowerLawModel(Model):
    __doc__ = powerlaw.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(PowerLawModel, self).__init__(powerlaw, **kwargs)

    def guess_starting_values(self, data, x=None, **kws):
        try:
            expon, amp = np.polyfit(log(x+1.e-14), log(data+1.e-14), 1)
        except:
            expon, amp = 1, np.log(abs(max(data)+1.e-9))
        self.params['%samplitude' % self.prefix].value = np.exp(amp)
        self.params['%sexponent' % self.prefix].value = expon
        self.has_initial_guess = True

class ExponentialModel(Model):
    __doc__ = exponential.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(ExponentialModel, self).__init__(exponential, **kwargs)

    def guess_starting_values(self, data, x=None, **kws):
        try:
            sval, oval = np.polyfit(x, np.log(abs(data)+1.e-15), 1)
        except:
            sval, oval = 1., np.log(abs(max(data)+1.e-9))
        self.params['%samplitude' % self.prefix].value = np.exp(oval)
        self.params['%sdecay' % self.prefix].value = -1/sval
        self.has_initial_guess = True

class StepModel(Model):
    __doc__ = step.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(StepModel, self).__init__(step, **kwargs)

    def guess_starting_values(self, data, x=None, **kws):
        if x is None:
            return
        ymin, ymax = min(data), max(data)
        xmin, xmax = min(x), max(x)
        self.set_paramval('amplitude', (ymax-ymin))
        self.set_paramval('center',    (xmax+xmin)/2.0)
        self.set_paramval('sigma',     (xmax-xmin)/7.0)
        self.has_initial_guess = True

class RectangleModel(Model):
    __doc__ = rectangle.__doc__ + COMMON_DOC
    def __init__(self, **kwargs):
        super(RectangleModel, self).__init__(rectangle, **kwargs)
        self.params.add('%smidpoint' % self.prefix,
                        expr='(%scenter1+%scenter2)/2.0' % (self.prefix,
                                                            self.prefix))
    def guess_starting_values(self, data, x=None, **kws):
        if x is None:
            return
        ymin, ymax = min(data), max(data)
        xmin, xmax = min(x), max(x)
        self.set_paramval('amplitude', (ymax-ymin))
        self.set_paramval('center1',   (xmax+xmin)/4.0)
        self.set_paramval('sigma1' ,   (xmax-xmin)/7.0)
        self.set_paramval('center2', 3*(xmax+xmin)/4.0)
        self.set_paramval('sigma2',    (xmax-xmin)/7.0)
        self.has_initial_guess = True

