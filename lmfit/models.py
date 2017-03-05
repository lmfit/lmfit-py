"""TODO: module docstring."""
import numpy as np

from . import lineshapes
from .asteval import Interpreter
from .astutils import get_ast_names
from .lineshapes import (breit_wigner, damped_oscillator, dho, donaich,
                         expgaussian, exponential, gaussian, linear, logistic,
                         lognormal, lorentzian, moffat, parabolic, pearson7,
                         powerlaw, pvoigt, rectangle, skewed_gaussian,
                         skewed_voigt, step, students_t, voigt)
from .model import Model


class DimensionalError(Exception):
    """TODO: class docstring."""

    pass


def _validate_1d(independent_vars):
    if len(independent_vars) != 1:
        raise DimensionalError(
            "This model requires exactly one independent variable.")


def index_of(arr, val):
    """Return index of array nearest to a value."""
    if val < min(arr):
        return 0
    return np.abs(arr-val).argmin()


def fwhm_expr(model):
    """Return constraint expression for fwhm."""
    fmt = "{factor:.7f}*{prefix:s}sigma"
    return fmt.format(factor=model.fwhm_factor, prefix=model.prefix)


def height_expr(model):
    """Return constraint expression for maximum peak height."""
    fmt = "{factor:.7f}*{prefix:s}amplitude/max(1.e-15, {prefix:s}sigma)"
    return fmt.format(factor=model.height_factor, prefix=model.prefix)


def guess_from_peak(model, y, x, negative, ampscale=1.0, sigscale=1.0):
    """Estimate amp, cen, sigma for a peak, create params."""
    if x is None:
        return 1.0, 0.0, 1.0
    maxy, miny = max(y), min(y)
    maxx, minx = max(x), min(x)
    imaxy = index_of(y, maxy)
    cen = x[imaxy]
    amp = (maxy - miny)*3.0
    sig = (maxx-minx)/6.0

    halfmax_vals = np.where(y > (maxy+miny)/2.0)[0]
    if negative:
        imaxy = index_of(y, miny)
        amp = -(maxy - miny)*3.0
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
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = "%s%s" % (prefix, key)
        if pname in pars:
            pars[pname].value = val
    return pars


COMMON_INIT_DOC = """
    Parameters
    ----------
    independent_vars: ['x']
        arguments to func that are independent variables
    prefix: ``None`` or string
       string to prepend to paramter names, needed to add two Models that
       have parameter names in common.
    missing:  ``None`` or string
        how to handle `nan` and missing values in data. One of:

        - 'none' or ``None``: Do not check for null or missing values (default)

        - 'drop': Drop null or missing observations in data. if pandas is
          installed, `pandas.isnull` is used, otherwise `numpy.isnan` is used.

        - 'raise': Raise a (more helpful) exception when data contains null
          or missing values.
    kwargs : optional
        keyword arguments to pass to :class:`Model`.
"""

COMMON_GUESS_DOC = """Guess starting values for the parameters of a model.

    Parameters
    ----------
    data : array-like
        array of data to use to guess parameter values.
    kws : additional keyword arguments, passed to model function.

    Returns
    -------
    params  : Parameters
"""

COMMON_DOC = COMMON_INIT_DOC

class ConstantModel(Model):
    """Constant model, with a single Parameter: ``c``

    Note that this is 'constant' in the sense of having no dependence on
    the independent variable ``x``, not in the sense of being non-varying.
    To be clear, ``c`` will be a Parameter that will be varied in the
    fit (by default, of course).
    """
    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        def constant(x, c):
            return c
        super(ConstantModel, self).__init__(constant, **kwargs)

    def guess(self, data, **kwargs):
        pars = self.make_params()
        pars['%sc' % self.prefix].set(value=data.mean())
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__    = COMMON_GUESS_DOC

class ComplexConstantModel(Model):
    """Complex constant model, with wo Parameters:
    ``re``, and ``im``.

    Note that ``re`` and ``im`` are 'constant' in the sense of having no
    dependence on the independent variable ``x``, not in the sense of
    being non-varying. To be clear, ``re`` and ``im`` will be Parameters
    that will be varied in the fit (by default, of course).
    """
    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 name=None,  **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        def constant(x, re, im):
            return re + 1j*im
        super(ComplexConstantModel, self).__init__(constant, **kwargs)

    def guess(self, data, **kwargs):
        pars = self.make_params()
        pars['%sre' % self.prefix].set(value=data.real.mean())
        pars['%sim' % self.prefix].set(value=data.imag.mean())
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__    = COMMON_GUESS_DOC

class LinearModel(Model):
    """Linear model, with two Parameters
    ``intercept`` and ``slope``, defined as

    .. math::

        f(x; m, b) = m x + b

    with  ``slope`` for :math:`m` and  ``intercept`` for :math:`b`.
    """
    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 name=None,  **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        super(LinearModel, self).__init__(linear, **kwargs)

    def guess(self, data, x=None, **kwargs):
        sval, oval = 0., 0.
        if x is not None:
            sval, oval = np.polyfit(x, data, 1)
        pars = self.make_params(intercept=oval, slope=sval)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__    = COMMON_GUESS_DOC


class QuadraticModel(Model):
    """A quadratic model, with three Parameters
    ``a``, ``b``, and ``c``, defined as

    .. math::

        f(x; a, b, c) = a x^2 + b x + c
    """
    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 name=None,  **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        super(QuadraticModel, self).__init__(parabolic, **kwargs)

    def guess(self, data, x=None, **kwargs):
        a, b, c = 0., 0., 0.
        if x is not None:
            a, b, c = np.polyfit(x, data, 2)
        pars = self.make_params(a=a, b=b, c=c)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__    = COMMON_GUESS_DOC

ParabolicModel = QuadraticModel


class PolynomialModel(Model):
    """A polynomial model with up to 7 Parameters, specfied by ``degree``.

    .. math::

    f(x; c_0, c_1, \ldots, c_7) = \sum_{i=0, 7} c_i  x^i

    with parameters ``c0``, ``c1``, ..., ``c7``.  The supplied ``degree``
    will specify how many of these are actual variable parameters.  This
    uses :numpydoc:`polyval` for its calculation of the polynomial.
    """
    MAX_DEGREE = 7
    DEGREE_ERR = "degree must be an integer less than %d."
    def __init__(self, degree, independent_vars=['x'], prefix='', missing=None,
                 name=None,  **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        if not isinstance(degree, int) or degree > self.MAX_DEGREE:
            raise TypeError(self.DEGREE_ERR % self.MAX_DEGREE)

        self.poly_degree = degree
        pnames = ['c%i' % (i) for i in range(degree + 1)]
        kwargs['param_names'] = pnames

        def polynomial(x, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0):
            return np.polyval([c7, c6, c5, c4, c3, c2, c1, c0], x)

        super(PolynomialModel, self).__init__(polynomial, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pars = self.make_params()
        if x is not None:
            out = np.polyfit(x, data, self.poly_degree)
            for i, coef in enumerate(out[::-1]):
                pars['%sc%i' % (self.prefix, i)].set(value=coef)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__    = COMMON_GUESS_DOC

class GaussianModel(Model):
    r"""A model based on a Gaussian or normal distribution lineshape
    (see http://en.wikipedia.org/wiki/Normal_distribution), with three Parameters:
    ``amplitude``, ``center``, and ``sigma``.
    In addition, parameters ``fwhm`` and ``height`` are included as constraints
    to report full width at half maximum and maximum peak height, respectively.

    .. math::

        f(x; A, \mu, \sigma) = \frac{A}{\sigma\sqrt{2\pi}} e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}

    where the parameter ``amplitude`` corresponds to :math:`A`, ``center`` to
    :math:`\mu`, and ``sigma`` to :math:`\sigma`.  The full width at
    half maximum is :math:`2\sigma\sqrt{2\ln{2}}`, approximately
    :math:`2.3548\sigma`.
    """
    fwhm_factor = 2.354820
    height_factor = 1./np.sqrt(2*np.pi)
    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 name=None,  **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        super(GaussianModel, self).__init__(gaussian, **kwargs)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))
        self.set_param_hint('height', expr=height_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__    = COMMON_GUESS_DOC

class LorentzianModel(Model):
    r"""A model based on a Lorentzian or Cauchy-Lorentz distribution function
    (see http://en.wikipedia.org/wiki/Cauchy_distribution), with three Parameters:
    ``amplitude``, ``center``, and ``sigma``.
    In addition, parameters ``fwhm`` and ``height`` are included as constraints
    to report full width at half maximum and maximum peak height, respectively.

    .. math::

        f(x; A, \mu, \sigma) = \frac{A}{\pi} \big[\frac{\sigma}{(x - \mu)^2 + \sigma^2}\big]

    where the parameter ``amplitude`` corresponds to :math:`A`, ``center`` to
    :math:`\mu`, and ``sigma`` to :math:`\sigma`.  The full width at
    half maximum is :math:`2\sigma`.
    """
    fwhm_factor = 2.0
    height_factor = 1./np.pi
    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 name=None,  **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        super(LorentzianModel, self).__init__(lorentzian, **kwargs)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))
        self.set_param_hint('height', expr=height_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative, ampscale=1.25)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__    = COMMON_GUESS_DOC


class VoigtModel(Model):
    r"""A model based on a Voigt distribution function (see
    http://en.wikipedia.org/wiki/Voigt_profile>), with four Parameters:
    ``amplitude``, ``center``, ``sigma``, and ``gamma``.  By default,
    ``gamma`` is constrained to have value equal to ``sigma``, though it
    can be varied independently.  In addition, parameters ``fwhm`` and
    ``height`` are included as constraints to report full width at half
    maximum and maximum peak height, respectively.  The definition for the
    Voigt function used here is

    .. math::

        f(x; A, \mu, \sigma, \gamma) = \frac{A \textrm{Re}[w(z)]}{\sigma\sqrt{2 \pi}}

    where

    .. math::
        :nowrap:

        \begin{eqnarray*}
            z &=& \frac{x-\mu +i\gamma}{\sigma\sqrt{2}} \\
            w(z) &=& e^{-z^2}{\operatorname{erfc}}(-iz)
        \end{eqnarray*}

    and :func:`erfc` is the complimentary error function.  As above,
    ``amplitude`` corresponds to :math:`A`, ``center`` to
    :math:`\mu`, and ``sigma`` to :math:`\sigma`. The parameter ``gamma``
    corresponds  to :math:`\gamma`.
    If ``gamma`` is kept at the default value (constrained to ``sigma``),
    the full width at half maximum is approximately :math:`3.6013\sigma`.

    """
    fwhm_factor = 3.60131
    height_factor = 1./np.sqrt(2*np.pi)
    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 name=None,  **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        super(VoigtModel, self).__init__(voigt, **kwargs)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('gamma', expr='%ssigma' % self.prefix)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))
        self.set_param_hint('height', expr=height_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative,
                               ampscale=1.5, sigscale=0.65)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__    = COMMON_GUESS_DOC


class PseudoVoigtModel(Model):
    r"""A model based on a pseudo-Voigt distribution function
    (see http://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_Approximation),
    which is a weighted sum of a Gaussian and Lorentzian distribution functions
    with that share values for ``amplitude`` (:math:`A`), ``center`` (:math:`\mu`)
    and full width at half maximum (and so have  constrained values of
    ``sigma`` (:math:`\sigma`).  A parameter ``fraction`` (:math:`\alpha`)
    controls the relative weight of the Gaussian and Lorentzian components,
    giving the full definition of

    .. math::

        f(x; A, \mu, \sigma, \alpha) = \frac{(1-\alpha)A}{\sigma_g\sqrt{2\pi}}
        e^{[{-{(x-\mu)^2}/{{2\sigma_g}^2}}]}
        + \frac{\alpha A}{\pi} \big[\frac{\sigma}{(x - \mu)^2 + \sigma^2}\big]

    where :math:`\sigma_g = {\sigma}/{\sqrt{2\ln{2}}}` so that the full width
    at half maximum of each component and of the sum is :math:`2\sigma`. The
    :meth:`guess` function always sets the starting value for ``fraction`` at 0.5.
    """

    fwhm_factor = 2.0

    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 name=None,  **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        super(PseudoVoigtModel, self).__init__(pvoigt, **kwargs)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('fraction', value=0.5)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))

    def guess(self, data, x=None, negative=False, **kwargs):
        pars = guess_from_peak(self, data, x, negative, ampscale=1.25)
        pars['%sfraction' % self.prefix].set(value=0.5)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__    = COMMON_GUESS_DOC


class MoffatModel(Model):
    __doc__ = moffat.__doc__ + COMMON_DOC if moffat.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(MoffatModel, self).__init__(moffat, *args, **kwargs)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('beta')
        self.set_param_hint('fwhm', expr="2*%ssigma*sqrt(2**(1.0/%sbeta)-1)" % (self.prefix, self.prefix))

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5, sigscale=1.)
        return update_param_vals(pars, self.prefix, **kwargs)


class Pearson7Model(Model):
    __doc__ = pearson7.__doc__ + COMMON_DOC if pearson7.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(Pearson7Model, self).__init__(pearson7, *args, **kwargs)
        self.set_param_hint('expon', value=1.5)

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = guess_from_peak(self, data, x, negative)
        pars['%sexpon' % self.prefix].set(value=1.5)
        return update_param_vals(pars, self.prefix, **kwargs)


class StudentsTModel(Model):
    __doc__ = students_t.__doc__ + COMMON_DOC if students_t.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(StudentsTModel, self).__init__(students_t, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)


class BreitWignerModel(Model):
    __doc__ = breit_wigner.__doc__ + COMMON_DOC if breit_wigner.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(BreitWignerModel, self).__init__(breit_wigner, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = guess_from_peak(self, data, x, negative)
        pars['%sq' % self.prefix].set(value=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class LognormalModel(Model):
    __doc__ = lognormal.__doc__ + COMMON_DOC if lognormal.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(LognormalModel, self).__init__(lognormal, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = self.make_params(amplitude=1.0, center=0.0, sigma=0.25)
        pars['%ssigma' % self.prefix].set(min=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class DampedOscillatorModel(Model):
    __doc__ = damped_oscillator.__doc__ + COMMON_DOC if damped_oscillator.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(DampedOscillatorModel, self).__init__(damped_oscillator, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = guess_from_peak(self, data, x, negative,
                               ampscale=0.1, sigscale=0.1)
        return update_param_vals(pars, self.prefix, **kwargs)


class DampedHarmonicOscillatorModel(Model):
    __doc__ = dho.__doc__ + COMMON_DOC if dho.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(DampedOscillatorModel, self).__init__(dho, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = guess_from_peak(self, data, x, negative,
                               ampscale=0.1, sigscale=0.1)
        pars['%sgamma' % self.prefix].set(value=1.0, min=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)


class ExponentialGaussianModel(Model):
    __doc__ = expgaussian.__doc__ + COMMON_DOC if expgaussian.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(ExponentialGaussianModel, self).__init__(expgaussian, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)


class SkewedGaussianModel(Model):
    __doc__ = skewed_gaussian.__doc__ + COMMON_DOC if skewed_gaussian.__doc__ else ""
    fwhm_factor = 2.354820

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(SkewedGaussianModel, self).__init__(skewed_gaussian, *args, **kwargs)
        self.set_param_hint('sigma', min=0)

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)


class DonaichModel(Model):
    __doc__ = donaich.__doc__ + COMMON_DOC if donaich.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(DonaichModel, self).__init__(donaich, *args, **kwargs)

    def guess(self, data, x=None, negative=False, **kwargs):
        """TODO: docstring in public method."""
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5)
        return update_param_vals(pars, self.prefix, **kwargs)


class PowerLawModel(Model):
    __doc__ = powerlaw.__doc__ + COMMON_DOC if powerlaw.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(PowerLawModel, self).__init__(powerlaw, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """TODO: docstring in public method."""
        try:
            expon, amp = np.polyfit(np.log(x+1.e-14), np.log(data+1.e-14), 1)
        except:
            expon, amp = 1, np.log(abs(max(data)+1.e-9))

        pars = self.make_params(amplitude=np.exp(amp), exponent=expon)
        return update_param_vals(pars, self.prefix, **kwargs)


class ExponentialModel(Model):
    __doc__ = exponential.__doc__ + COMMON_DOC if exponential.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(ExponentialModel, self).__init__(exponential, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """TODO: docstring in public method."""
        try:
            sval, oval = np.polyfit(x, np.log(abs(data)+1.e-15), 1)
        except:
            sval, oval = 1., np.log(abs(max(data)+1.e-9))
        pars = self.make_params(amplitude=np.exp(oval), decay=-1.0/sval)
        return update_param_vals(pars, self.prefix, **kwargs)


class StepModel(Model):
    __doc__ = step.__doc__ + COMMON_DOC if step.__doc__ else ""

    def __init__(self, *args, **kwargs):
        """TODO: docstring in public method."""
        super(StepModel, self).__init__(step, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """TODO: docstring in public method."""
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
        """TODO: docstring in public method."""
        super(RectangleModel, self).__init__(rectangle, *args, **kwargs)

        self.set_param_hint('center1')
        self.set_param_hint('center2')
        self.set_param_hint('midpoint',
                            expr='(%scenter1+%scenter2)/2.0' % (self.prefix,
                                                                self.prefix))

    def guess(self, data, x=None, **kwargs):
        """TODO: docstring in public method."""
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
    """Model from User-supplied expression.

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

    idvar_missing = "No independent variable found in\n %s"
    idvar_notfound = "Cannot find independent variables '%s' in\n %s"
    no_prefix = "ExpressionModel does not support `prefix` argument"

    def __init__(self, expr, independent_vars=None, init_script=None,
                 *args, **kwargs):
        """TODO: docstring in public method."""
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
        """TODO: docstring in magic method."""
        return "<lmfit.ExpressionModel('%s')>" % (self.expr)

    def _parse_params(self):
        """Over-write ExpressionModel._parse_params with `pass`.

        This prevents normal parsing of function for parameter names.

        """
        pass
