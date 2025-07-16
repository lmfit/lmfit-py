"""Module containing built-in fitting models."""
import inspect

from asteval import Interpreter, get_ast_names
import numpy as np
from scipy.interpolate import splev, splrep

from . import lineshapes
from .lineshapes import (bose, breit_wigner, damped_oscillator, dho, doniach,
                         expgaussian, exponential, fermi, gaussian, gaussian2d,
                         linear, lognormal, lorentzian, moffat, parabolic,
                         pearson4, pearson7, powerlaw, pvoigt, rectangle, sine,
                         skewed_gaussian, skewed_voigt, split_lorentzian, step,
                         students_t, thermal_distribution, tiny, voigt)
from .model import Model

tau = 2.0 * np.pi


class DimensionalError(Exception):
    """Raise exception when number of independent variables is not one."""


def _validate_1d(independent_vars):
    if len(independent_vars) != 1:
        raise DimensionalError(
            "This model requires exactly one independent variable.")


def fwhm_expr(model):
    """Return constraint expression for fwhm."""
    fmt = "{factor:.7f}*{prefix:s}sigma"
    return fmt.format(factor=model.fwhm_factor, prefix=model.prefix)


def height_expr(model):
    """Return constraint expression for maximum peak height."""
    fmt = "{factor:.7f}*{prefix:s}amplitude/max({}, {prefix:s}sigma)"
    return fmt.format(tiny, factor=model.height_factor, prefix=model.prefix)


def guess_from_peak(model, y, x, negative, ampscale=1.0, sigscale=1.0):
    """Estimate starting values from 1D peak data and create Parameters."""
    sort_increasing = np.argsort(x)
    x = x[sort_increasing]
    y = y[sort_increasing]

    maxy, miny = max(y), min(y)
    maxx, minx = max(x), min(x)
    cen = x[np.argmax(y)]
    height = (maxy - miny)*3.0
    sig = (maxx-minx)/6.0

    # the explicit conversion to a NumPy array is to make sure that the
    # indexing on line 65 also works if the data is supplied as pandas.Series
    x_halfmax = np.array(x[y > (maxy+miny)/2.0])
    if negative:
        height = -(maxy - miny)*3.0
        x_halfmax = x[y < (maxy+miny)/2.0]
    if len(x_halfmax) > 2:
        sig = (x_halfmax[-1] - x_halfmax[0])/2.0
        cen = x_halfmax.mean()
    amp = height*sig*ampscale
    sig = sig*sigscale

    pars = model.make_params(amplitude=amp, center=cen, sigma=sig)
    pars[f'{model.prefix}sigma'].set(min=0.0)
    return pars


def guess_from_peak2d(model, z, x, y, negative):
    """Estimate starting values from 2D peak data and create Parameters."""
    maxx, minx = max(x), min(x)
    maxy, miny = max(y), min(y)
    maxz, minz = max(z), min(z)

    centerx = x[np.argmax(z)]
    centery = y[np.argmax(z)]
    height = (maxz - minz)
    sigmax = (maxx-minx)/6.0
    sigmay = (maxy-miny)/6.0

    if negative:
        centerx = x[np.argmin(z)]
        centery = y[np.argmin(z)]
        height = (minz - maxz)

    amp = height*sigmax*sigmay

    pars = model.make_params(amplitude=amp, centerx=centerx, centery=centery,
                             sigmax=sigmax, sigmay=sigmay)
    pars[f'{model.prefix}sigmax'].set(min=0.0)
    pars[f'{model.prefix}sigmay'].set(min=0.0)
    return pars


def guess_thermal(model, y, x):
    "guess params for thermal distribution"
    center = np.mean(x)
    kt = (max(x) - min(x))/10
    amplitude = y.max() * np.exp((x.min() - center)/max(1.e-15, kt))
    return model.make_params(amplitude=amplitude, center=center, kt=kt)


def update_param_vals(pars, prefix, **kwargs):
    """Update parameter values with keyword arguments."""
    for key, val in kwargs.items():
        pname = f"{prefix}{key}"
        if pname in pars:
            pars[pname].value = val
    pars.update_constraints()
    return pars


COMMON_INIT_DOC = """
    Parameters
    ----------
    independent_vars : :obj:`list` of :obj:`str`, optional
        Arguments to the model function that are independent variables
        default is ['x']).
    prefix : str, optional
        String to prepend to parameter names, needed to add two Models
        that have parameter names in common.
    nan_policy : {'raise', 'propagate', 'omit'}, optional
        How to handle NaN and missing values in data. See Notes below.
    **kwargs : optional
        Keyword arguments to pass to :class:`Model`.

    Notes
    -----
    1. `nan_policy` sets what to do when a NaN or missing value is seen in
    the data. Should be one of:

        - `'raise'` : raise a `ValueError` (default)
        - `'propagate'` : do nothing
        - `'omit'` : drop missing data

    """

COMMON_GUESS_DOC = """Guess starting values for the parameters of a model.

    Parameters
    ----------
    data : array_like
        Array of data (i.e., y-values) to use to guess parameter values.
    x : array_like
        Array of values for the independent variable (i.e., x-values).
    **kws : optional
        Additional keyword arguments, passed to model function.

    Returns
    -------
    params : Parameters
        Initial, guessed values for the parameters of a Model.

    .. versionchanged:: 1.0.3
       Argument ``x`` is now explicitly required to estimate starting values.

    """


class ConstantModel(Model):
    """Constant model, with a single Parameter: `c`.

    Note that this is 'constant' in the sense of having no dependence on
    the independent variable `x`, not in the sense of being non-varying.
    To be clear, `c` will be a Parameter that will be varied in the fit
    (by default, of course).

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def constant(x, c=0.0):
            return c * np.ones(np.shape(x))
        super().__init__(constant, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()

        pars[f'{self.prefix}c'].set(value=data.mean())
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class ComplexConstantModel(Model):
    """Complex constant model, with two Parameters: `re` and `im`.

    Note that `re` and `im` are 'constant' in the sense of having no
    dependence on the independent variable `x`, not in the sense of being
    non-varying. To be clear, `re` and `im` will be Parameters that will
    be varied in the fit (by default, of course).

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        def constant(x, re=0., im=0.):
            return (re + 1j*im) * np.ones(np.shape(x))
        super().__init__(constant, **kwargs)

    def guess(self, data, x=None, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()
        pars[f'{self.prefix}re'].set(value=np.real(data).mean())
        pars[f'{self.prefix}im'].set(value=np.imag(data).mean())
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class LinearModel(Model):
    """Linear model, with two Parameters: `intercept` and `slope`.

    Defined as:

    .. math::

        f(x; m, b) = m x + b

    with `slope` for :math:`m` and `intercept` for :math:`b`.

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(linear, **kwargs)

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from data."""
        sval, oval = np.polyfit(x, data, 1)
        pars = self.make_params(intercept=oval, slope=sval)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class QuadraticModel(Model):
    """A quadratic model, with three Parameters: `a`, `b`, and `c`.

    Defined as:

    .. math::

        f(x; a, b, c) = a x^2 + b x + c

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(parabolic, **kwargs)

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from data."""
        a, b, c = np.polyfit(x, data, 2)
        pars = self.make_params(a=a, b=b, c=c)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


ParabolicModel = QuadraticModel


class PolynomialModel(Model):
    r"""A polynomial model with up to 7 Parameters, specified by `degree`.

    .. math::

        f(x; c_0, c_1, \ldots, c_7) = \sum_{i=0, 7} c_i x^i

    with parameters `c0`, `c1`, ..., `c7`. The supplied `degree` will
    specify how many of these are actual variable parameters. This uses
    :numpydoc:`polyval` for its calculation of the polynomial.

    """

    MAX_DEGREE = 7
    DEGREE_ERR = f"degree must be an integer equal to or smaller than {MAX_DEGREE}."

    valid_forms = (0, 1, 2, 3, 4, 5, 6, 7)

    def __init__(self, degree=7, independent_vars=['x'], prefix='',
                 nan_policy='raise', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        if 'form' in kwargs:
            degree = int(kwargs.pop('form'))
        if not isinstance(degree, int) or degree > self.MAX_DEGREE:
            raise TypeError(self.DEGREE_ERR)

        self.poly_degree = degree
        pnames = [f'c{i}' for i in range(degree + 1)]
        kwargs['param_names'] = pnames

        def polynomial(x, c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0):
            return np.polyval([c7, c6, c5, c4, c3, c2, c1, c0], x)

        super().__init__(polynomial, **kwargs)

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()
        out = np.polyfit(x, data, self.poly_degree)
        for i, coef in enumerate(out[::-1]):
            pars[f'{self.prefix}c{i}'].set(value=coef)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class SplineModel(Model):
    r"""A 1-D cubic spline model with a variable number of `knots` and
    parameters `s0`, `s1`, ..., `sN`, for `N` knots.

    The user must supply a list or ndarray `xknots`: the `x` values for the
    'knots' which control the flexibility of the spline function.

    The parameters `s0`, ..., `sN` (where `N` is the size of `xknots`) will
    correspond to the `y` values for the spline knots at the `x=xknots`
    positions where the highest order derivative will be discontinuous.
    The resulting curve will not necessarily pass through these knot
    points, but for finely-spaced knots, the spline parameter values will
    be very close to the `y` values of the resulting curve.

    The maximum number of knots supported is 300.

    Using the `guess()` method to initialize parameter values is highly
    recommended.

    Parameters
    ----------
    xknots : :obj:`list` of floats or :obj:`ndarray`, required
        x-values of knots for spline.
    independent_vars : :obj:`list` of :obj:`str`, optional
        Arguments to the model function that are independent variables
        default is ['x']).
    prefix : str, optional
        String to prepend to parameter names, needed to add two Models
        that have parameter names in common.
    nan_policy : {'raise', 'propagate', 'omit'}, optional
        How to handle NaN and missing values in data. See Notes below.

    Notes
    -----
    1.  There must be at least 4 knot points, and not more than 300.

    2. `nan_policy` sets what to do when a NaN or missing value is seen in
          the data. Should be one of:

        - `'raise'` : raise a `ValueError` (default)
        - `'propagate'` : do nothing
        - `'omit'` : drop missing data

    """

    MAX_KNOTS = 100
    NKNOTS_MAX_ERR = f"SplineModel supports up to {MAX_KNOTS:d} knots"
    NKNOTS_NDARRY_ERR = "SplineModel xknots must be 1-D array-like"
    DIM_ERR = "SplineModel supports only 1-d spline interpolation"

    def __init__(self, xknots, independent_vars=['x'], prefix='',
                 nan_policy='raise', **kwargs):
        """ """
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})

        if isinstance(xknots, (list, tuple)):
            xknots = np.asarray(xknots, dtype=np.float64)
        try:
            xknots = xknots.flatten()
        except Exception:
            raise TypeError(self.NKNOTS_NDARRAY_ERR)

        if len(xknots) > self.MAX_KNOTS:
            raise TypeError(self.NKNOTS_MAX_ERR)

        if len(independent_vars) > 1:
            raise TypeError(self.DIM_ERR)

        self.xknots = xknots
        self.nknots = len(xknots)
        self.order = 3   # cubic splines only

        def spline_model(x, s0=1, s1=1, s2=1, s3=1, s4=1, s5=1, s6=1, s7=1,
                         s8=1, s9=1, s10=1, s11=1, s12=1, s13=1, s14=1, s15=1,
                         s16=1, s17=1, s18=1, s19=1, s20=1, s21=1, s22=1, s23=1,
                         s24=1, s25=1, s26=1, s27=1, s28=1, s29=1, s30=1, s31=1,
                         s32=1, s33=1, s34=1, s35=1, s36=1, s37=1, s38=1, s39=1,
                         s40=1, s41=1, s42=1, s43=1, s44=1, s45=1, s46=1, s47=1,
                         s48=1, s49=1, s50=1, s51=1, s52=1, s53=1, s54=1, s55=1,
                         s56=1, s57=1, s58=1, s59=1, s60=1, s61=1, s62=1, s63=1,
                         s64=1, s65=1, s66=1, s67=1, s68=1, s69=1, s70=1, s71=1,
                         s72=1, s73=1, s74=1, s75=1, s76=1, s77=1, s78=1, s79=1,
                         s80=1, s81=1, s82=1, s83=1, s84=1, s85=1, s86=1, s87=1,
                         s88=1, s89=1, s90=1, s91=1, s92=1, s93=1, s94=1, s95=1,
                         s96=1, s97=1, s98=1, s99=1, knots=None, order=None):
            """spline evaluation"""
            coefs = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
                     s12, s13, s14, s15, s16, s17, s18, s19, s20, s21,
                     s22, s23, s24, s25, s26, s27, s28, s29, s30, s31,
                     s32, s33, s34, s35, s36, s37, s38, s39, s40, s41,
                     s42, s43, s44, s45, s46, s47, s48, s49, s50, s51,
                     s52, s53, s54, s55, s56, s57, s58, s59, s60, s61,
                     s62, s63, s64, s65, s66, s67, s68, s69, s70, s71,
                     s72, s73, s74, s75, s76, s77, s78, s79, s80, s81,
                     s82, s83, s84, s85, s86, s87, s88, s89, s90, s91,
                     s92, s93, s94, s95, s96, s97, s98, s99]
            if knots is None:
                knots = self.knots
            if order is None:
                order = self.order
            coefs = coefs[:len(knots)]
            coefs.extend([coefs[-1]]*(order+1))
            return splev(x, [knots, np.array(coefs), order])

        super().__init__(spline_model, **kwargs)

        if 'x' not in independent_vars:
            self.independent_vars.pop('x')

        self._param_root_names = [f's{d}' for d in range(self.nknots)]
        self._param_names = [f'{prefix}{s}' for s in self._param_root_names]
        self.knots, _c, _k = splrep(self.xknots, np.ones(self.nknots), k=self.order)

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params()

        for i, xk in enumerate(self.xknots):
            ix = np.abs(x-xk).argmin()
            this = data[ix]
            pone = data[ix+1] if ix < len(x)-2 else this
            mone = data[ix-1] if ix > 0 else this
            pars[f'{self.prefix}s{i}'].value = (4.*this + pone + mone)/6.

        return update_param_vals(pars, self.prefix, **kwargs)

    guess.__doc__ = COMMON_GUESS_DOC


class SineModel(Model):
    r"""A model based on a sinusoidal lineshape.

    The model has three Parameters: `amplitude`, `frequency`, and `shift`.

    .. math::

        f(x; A, \phi, f) = A \sin (f x + \phi)

    where the parameter `amplitude` corresponds to :math:`A`, `frequency` to
    :math:`f`, and `shift` to :math:`\phi`. All are constrained to be
    non-negative, and `shift` additionally to be smaller than :math:`2\pi`.

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(sine, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('amplitude', min=0)
        self.set_param_hint('frequency', min=0)
        self.set_param_hint('shift', min=-tau-1.e-5, max=tau+1.e-5)

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from the FFT of the data."""
        data = data - data.mean()
        # assume uniform spacing
        frequencies = np.fft.fftfreq(len(x), abs(x[-1] - x[0]) / (len(x) - 1))
        fft = abs(np.fft.fft(data))
        argmax = abs(fft).argmax()
        amplitude = 2.0 * fft[argmax] / len(fft)
        frequency = tau * abs(frequencies[argmax])
        # try shifts in the range [0, 2*pi) and take the one with best residual
        shift_guesses = np.linspace(0, tau, 11, endpoint=False)
        errors = [np.linalg.norm(self.eval(x=x, amplitude=amplitude,
                                           frequency=frequency,
                                           shift=shift_guess) - data)
                  for shift_guess in shift_guesses]
        shift = shift_guesses[np.argmin(errors)]
        pars = self.make_params(amplitude=amplitude, frequency=frequency,
                                shift=shift)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class GaussianModel(Model):
    r"""A model based on a Gaussian or normal distribution lineshape.

    The model has three Parameters: `amplitude`, `center`, and `sigma`.
    In addition, parameters `fwhm` and `height` are included as
    constraints to report full width at half maximum and maximum peak
    height, respectively.

    .. math::

        f(x; A, \mu, \sigma) = \frac{A}{\sigma\sqrt{2\pi}} e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}

    where the parameter `amplitude` corresponds to :math:`A`, `center` to
    :math:`\mu`, and `sigma` to :math:`\sigma`. The full width at half
    maximum is :math:`2\sigma\sqrt{2\ln{2}}`, approximately
    :math:`2.3548\sigma`.

    For more information, see: https://en.wikipedia.org/wiki/Normal_distribution

    """

    fwhm_factor = 2*np.sqrt(2*np.log(2))
    height_factor = 1./np.sqrt(2*np.pi)

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(gaussian, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))
        self.set_param_hint('height', expr=height_expr(self))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class Gaussian2dModel(Model):
    r"""A model based on a two-dimensional Gaussian function.

    The model has two independent variables `x` and `y` and five
    Parameters: `amplitude`, `centerx`, `sigmax`, `centery`, and `sigmay`.
    In addition, parameters `fwhmx`, `fwhmy`, and `height` are included as
    constraints to report the maximum peak height and the two full width
    at half maxima, respectively.

    .. math::

        f(x, y; A, \mu_x, \sigma_x, \mu_y, \sigma_y) =
        A g(x; A=1, \mu_x, \sigma_x) g(y; A=1, \mu_y, \sigma_y)

    where subfunction :math:`g(x; A, \mu, \sigma)` is a Gaussian lineshape:

    .. math::

        g(x; A, \mu, \sigma) =
        \frac{A}{\sigma\sqrt{2\pi}} e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}.

    """

    fwhm_factor = 2*np.sqrt(2*np.log(2))
    height_factor = 1./(2*np.pi)

    def __init__(self, independent_vars=['x', 'y'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(gaussian2d, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigmax', min=0)
        self.set_param_hint('sigmay', min=0)
        expr = fwhm_expr(self)
        self.set_param_hint('fwhmx', expr=expr.replace('sigma', 'sigmax'))
        self.set_param_hint('fwhmy', expr=expr.replace('sigma', 'sigmay'))
        fmt = ("{factor:.7f}*{prefix:s}amplitude/(max({tiny}, {prefix:s}sigmax)"
               + "*max({tiny}, {prefix:s}sigmay))")
        expr = fmt.format(tiny=tiny, factor=self.height_factor, prefix=self.prefix)
        self.set_param_hint('height', expr=expr)

    def guess(self, data, x, y, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak2d(self, data, x, y, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC.replace("['x']", "['x', 'y']")
    guess.__doc__ = COMMON_GUESS_DOC


class LorentzianModel(Model):
    r"""A model based on a Lorentzian or Cauchy-Lorentz distribution function.

    The model has three Parameters: `amplitude`, `center`, and `sigma`. In
    addition, parameters `fwhm` and `height` are included as constraints
    to report full width at half maximum and maximum peak height,
    respectively.

    .. math::

        f(x; A, \mu, \sigma) = \frac{A}{\pi} \big[\frac{\sigma}{(x - \mu)^2 + \sigma^2}\big]

    where the parameter `amplitude` corresponds to :math:`A`, `center` to
    :math:`\mu`, and `sigma` to :math:`\sigma`. The full width at half
    maximum is :math:`2\sigma`.

    For more information, see:
    https://en.wikipedia.org/wiki/Cauchy_distribution

    """

    fwhm_factor = 2.0
    height_factor = 1./np.pi

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(lorentzian, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))
        self.set_param_hint('height', expr=height_expr(self))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative, ampscale=1.25)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class SplitLorentzianModel(Model):
    r"""A model based on a Lorentzian or Cauchy-Lorentz distribution function.

    The model has four parameters: `amplitude`, `center`, `sigma`, and
    `sigma_r`. In addition, parameters `fwhm` and `height` are included
    as constraints to report full width at half maximum and maximum peak
    height, respectively.

    'Split' means that the width of the distribution is different between
    left and right slopes.

    .. math::

        f(x; A, \mu, \sigma, \sigma_r) = \frac{2 A}{\pi (\sigma+\sigma_r)} \big[\frac{\sigma^2}{(x - \mu)^2 + \sigma^2} * H(\mu-x) + \frac{\sigma_r^2}{(x - \mu)^2 + \sigma_r^2} * H(x-\mu)\big]

    where the parameter `amplitude` corresponds to :math:`A`, `center` to
    :math:`\mu`, `sigma` to :math:`\sigma`, `sigma_l` to :math:`\sigma_l`,
    and :math:`H(x)` is a Heaviside step function:

    .. math::

        H(x) = 0 | x < 0, 1 | x \geq 0

    The full width at half maximum is :math:`\sigma_l+\sigma_r`. Just as
    with the Lorentzian model, integral of this function from `-.inf` to
    `+.inf` equals to `amplitude`.

    For more information, see:
    https://en.wikipedia.org/wiki/Cauchy_distribution

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(split_lorentzian, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        fwhm_expr = '{pre:s}sigma+{pre:s}sigma_r'
        height_expr = '2*{pre:s}amplitude/{0:.7f}/max({1:.7f}, ({pre:s}sigma+{pre:s}sigma_r))'
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('sigma_r', min=0)
        self.set_param_hint('fwhm', expr=fwhm_expr.format(pre=self.prefix))
        self.set_param_hint('height', expr=height_expr.format(np.pi, tiny, pre=self.prefix))

#     def post_fit(self, result):
#         fwhm_expr = '{pre:s}sigma+{pre:s}sigma_r'
#         height_expr = '2*{pre:s}amplitude/{0:.7f}/max({1:.7f}, ({pre:s}sigma+{pre:s}sigma_r))'
#         addpar = result.params.add
#         prefix = self.prefix
#         addpar(name=f'{prefix}fwhm', expr=fwhm_expr.format(pre=prefix))
#         addpar(name=f'{prefix}height', expr=height_expr.format(np.pi, tiny, pre=prefix))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative, ampscale=1.25)
        sigma = pars[f'{self.prefix}sigma']
        pars[f'{self.prefix}sigma_r'].set(value=sigma.value, min=sigma.min, max=sigma.max)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class VoigtModel(Model):
    r"""A model based on a Voigt distribution function.

    The model has four Parameters: `amplitude`, `center`, `sigma`, and
    `gamma`. By default, `gamma` is constrained to have a value equal to
    `sigma`, though it can be varied independently. In addition,
    parameters `fwhm` and `height` are included as constraints to report
    full width at half maximum and maximum peak height, respectively. The
    definition for the Voigt function used here is:

    .. math::

        f(x; A, \mu, \sigma, \gamma) = \frac{A \textrm{Re}[w(z)]}{\sigma\sqrt{2 \pi}}

    where

    .. math::
        :nowrap:

        \begin{eqnarray*}
            z &=& \frac{x-\mu +i\gamma}{\sigma\sqrt{2}} \\
            w(z) &=& e^{-z^2}{\operatorname{erfc}}(-iz)
        \end{eqnarray*}

    and :func:`erfc` is the complementary error function. As above,
    `amplitude` corresponds to :math:`A`, `center` to :math:`\mu`, and
    `sigma` to :math:`\sigma`. The parameter `gamma` corresponds to
    :math:`\gamma`. If `gamma` is kept at the default value (constrained
    to `sigma`), the full width at half maximum is approximately
    :math:`3.6013\sigma`.

    For more information, see: https://en.wikipedia.org/wiki/Voigt_profile

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(voigt, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('gamma', expr=f'{self.prefix}sigma')
        fexpr = ("1.0692*{pre:s}gamma+" +
                 "sqrt(0.8664*{pre:s}gamma**2+5.545083*{pre:s}sigma**2)")
        hexpr = ("({pre:s}amplitude/(max({0}, {pre:s}sigma*sqrt(2*pi))))*"
                 "real(wofz((1j*{pre:s}gamma)/(max({0}, {pre:s}sigma*sqrt(2)))))")
        self.set_param_hint('fwhm', expr=fexpr.format(pre=self.prefix))
        self.set_param_hint('height', expr=hexpr.format(tiny, pre=self.prefix))

#     def post_fit(self, result):
#         fexpr = ("1.0692*{pre:s}gamma+" +
#                  "sqrt(0.8664*{pre:s}gamma**2+5.545083*{pre:s}sigma**2)")
#         hexpr = ("({pre:s}amplitude/(max({0}, {pre:s}sigma*sqrt(2*pi))))*"
#                  "wofz((1j*{pre:s}gamma)/(max({0}, {pre:s}sigma*sqrt(2)))).real")
#
#         addpar = result.params.add
#         prefix = self.prefix
#         addpar(name=f'{prefix}fwhm', expr=fexpr.format(pre=prefix))
#         addpar(name=f'{prefix}height', expr=hexpr.format(tiny, pre=prefix))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative,
                               ampscale=1.5, sigscale=0.65)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class PseudoVoigtModel(Model):
    r"""A model based on a pseudo-Voigt distribution function.

    This is a weighted sum of a Gaussian and Lorentzian distribution
    function that share values for `amplitude` (:math:`A`), `center`
    (:math:`\mu`), and full width at half maximum `fwhm` (and so has
    constrained values of `sigma` (:math:`\sigma`) and `height` (maximum
    peak height). The parameter `fraction` (:math:`\alpha`) controls the
    relative weight of the Gaussian and Lorentzian components, giving the
    full definition of:

    .. math::

        f(x; A, \mu, \sigma, \alpha) = \frac{(1-\alpha)A}{\sigma_g\sqrt{2\pi}}
        e^{[{-{(x-\mu)^2}/{{2\sigma_g}^2}}]}
        + \frac{\alpha A}{\pi} \big[\frac{\sigma}{(x - \mu)^2 + \sigma^2}\big]

    where :math:`\sigma_g = {\sigma}/{\sqrt{2\ln{2}}}` so that the full
    width at half maximum of each component and of the sum is
    :math:`2\sigma`. The :meth:`guess` function always sets the starting
    value for `fraction` at 0.5.

    For more information, see:
    https://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_Approximation

    """

    fwhm_factor = 2.0

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(pvoigt, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('fraction', value=0.5, min=0.0, max=1.0)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))
        fmt = ("(((1-{prefix:s}fraction)*{prefix:s}amplitude)/"
               "max({0}, ({prefix:s}sigma*sqrt(pi/log(2))))+"
               "({prefix:s}fraction*{prefix:s}amplitude)/"
               "max({0}, (pi*{prefix:s}sigma)))")

        self.set_param_hint('height', expr=fmt.format(tiny, prefix=self.prefix))

#     def post_fit(self, result):
#         addpar = result.params.add
#         prefix = self.prefix
#         hexpr = ("(((1-{prefix:s}fraction)*{prefix:s}amplitude)/"
#                  "max({0}, ({prefix:s}sigma*sqrt(pi/log(2))))+"
#                  "({prefix:s}fraction*{prefix:s}amplitude)/"
#                  "max({0}, (pi*{prefix:s}sigma)))")
#
#         addpar(name=f'{prefix}fwhm', expr=fwhm_expr(self))
#         addpar(name=f'{prefix}height', expr=hexpr.format(tiny, prefix=prefix))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative, ampscale=1.25)
        pars[f'{self.prefix}fraction'].set(value=0.5, min=0.0, max=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class MoffatModel(Model):
    r"""A model based on the Moffat distribution function.

    The model has four Parameters: `amplitude` (:math:`A`), `center`
    (:math:`\mu`), a width parameter `sigma` (:math:`\sigma`), and an
    exponent `beta` (:math:`\beta`). In addition, parameters `fwhm` and
    `height` are included as constraints to report full width at half
    maximum and maximum peak height, respectively.

    .. math::

        f(x; A, \mu, \sigma, \beta) = A \big[(\frac{x-\mu}{\sigma})^2+1\big]^{-\beta}

    the full width at half maximum is :math:`2\sigma\sqrt{2^{1/\beta}-1}`.
    The :meth:`guess` function always sets the starting value for `beta`
    to 1.

    Note that for (:math:`\beta=1`) the Moffat has a Lorentzian shape. For
    more information, see:
    https://en.wikipedia.org/wiki/Moffat_distribution

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(moffat, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('beta')
        self.set_param_hint('fwhm', expr=f"2*{self.prefix}sigma*sqrt(2**(1.0/max(1e-3, {self.prefix}beta))-1)")
        self.set_param_hint('height', expr=f"{self.prefix}amplitude")

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5, sigscale=1.)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class Pearson4Model(Model):
    r"""A model based on a Pearson IV distribution.

    The model has five parameters: `amplitude` (:math:`A`), `center`
    (:math:`\mu`), `sigma` (:math:`\sigma`), `expon` (:math:`m`) and `skew` (:math:`\nu`).
    In addition, parameters `fwhm`, `height` and `position` are included as
    constraints to report estimates for the approximate full width at half maximum (20% error),
    the peak height, and the peak position (the position of the maximal  function value), respectively.
    The fwhm value has an error of about 20% in the
    parameter range expon: (0.5, 1000], skew: [-1000, 1000].

    .. math::

        f(x;A,\mu,\sigma,m,\nu)=A \frac{\left|\frac{\Gamma(m+i\tfrac{\nu}{2})}{\Gamma(m)}\right|^2}{\sigma\beta(m-\tfrac{1}{2},\tfrac{1}{2})}\left[1+\frac{(x-\mu)^2}{\sigma^2}\right]^{-m}\exp\left(-\nu \arctan\left(\frac{x-\mu}{\sigma}\right)\right)

    where :math:`\beta` is the beta function (see :scipydoc:`special.beta`).
    The :meth:`guess` function always gives a starting value of 1.5 for `expon`,
    and 0 for `skew`.

    For more information, see:
    https://en.wikipedia.org/wiki/Pearson_distribution#The_Pearson_type_IV_distribution

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(pearson4, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('expon', value=1.5, min=0.5 + tiny, max=1000)
        self.set_param_hint('skew', value=0.0, min=-1000, max=1000)
        fmt = ("{prefix:s}sigma*sqrt(2**(1/{prefix:s}expon)-1)*pi/arctan2(exp(1)*{prefix:s}expon, {prefix:s}skew)")
        self.set_param_hint('fwhm', expr=fmt.format(prefix=self.prefix))
        fmt = ("({prefix:s}amplitude / {prefix:s}sigma) * exp(2 * (real(loggammafcn({prefix:s}expon + {prefix:s}skew * 0.5j)) - loggammafcn({prefix:s}expon)) - betalnfnc({prefix:s}expon-0.5, 0.5) - "
               "{prefix:s}expon * log1p(square({prefix:s}skew/(2*{prefix:s}expon))) - {prefix:s}skew * arctan(-{prefix:s}skew/(2*{prefix:s}expon)))")
        self.set_param_hint('height', expr=fmt.format(tiny, prefix=self.prefix))
        fmt = ("{prefix:s}center-{prefix:s}sigma*{prefix:s}skew/(2*{prefix:s}expon)")
        self.set_param_hint('position', expr=fmt.format(prefix=self.prefix))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        pars[f'{self.prefix}expon'].set(value=1.5)
        pars[f'{self.prefix}skew'].set(value=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class Pearson7Model(Model):
    r"""A model based on a Pearson VII distribution.

    The model has four parameters: `amplitude` (:math:`A`), `center`
    (:math:`\mu`), `sigma` (:math:`\sigma`), and `exponent` (:math:`m`).
    In addition, parameters `fwhm` and `height` are included as
    constraints to report estimates for the full width at half maximum and
    maximum peak height, respectively.

    .. math::

        f(x; A, \mu, \sigma, m) = \frac{A}{\sigma{\beta(m-\frac{1}{2}, \frac{1}{2})}} \bigl[1 + \frac{(x-\mu)^2}{\sigma^2} \bigr]^{-m}

    where :math:`\beta` is the beta function (see :scipydoc:`special.beta`).
    The :meth:`guess` function always gives a starting value for `exponent`
    of 1.5. In addition, parameters `fwhm` and `height` are included as
    constraints to report full width at half maximum and maximum peak
    height, respectively.

    For more information, see:
    https://en.wikipedia.org/wiki/Pearson_distribution#The_Pearson_type_VII_distribution

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(pearson7, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('expon', value=1.5, max=100)
        fmt = ("sqrt(2**(1/{prefix:s}expon)-1)*2*{prefix:s}sigma")
        self.set_param_hint('fwhm', expr=fmt.format(prefix=self.prefix))
        fmt = ("{prefix:s}amplitude * gamfcn({prefix:s}expon)/"
               "max({0}, (gamfcn(0.5)*gamfcn({prefix:s}expon-0.5)*{prefix:s}sigma))")
        self.set_param_hint('height', expr=fmt.format(tiny, prefix=self.prefix))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        pars[f'{self.prefix}expon'].set(value=1.5)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class StudentsTModel(Model):
    r"""A model based on a Student's t-distribution function.

    The model has three Parameters: `amplitude` (:math:`A`), `center`
    (:math:`\mu`), and `sigma` (:math:`\sigma`). In addition, parameters
    `fwhm` and `height` are included as constraints to report full width
    at half maximum and maximum peak height, respectively.

    .. math::

        f(x; A, \mu, \sigma) = \frac{A \Gamma(\frac{\sigma+1}{2})} {\sqrt{\sigma\pi}\,\Gamma(\frac{\sigma}{2})} \Bigl[1+\frac{(x-\mu)^2}{\sigma}\Bigr]^{-\frac{\sigma+1}{2}}

    where :math:`\Gamma(x)` is the gamma function.

    For more information, see:
    https://en.wikipedia.org/wiki/Student%27s_t-distribution

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(students_t, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0.0, max=100)
        fmt = ("{prefix:s}amplitude*gamfcn(({prefix:s}sigma+1)/2)/"
               "(sqrt({prefix:s}sigma*pi)*gamfcn({prefix:s}sigma/2))")
        self.set_param_hint('height', expr=fmt.format(prefix=self.prefix))
        fmt = ("2*sqrt(2**(2/({prefix:s}sigma+1))*"
               "{prefix:s}sigma-{prefix:s}sigma)")
        self.set_param_hint('fwhm', expr=fmt.format(prefix=self.prefix))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class BreitWignerModel(Model):
    r"""A model based on a Breit-Wigner-Fano function.

    The model has four Parameters: `amplitude` (:math:`A`), `center`
    (:math:`\mu`), `sigma` (:math:`\sigma`), and `q` (:math:`q`).

    .. math::

        f(x; A, \mu, \sigma, q) = \frac{A (q\sigma/2 + x - \mu)^2}{(\sigma/2)^2 + (x - \mu)^2}

    For more information, see: https://en.wikipedia.org/wiki/Fano_resonance

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(breit_wigner, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0.0)

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        pars[f'{self.prefix}q'].set(value=1.0)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class LognormalModel(Model):
    r"""A model based on the Log-normal distribution function.

    The modal has three Parameters `amplitude` (:math:`A`), `center`
    (:math:`\mu`), and `sigma` (:math:`\sigma`). In addition, parameters
    `fwhm` and `height` are included as constraints to report estimates of
    full width at half maximum and maximum peak height, respectively.

    .. math::

        f(x; A, \mu, \sigma) = \frac{A}{\sigma\sqrt{2\pi}}\frac{e^{-(\ln(x) - \mu)^2/ 2\sigma^2}}{x}

    For more information, see: https://en.wikipedia.org/wiki/Lognormal

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(lognormal, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)

        fmt = ("{prefix:s}amplitude/max({0}, ({prefix:s}sigma*sqrt(2*pi)))"
               "*exp({prefix:s}sigma**2/2-{prefix:s}center)")
        self.set_param_hint('height', expr=fmt.format(tiny, prefix=self.prefix))
        fmt = ("exp({prefix:s}center-{prefix:s}sigma**2+{prefix:s}sigma*sqrt("
               "2*log(2)))-"
               "exp({prefix:s}center-{prefix:s}sigma**2-{prefix:s}sigma*sqrt("
               "2*log(2)))")
        self.set_param_hint('fwhm', expr=fmt.format(prefix=self.prefix))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = self.make_params(amplitude=1.0, center=0.0, sigma=0.25)
        pars[f'{self.prefix}sigma'].set(min=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class DampedOscillatorModel(Model):
    r"""A model based on the Damped Harmonic Oscillator Amplitude.

    The model has three Parameters: `amplitude` (:math:`A`), `center`
    (:math:`\mu`), and `sigma` (:math:`\sigma`). In addition, the
    parameter `height` is included as a constraint to report the maximum
    peak height.

    .. math::

        f(x; A, \mu, \sigma) = \frac{A}{\sqrt{ [1 - (x/\mu)^2]^2 + (2\sigma x/\mu)^2}}

    For more information, see:
    https://en.wikipedia.org/wiki/Harmonic_oscillator#Amplitude_part

    """

    height_factor = 0.5

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(damped_oscillator, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('height', expr=height_expr(self))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative,
                               ampscale=0.1, sigscale=0.1)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class DampedHarmonicOscillatorModel(Model):
    r"""A model based on a variation of the Damped Harmonic Oscillator.

    The model follows the definition given in DAVE/PAN (see:
    https://www.ncnr.nist.gov/dave) and has four Parameters: `amplitude`
    (:math:`A`), `center` (:math:`\mu`), `sigma` (:math:`\sigma`), and
    `gamma` (:math:`\gamma`). In addition, parameters `fwhm` and `height`
    are included as constraints to report estimates for full width at half
    maximum and maximum peak height, respectively.

    .. math::

        f(x; A, \mu, \sigma, \gamma) = \frac{A\sigma}{\pi [1 - \exp(-x/\gamma)]}
                \Big[ \frac{1}{(x-\mu)^2 + \sigma^2} - \frac{1}{(x+\mu)^2 + \sigma^2} \Big]

    where :math:`\gamma=kT`, `k` is the Boltzmann constant in
    :math:`evK^-1`, and `T` is the temperature in :math:`K`.

    For more information, see:
    https://en.wikipedia.org/wiki/Harmonic_oscillator

    """

    fwhm_factor = 2.0

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(dho, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('center', min=0)
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('gamma', min=1.e-19)
        fmt = ("({prefix:s}amplitude*{prefix:s}sigma)/"
               "max({0}, (pi*(1-exp(-{prefix:s}center/max({0}, {prefix:s}gamma)))))*"
               "(1/max({0}, {prefix:s}sigma**2)-1/"
               "max({0}, (4*{prefix:s}center**2+{prefix:s}sigma**2)))")
        self.set_param_hint('height', expr=fmt.format(tiny, prefix=self.prefix))
        self.set_param_hint('fwhm', expr=fwhm_expr(self))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative,
                               ampscale=0.1, sigscale=0.1)
        pars[f'{self.prefix}gamma'].set(value=1.0, min=0.0)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class ExponentialGaussianModel(Model):
    r"""A model of an Exponentially modified Gaussian distribution.

    The model has four Parameters: `amplitude` (:math:`A`), `center`
    (:math:`\mu`), `sigma` (:math:`\sigma`), and `gamma` (:math:`\gamma`).

    .. math::

        f(x; A, \mu, \sigma, \gamma) = \frac{A\gamma}{2}
        \exp\bigl[\gamma({\mu - x + \gamma\sigma^2/2})\bigr]
        {\operatorname{erfc}}\Bigl(\frac{\mu + \gamma\sigma^2 - x}{\sqrt{2}\sigma}\Bigr)


    where :func:`erfc` is the complementary error function.

    For more information, see:
    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(expgaussian, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('gamma', min=0, max=20)

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class SkewedGaussianModel(Model):
    r"""A skewed Gaussian model, using a skewed normal distribution.

    The model has four Parameters: `amplitude` (:math:`A`), `center`
    (:math:`\mu`), `sigma` (:math:`\sigma`), and `gamma` (:math:`\gamma`).

    .. math::

        f(x; A, \mu, \sigma, \gamma) = \frac{A}{\sigma\sqrt{2\pi}}
        e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]} \Bigl\{ 1 +
        {\operatorname{erf}}\bigl[
        \frac{{\gamma}(x-\mu)}{\sigma\sqrt{2}}
        \bigr] \Bigr\}

    where :func:`erf` is the error function.

    For more information, see:
    https://en.wikipedia.org/wiki/Skew_normal_distribution

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(skewed_gaussian, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class SkewedVoigtModel(Model):
    r"""A skewed Voigt model, modified using a skewed normal distribution.

    The model has five Parameters `amplitude` (:math:`A`), `center`
    (:math:`\mu`), `sigma` (:math:`\sigma`), and `gamma` (:math:`\gamma`),
    as usual for a Voigt distribution, and adds a new Parameter `skew`.

    .. math::

        f(x; A, \mu, \sigma, \gamma, \rm{skew}) = {\rm{Voigt}}(x; A, \mu, \sigma, \gamma)
        \Bigl\{ 1 + {\operatorname{erf}}\bigl[
        \frac{{\rm{skew}}(x-\mu)}{\sigma\sqrt{2}}
        \bigr] \Bigr\}

    where :func:`erf` is the error function.

    For more information, see:
    https://en.wikipedia.org/wiki/Skew_normal_distribution

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(skewed_voigt, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('gamma', expr=f'{self.prefix}sigma')

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class ThermalDistributionModel(Model):
    r"""Return a thermal distribution function.

    Variable `form` defines the kind of distribution as below with three
    Parameters: `amplitude` (:math:`A`), `center` (:math:`x_0`), and `kt`
    (:math:`kt`). The following distributions are available:

    - `'bose'` : Bose-Einstein distribution (default)
    - `'maxwell'` : Maxwell-Boltzmann distribution
    - `'fermi'` : Fermi-Dirac distribution

    The functional forms are defined as:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        & f(x; A, x_0, kt, {\mathrm{form={}'bose{}'}}) & = \frac{1}{A \exp(\frac{x - x_0}{kt}) - 1} \\
        & f(x; A, x_0, kt, {\mathrm{form={}'maxwell{}'}}) & = \frac{1}{A \exp(\frac{x - x_0}{kt})} \\
        & f(x; A, x_0, kt, {\mathrm{form={}'fermi{}'}}) & = \frac{1}{A \exp(\frac{x - x_0}{kt}) + 1} ]
        \end{eqnarray*}

    Notes
    -----
    - `kt` should be defined in the same units as `x` (:math:`k_B =
      8.617\times10^{-5}` eV/K).
    - set :math:`kt<0` to implement the energy loss convention common in
      scattering research.

    For more information, see:
    http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/disfcn.html

    """

    valid_forms = ('bose', 'maxwell', 'fermi')

    def __init__(self, independent_vars=['x', 'form'], prefix='',
                 nan_policy='raise', form='bose', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'form': form, 'independent_vars': independent_vars})
        super().__init__(thermal_distribution, **kwargs)
        self._set_paramhints_prefix()

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        return guess_thermal(self, data, x)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class BoseModel(Model):
    r"""Return a Bose-Einstein thermal distribution function, with functional form:

    .. math::
        :nowrap:

        \begin{equation}
         f(x; A, x_0, kt) = \frac{A}{\exp(\frac{x - x_0}{kt}) - 1} \\
        \end{equation}

    Notes
    -----
    - `kt` should be defined in the same units as `x` (:math:`k_B = 8.617\times10^{-5}` eV/K).
    """
    def __init__(self, prefix='', nan_policy='raise', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy})
        super().__init__(bose, **kwargs)
        self._set_paramhints_prefix()

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        return guess_thermal(self, data, x)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class FermiModel(Model):
    r"""Return a Fermi-Dirac thermal distribution function, with functional form:

    .. math::
        :nowrap:

        \begin{equation}
         f(x; A, x_0, kt) = \frac{A}{\exp(\frac{x - x_0}{kt}) + 1} \\
        \end{equation}

    Notes
    -----
    - `kt` should be defined in the same units as `x` (:math:`k_B = 8.617\times10^{-5}` eV/K).
    """
    def __init__(self, prefix='', nan_policy='raise', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy})
        super().__init__(fermi, **kwargs)
        self._set_paramhints_prefix()

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        return guess_thermal(self, data, x)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class DoniachModel(Model):
    r"""A model of an Doniach Sunjic asymmetric lineshape.

    This model is used in photo-emission and has four Parameters:
    `amplitude` (:math:`A`), `center` (:math:`\mu`), `sigma`
    (:math:`\sigma`), and `gamma` (:math:`\gamma`). In addition, parameter
    `height` is included as a constraint to report maximum peak height.

    .. math::

        f(x; A, \mu, \sigma, \gamma) = \frac{A}{\sigma^{1-\gamma}}
        \frac{\cos\bigl[\pi\gamma/2 + (1-\gamma)
        \arctan{((x - \mu)/\sigma)}\bigr]} {\bigl[1 + {((x-\mu)/\sigma)}^2\bigr]^{(1-\gamma)/2}}

    For more information, see:
    https://www.casaxps.com/help_manual/line_shapes.htm

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(doniach, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        fmt = ("{prefix:s}amplitude/max({0}, ({prefix:s}sigma**(1-{prefix:s}gamma)))"
               "*cos(pi*{prefix:s}gamma/2)")
        self.set_param_hint('height', expr=fmt.format(tiny, prefix=self.prefix))

    def guess(self, data, x, negative=False, **kwargs):
        """Estimate initial model parameter values from data."""
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class PowerLawModel(Model):
    r"""A model based on a Power Law.

    The model has two Parameters: `amplitude` (:math:`A`) and `exponent`
    (:math:`k`) and is defined as:

    .. math::

        f(x; A, k) = A x^k

    For more information, see: https://en.wikipedia.org/wiki/Power_law

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(powerlaw, **kwargs)

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from data."""
        try:
            expon, amp = np.polyfit(np.log(x+1.e-14), np.log(data+1.e-14), 1)
        except TypeError:
            expon, amp = 1, np.log(abs(max(data)+1.e-9))

        pars = self.make_params(amplitude=np.exp(amp), exponent=expon)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class ExponentialModel(Model):
    r"""A model based on an exponential decay function.

    The model has two Parameters: `amplitude` (:math:`A`) and `decay`
    (:math:`\tau`) and is defined as:

    .. math::

        f(x; A, \tau) = A e^{-x/\tau}

    For more information, see:
    https://en.wikipedia.org/wiki/Exponential_decay

    """

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(exponential, **kwargs)

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from data."""
        try:
            sval, oval = np.polyfit(x, np.log(abs(data)+1.e-15), 1)
        except TypeError:
            sval, oval = 1., np.log(abs(max(data)+1.e-9))
        pars = self.make_params(amplitude=np.exp(oval), decay=-1.0/sval)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class StepModel(Model):
    r"""A model based on a Step function.

    The model has three Parameters: `amplitude` (:math:`A`), `center`
    (:math:`\mu`), and `sigma` (:math:`\sigma`).

    There are four choices for `form`:

    - `'linear'` (default)
    - `'atan'` or `'arctan'` for an arc-tangent function
    - `'erf'` for an error function
    - `'logistic'` for a logistic function (for more information, see:
      https://en.wikipedia.org/wiki/Logistic_function)

    The step function starts with a value 0 and ends with a value of
    :math:`\tt{sign}(\sigma)A` rising or falling to :math:`A/2` at :math:`\mu`,
    with :math:`\sigma` setting the characteristic width of the step.
    The functional forms are defined as:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        & f(x; A, \mu, \sigma, {\mathrm{form={}'linear{}'}})  & = A \min{[1, \max{(0, \alpha + 1/2)}]} \\
        & f(x; A, \mu, \sigma, {\mathrm{form={}'arctan{}'}})  & = A [1/2 + \arctan{(\alpha)}/{\pi}] \\
        & f(x; A, \mu, \sigma, {\mathrm{form={}'erf{}'}})     & = A [1 + {\operatorname{erf}}(\alpha)]/2 \\
        & f(x; A, \mu, \sigma, {\mathrm{form={}'logistic{}'}})& = A \left[1 - \frac{1}{1 + e^{\alpha}} \right]
        \end{eqnarray*}

    where :math:`\alpha = (x - \mu)/{\sigma}`.

    Note that :math:`\sigma > 0` gives a rising step, while :math:`\sigma < 0` gives
    a falling step.
    """

    valid_forms = ('linear', 'atan', 'arctan', 'erf', 'logistic')

    def __init__(self, independent_vars=['x', 'form'], prefix='',
                 nan_policy='raise', form='linear', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'form': form, 'independent_vars': independent_vars})
        super().__init__(step, **kwargs)

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from data."""
        ymin, ymax = min(data), max(data)
        xmin, xmax = min(x), max(x)
        pars = self.make_params(amplitude=(ymax-ymin),
                                center=(xmax+xmin)/2.0)
        n = len(data)
        sigma = 0.1*(xmax - xmin)
        if data[:n//5].mean() > data[-n//5:].mean():
            sigma = -sigma
        pars[f'{self.prefix}sigma'].set(value=sigma)
        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class RectangleModel(Model):
    r"""A model based on a Step-up and Step-down function.

    The model has five Parameters: `amplitude` (:math:`A`), `center1`
    (:math:`\mu_1`), `center2` (:math:`\mu_2`), `sigma1`
    (:math:`\sigma_1`), and `sigma2` (:math:`\sigma_2`).

    There are four choices for `form`, which is used for both the Step up
    and the Step down:

    - `'linear'` (default)
    - `'atan'` or `'arctan'` for an arc-tangent function
    - `'erf'` for an error function
    - `'logistic'` for a logistic function (for more information, see:
      https://en.wikipedia.org/wiki/Logistic_function)

    The function starts with a value 0 and transitions to a value of
    :math:`A`, taking the value :math:`A/2` at :math:`\mu_1`, with
    :math:`\sigma_1` setting the characteristic width. The function then
    transitions again to the value :math:`A/2` at :math:`\mu_2`, with
    :math:`\sigma_2` setting the characteristic width. The functional
    forms are defined as:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        &f(x; A, \mu, \sigma, {\mathrm{form={}'linear{}'}})   &= A \{ \min{[1, \max{(-1, \alpha_1)}]} + \min{[1, \max{(-1, \alpha_2)}]} \}/2 \\
        &f(x; A, \mu, \sigma, {\mathrm{form={}'arctan{}'}})   &= A [\arctan{(\alpha_1)} + \arctan{(\alpha_2)}]/{\pi} \\
        &f(x; A, \mu, \sigma, {\mathrm{form={}'erf{}'}})      &= A \left[{\operatorname{erf}}(\alpha_1) + {\operatorname{erf}}(\alpha_2)\right]/2 \\
        &f(x; A, \mu, \sigma, {\mathrm{form={}'logistic{}'}}) &= A \left[1 - \frac{1}{1 + e^{\alpha_1}} - \frac{1}{1 + e^{\alpha_2}} \right]
        \end{eqnarray*}


    where :math:`\alpha_1 = (x - \mu_1)/{\sigma_1}` and
    :math:`\alpha_2 = -(x - \mu_2)/{\sigma_2}`.

    Note that, unlike a StepModel, :math:`\sigma_1 > 0` is enforced, giving a
    rising initial step, and  :math:`\sigma_2 > 0` gives a falling final step.
    """

    valid_forms = ('linear', 'atan', 'arctan', 'erf', 'logistic')

    def __init__(self, independent_vars=['x', 'form'], prefix='',
                 nan_policy='raise', form='linear', **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'form': form, 'independent_vars': independent_vars})
        super().__init__(rectangle, **kwargs)

        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('center1')
        self.set_param_hint('center2')
        self.set_param_hint('midpoint',
                            expr=f'({self.prefix}center1+{self.prefix}center2)/2.0')

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from data."""
        ymin, ymax = min(data), max(data)
        xmin, xmax = min(x), max(x)
        pars = self.make_params(amplitude=(ymax-ymin),
                                center1=(xmax+xmin)/4.0,
                                center2=3*(xmax+xmin)/4.0,
                                sigma1={'value': (xmax-xmin)/10.0, 'min': 0},
                                sigma2={'value': (xmax-xmin)/10.0, 'min': 0})

        return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    guess.__doc__ = COMMON_GUESS_DOC


class ExpressionModel(Model):
    """ExpressionModel class."""

    idvar_missing = "No independent variable found in\n {}"
    idvar_notfound = "Cannot find independent variables '{}' in\n {}"
    no_prefix = "ExpressionModel does not support `prefix` argument"

    def __init__(self, expr, independent_vars=None, init_script=None,
                 nan_policy='raise', **kws):
        """Generate a model from user-supplied expression.

        Parameters
        ----------
        expr : str
            Mathematical expression for model.
        independent_vars : :obj:`list` of :obj:`str` or None, optional
            Variable names to use as independent variables.
        init_script : str or None, optional
            Initial script to run in asteval interpreter.
        nan_policy : {'raise, 'propagate', 'omit'}, optional
            How to handle NaN and missing values in data. See Notes below.
        **kws : optional
            Keyword arguments to pass to :class:`Model`.

        Notes
        -----
        1. each instance of ExpressionModel will create and use its own
           version of an asteval interpreter.

        2. `prefix` is **not supported** for ExpressionModel.

        3. `nan_policy` sets what to do when a NaN or missing value is
           seen in the data. Should be one of:

            - `'raise'` : raise a `ValueError` (default)
            - `'propagate'` : do nothing
            - `'omit'` : drop missing data

        """
        if 'prefix' in kws:
            raise Warning(self.no_prefix)
        kws["nan_policy"] = nan_policy

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
            raise ValueError(self.idvar_missing.format(self.expr))

        # determine which named symbols are parameter names,
        # try to find all independent variables
        idvar_found = [False]*len(independent_vars)
        param_names = []
        for name in sym_names:
            if name in independent_vars:
                idvar_found[independent_vars.index(name)] = True
            elif name not in param_names and name not in self.asteval.symtable:
                param_names.append(name)

        # make sure we have all independent parameters
        if not all(idvar_found):
            lost = []
            for ix, found in enumerate(idvar_found):
                if not found:
                    lost.append(independent_vars[ix])
            lost = ', '.join(lost)
            raise ValueError(self.idvar_notfound.format(lost, self.expr))

        kws['independent_vars'] = self.independent_vars = independent_vars

        def _eval(**kwargs):
            for name, val in kwargs.items():
                self.asteval.symtable[name] = val
            return self.asteval.run(self.astcode)

        kws["nan_policy"] = nan_policy

        super().__init__(_eval, **kws)

        # set param names here, and other things normally
        # set in _parse_params(), which will be short-circuited.
        self._func_allargs = independent_vars + param_names
        self._param_names = param_names
        self._func_haskeywords = True
        self.independent_var_defvals = {'x': inspect._empty}
        self.def_vals = {}

    def __repr__(self):
        """Return printable representation of ExpressionModel."""
        return f"<lmfit.ExpressionModel('{self.expr}')>"

    def _reprstring(self, long=False):
        """Return printable representation of ExpressionModel."""
        return f"<lmfit.ExpressionModel('{self.expr}')>"

    def _parse_params(self):
        """Over-write ExpressionModel._parse_params with `pass`.

        This prevents normal parsing of function for parameter names.

        """
        pass


lmfit_models = {'Constant': ConstantModel,
                'Complex Constant': ComplexConstantModel,
                'Linear': LinearModel,
                'Quadratic': QuadraticModel,
                'Polynomial': PolynomialModel,
                'Spline': SplineModel,
                'Gaussian': GaussianModel,
                'Gaussian-2D': Gaussian2dModel,
                'Lorentzian': LorentzianModel,
                'Split-Lorentzian': SplitLorentzianModel,
                'Voigt': VoigtModel,
                'PseudoVoigt': PseudoVoigtModel,
                'Moffat': MoffatModel,
                'Pearson4': Pearson4Model,
                'Pearson7': Pearson7Model,
                'StudentsT': StudentsTModel,
                'Breit-Wigner': BreitWignerModel,
                'Log-Normal': LognormalModel,
                'Damped Oscillator': DampedOscillatorModel,
                'Damped Harmonic Oscillator': DampedHarmonicOscillatorModel,
                'Exponential Gaussian': ExponentialGaussianModel,
                'Skewed Gaussian': SkewedGaussianModel,
                'Skewed Voigt': SkewedVoigtModel,
                'Thermal Distribution': ThermalDistributionModel,
                'Bose-Einstein Distribution': BoseModel,
                'Fermi-Dirac Distribution': FermiModel,
                'Doniach': DoniachModel,
                'Power Law': PowerLawModel,
                'Exponential': ExponentialModel,
                'Step': StepModel,
                'Rectangle': RectangleModel,
                'Expression': ExpressionModel}
