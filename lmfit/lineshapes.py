"""Basic model lineshapes and distribution functions."""

from numpy import (arctan, array, copysign, cos, exp, isclose, isnan, log,
                   log1p, maximum, minimum, ones, pi, polyval, real, shape,
                   sign, sin, sqrt, where)
from scipy.interpolate import splev
from scipy.special import betaln as betalnfcn
from scipy.special import erf, erfc
from scipy.special import gamma as gamfcn
from scipy.special import loggamma as loggammafcn
from scipy.special import wofz

log2 = log(2)
s2pi = sqrt(2*pi)
s2 = sqrt(2.0)
# tiny had been numpy.finfo(numpy.float64).eps ~=2.2e16.
# here, we explicitly set it to 1.e-15 == numpy.finfo(numpy.float64).resolution
tiny = 1.0e-15

functions = ('gaussian', 'gaussian2d', 'lorentzian', 'voigt',
             'pvoigt', 'moffat', 'pearson4', 'pearson7',
             'breit_wigner', 'damped_oscillator', 'dho', 'logistic',
             'lognormal', 'students_t', 'expgaussian', 'doniach',
             'skewed_gaussian', 'skewed_voigt',
             'thermal_distribution', 'bose', 'fermi', 'step',
             'rectangle', 'exponential', 'powerlaw', 'linear',
             'parabolic', 'sine', 'expsine', 'split_lorentzian',
             'constant', 'complex_constant', 'polynomial')


def not_zero(value):
    """Return value with a minimal absolute size of tiny, preserving the sign.

    This is a helper function to prevent ZeroDivisionError's.

    Parameters
    ----------
    value : scalar
        Value to be ensured not to be zero.

    Returns
    -------
    scalar
        Value ensured not to be zero.

    """
    return float(copysign(max(tiny, abs(value)), value))


def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * exp(-(1.0*x-center)**2 / max(tiny, (2*sigma**2))))


def gaussian2d(x, y=0.0, amplitude=1.0, centerx=0.0, centery=0.0, sigmax=1.0,
               sigmay=1.0):
    """Return a 2-dimensional Gaussian function.

    gaussian2d(x, y, amplitude, centerx, centery, sigmax, sigmay) =
        amplitude/(2*pi*sigmax*sigmay) * exp(-(x-centerx)**2/(2*sigmax**2)
                                             -(y-centery)**2/(2*sigmay**2))

    """
    z = amplitude*(gaussian(x, amplitude=1, center=centerx, sigma=sigmax) *
                   gaussian(y, amplitude=1, center=centery, sigma=sigmay))
    return z


def lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Lorentzian function.

    lorentzian(x, amplitude, center, sigma) =
        (amplitude/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)

    """
    return ((amplitude/(1 + ((1.0*x-center)/max(tiny, sigma))**2))
            / max(tiny, (pi*sigma)))


def split_lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0, sigma_r=1.0):
    """Return a 1-dimensional piecewise Lorentzian function.

    Split means that width of the function is different between
    left and right slope of the function. The peak height is calculated
    from the condition that the integral from ``-numpy.inf`` to
    ``numpy.inf`` is equal to `amplitude`.

    split_lorentzian(x, amplitude, center, sigma, sigma_r) =
        [2*amplitude / (pi* (sigma + sigma_r)] *
         {  sigma**2   * (x<center)  / [sigma**2  + (x - center)**2]
          + sigma_r**2 * (x>=center) / [sigma_r**2+ (x - center)**2] }

    """
    s = max(tiny, sigma)
    r = max(tiny, sigma_r)
    ss = s*s
    rr = r*r
    xc2 = (x-center)**2
    amp = 2*amplitude/(pi*(s+r))
    return amp*(ss*(x < center)/(ss+xc2) + rr*(x >= center)/(rr+xc2))


def voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None):
    """Return a 1-dimensional Voigt function.

    voigt(x, amplitude, center, sigma, gamma) =
        amplitude*real(wofz(z)) / (sigma*s2pi)

    For more information, see: https://en.wikipedia.org/wiki/Voigt_profile

    """
    if gamma is None:
        gamma = sigma
    z = (x-center + 1j*gamma) / max(tiny, (sigma*s2))
    return amplitude*real(wofz(z)) / max(tiny, (sigma*s2pi))


def pvoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5):
    """Return a 1-dimensional pseudo-Voigt function.

    pvoigt(x, amplitude, center, sigma, fraction) =
        amplitude*(1-fraction)*gaussian(x, center, sigma_g) +
        amplitude*fraction*lorentzian(x, center, sigma)

    where `sigma_g` (the sigma for the Gaussian component) is

        ``sigma_g = sigma / sqrt(2*log(2)) ~= sigma / 1.17741``

    so that the Gaussian and Lorentzian components have the same FWHM of
    ``2.0*sigma``.

    """
    sigma_g = sigma / sqrt(2*log2)
    return ((1-fraction)*gaussian(x, amplitude, center, sigma_g) +
            fraction*lorentzian(x, amplitude, center, sigma))


def moffat(x, amplitude=1, center=0., sigma=1, beta=1.):
    """Return a 1-dimensional Moffat function.

    moffat(x, amplitude, center, sigma, beta) =
        amplitude / (((x - center)/sigma)**2 + 1)**beta

    """
    return amplitude / (((x - center)/max(tiny, sigma))**2 + 1)**beta


def pearson4(x, amplitude=1.0, center=0.0, sigma=1.0, expon=1.0, skew=0.0):
    """Return a Pearson4 lineshape.

    Using the Wikipedia definition:

    pearson4(x, amplitude, center, sigma, expon, skew) =
        amplitude*|gamma(expon + I skew/2)/gamma(m)|**2/(w*beta(expon-0.5, 0.5)) * (1+arg**2)**(-expon) * exp(-skew * arctan(arg))

    where ``arg = (x-center)/sigma``, `gamma` is the gamma function and `beta` is the beta function.

    For more information, see: https://en.wikipedia.org/wiki/Pearson_distribution#The_Pearson_type_IV_distribution

    """
    expon = max(tiny, expon)
    sigma = max(tiny, sigma)
    arg = (x - center) / sigma
    logprefactor = 2 * (real(loggammafcn(expon + skew * 0.5j)) - loggammafcn(expon)) - betalnfcn(expon - 0.5, 0.5)
    return (amplitude / sigma) * exp(logprefactor - expon * log1p(arg * arg) - skew * arctan(arg))


def pearson7(x, amplitude=1.0, center=0.0, sigma=1.0, expon=1.0):
    """Return a Pearson7 lineshape.

    Using the Wikipedia definition:

    pearson7(x, center, sigma, expon) =
        amplitude*(1+arg**2)**(-expon)/(sigma*beta(expon-0.5, 0.5))

    where ``arg = (x-center)/sigma`` and `beta` is the beta function.

    """
    expon = max(tiny, expon)
    arg = (x-center)/max(tiny, sigma)
    scale = amplitude * gamfcn(expon)/(gamfcn(0.5)*gamfcn(expon-0.5))
    return scale*(1+arg**2)**(-expon)/max(tiny, sigma)


def breit_wigner(x, amplitude=1.0, center=0.0, sigma=1.0, q=1.0):
    """Return a Breit-Wigner-Fano lineshape.

    breit_wigner(x, amplitude, center, sigma, q) =
        amplitude*(q*sigma/2 + x - center)**2 /
            ( (sigma/2)**2 + (x - center)**2 )

    """
    gam = sigma/2.0
    return amplitude*(q*gam + x - center)**2 / (gam*gam + (x-center)**2)


def damped_oscillator(x, amplitude=1.0, center=1., sigma=0.1):
    """Return the amplitude for a damped harmonic oscillator.

    damped_oscillator(x, amplitude, center, sigma) =
        amplitude/sqrt( (1.0 - (x/center)**2)**2 + (2*sigma*x/center)**2))

    """
    center = max(tiny, abs(center))
    return amplitude/sqrt((1.0 - (x/center)**2)**2 + (2*sigma*x/center)**2)


def dho(x, amplitude=1., center=1., sigma=1., gamma=1.0):
    """Return a Damped Harmonic Oscillator.

    Similar to the version from PAN:

    dho(x, amplitude, center, sigma, gamma) =
        amplitude*sigma*pi * (lm - lp) / (1.0 - exp(-x/gamma))

    where
        ``lm(x, center, sigma) = 1.0 / ((x-center)**2 + sigma**2)``
        ``lp(x, center, sigma) = 1.0 / ((x+center)**2 + sigma**2)``

    """
    factor = amplitude * sigma / pi
    bose = (1.0 - exp(-x/max(tiny, gamma)))
    if isinstance(bose, (int, float)):
        bose = not_zero(bose)
    else:
        bose[where(isnan(bose))] = tiny
        bose[where(abs(bose) <= tiny)] = tiny

    lm = 1.0/((x-center)**2 + sigma**2)
    lp = 1.0/((x+center)**2 + sigma**2)
    return factor * where(isclose(x, 0.0),
                          4*gamma*center/(center**2+sigma**2)**2,
                          (lm - lp)/bose)


def logistic(x, amplitude=1., center=0., sigma=1.):
    """Return a Logistic lineshape (yet another sigmoidal curve).

    logistic(x, amplitude, center, sigma) =
        amplitude*(1. - 1. / (1 + exp((x-center)/sigma)))

    """
    return amplitude*(1. - 1./(1. + exp((x-center)/max(tiny, sigma))))


def lognormal(x, amplitude=1.0, center=0., sigma=1):
    """Return a log-normal function.

    lognormal(x, amplitude, center, sigma) =
        (amplitude/(x*sigma*s2pi)) * exp(-(ln(x) - center)**2/ (2* sigma**2))

    """
    if isinstance(x, (int, float)):
        x = max(tiny, x)
    else:
        x[where(x <= tiny)] = tiny
    return ((amplitude/(x*max(tiny, sigma*s2pi))) *
            exp(-(log(x)-center)**2 / max(tiny, (2*sigma**2))))


def students_t(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return Student's t-distribution.

    students_t(x, amplitude, center, sigma) =

             gamma((sigma+1)/2)
        ---------------------------- * (1 + (x-center)**2/sigma)^(-(sigma+1)/2)
        sqrt(sigma*pi)gamma(sigma/2)

    """
    s1 = (sigma+1)/2.0
    denom = max(tiny, (sqrt(sigma*pi)*gamfcn(max(tiny, sigma/2))))
    return amplitude*(1 + (x-center)**2/max(tiny, sigma))**(-s1) * gamfcn(s1) / denom


def expgaussian(x, amplitude=1, center=0, sigma=1.0, gamma=1.0):
    """Return an exponentially modified Gaussian.

    expgaussian(x, amplitude, center, sigma, gamma) =
        amplitude * (gamma/2) *
        exp[center*gamma + (gamma*sigma)**2/2 - gamma*x] *
        erfc[(center + gamma*sigma**2 - x)/(sqrt(2)*sigma)]

    For more information, see:
    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

    """
    gss = gamma*sigma*sigma
    arg1 = gamma*(center + gss/2.0 - x)
    arg2 = (center + gss - x)/max(tiny, (s2*sigma))
    return amplitude*(gamma/2) * exp(arg1) * erfc(arg2)


def doniach(x, amplitude=1.0, center=0, sigma=1.0, gamma=0.0):
    """Return a Doniach Sunjic asymmetric lineshape.

    doniach(x, amplitude, center, sigma, gamma) =
        amplitude / sigma^(1-gamma) *
        cos(pi*gamma/2 + (1-gamma) arctan((x-center)/sigma) /
        (sigma**2 + (x-center)**2)**[(1-gamma)/2]

    For example used in photo-emission; see
    http://www.casaxps.com/help_manual/line_shapes.htm for more information.

    """
    arg = (x-center)/max(tiny, sigma)
    gm1 = (1.0 - gamma)
    scale = amplitude/max(tiny, (sigma**gm1))
    return scale*cos(pi*gamma/2 + gm1*arctan(arg))/(1 + arg**2)**(gm1/2)


def skewed_gaussian(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=0.0):
    """Return a Gaussian lineshape, skewed with error function.

    Equal to: gaussian(x, center, sigma)*(1+erf(beta*(x-center)))

    where ``beta = gamma/(sigma*sqrt(2))``

    with ``gamma < 0``: tail to low value of centroid
         ``gamma > 0``: tail to high value of centroid

    For more information, see:
    https://en.wikipedia.org/wiki/Skew_normal_distribution

    """
    asym = 1 + erf(gamma*(x-center)/max(tiny, (s2*sigma)))
    return asym * gaussian(x, amplitude, center, sigma)


def skewed_voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None, skew=0.0):
    """Return a Voigt lineshape, skewed with error function.

    Equal to: voigt(x, center, sigma, gamma)*(1+erf(beta*(x-center)))

    where ``beta = skew/(sigma*sqrt(2))``

    with ``skew < 0``: tail to low value of centroid
         ``skew > 0``: tail to high value of centroid

    Useful, for example, for ad-hoc Compton scatter profile. For more
    information, see: https://en.wikipedia.org/wiki/Skew_normal_distribution

    """
    beta = skew/max(tiny, (s2*sigma))
    asym = 1 + erf(beta*(x-center))
    return asym * voigt(x, amplitude, center, sigma, gamma=gamma)


def sine(x, amplitude=1.0, frequency=1.0, shift=0.0):
    """Return a sinusoidal function.

    sine(x, amplitude, frequency, shift) =
        amplitude * sin(x*frequency + shift)

    """
    return amplitude*sin(x*frequency + shift)


def expsine(x, amplitude=1.0, frequency=1.0, shift=0.0, decay=0.0):
    """Return an exponentially decaying sinusoidal function.

    expsine(x, amplitude, frequency, shift, decay) =
        amplitude * sin(x*frequency + shift) * exp(-x*decay)

    """
    return amplitude*sin(x*frequency + shift) * exp(-x*decay)


def thermal_distribution(x, amplitude=1.0, center=0.0, kt=1.0, form='bose'):
    """Return a thermal distribution function.

    The variable `form` defines the kind of distribution:

    - ``form='bose'`` (default) is the Bose-Einstein distribution:

        thermal_distribution(x, amplitude=1.0, center=0.0, kt=1.0) =
            1/(amplitude*exp((x - center)/kt) - 1)

    - ``form='maxwell'`` is the Maxwell-Boltzmann distribution:
        thermal_distribution(x, amplitude=1.0, center=0.0, kt=1.0) =
            1/(amplitude*exp((x - center)/kt))

    - ``form='fermi'`` is the Fermi-Dirac distribution:
        thermal_distribution(x, amplitude=1.0, center=0.0, kt=1.0) =
            1/(amplitude*exp((x - center)/kt) + 1)

    Notes
    -----
    - `kt` should be defined in the same units as `x`. The Boltzmann
      constant is ``kB = 8.617e-5 eV/K``.
    - set ``kt<0`` to implement the energy loss convention common in
      scattering research.

    For more information, see:
    http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/disfcn.html

    """
    form = form.lower()
    if form.startswith('bose'):
        offset = -1
    elif form.startswith('maxwell'):
        offset = 0
    elif form.startswith('fermi'):
        offset = 1
    else:
        msg = (f"Invalid value ('{form}') for argument 'form'; should be one "
               "of 'maxwell', 'fermi', or 'bose'.")
        raise ValueError(msg)

    return real(1/(amplitude*exp((x - center)/not_zero(kt)) + offset + tiny*1j))


def bose(x, amplitude=1.0, center=0.0, kt=1.0):
    """Return a Bose-Einstein thermal distribution function.

    bose(x, amplitude=1.0, center=0.0, kt=1.0)
       = amplitude/(exp((x - center)/kt) - 1)

    Notes
    -----
    - `kt` should be defined in the same units as `x`. The Boltzmann
      constant is ``kB = 8.617e-5 eV/K``.

    See Also
    --------
    thermal_distribution, fermi
    """
    denom = exp((x - center)/not_zero(kt)) - 1.0
    if isinstance(x, (int, float)):
        denom = not_zero(denom)
    else:
        denom[where(abs(denom) < tiny*tiny)] = tiny*tiny
    return amplitude/denom


def fermi(x, amplitude=1.0, center=0.0, kt=1.0):
    """Return a Fermi-Dirac thermal distribution function.

    fermi(x, amplitude=1.0, center=0.0, kt=1.0)
       = 1/(amplitude *exp((x - center)/kt) + 1)

    Notes
    -----
    - `kt` should be defined in the same units as `x`. The Boltzmann
      constant is ``kB = 8.617e-5 eV/K``.

    See Also
    --------
    thermal_distribution, bose
    """
    return amplitude/(exp((x - center)/not_zero(kt)) + 1.0)


def step(x, amplitude=1.0, center=0.0, sigma=1.0, form='linear'):
    """Return a step function.

    Starts at 0.0, ends at `sign(sigma)*amplitude`, has a half-max at
    `center`, rising or falling with `form`:

    - `'linear'` (default) = amplitude * min(1, max(0, arg + 0.5))
    - `'atan'`, `'arctan'` = amplitude * (0.5 + atan(arg)/pi)
    - `'erf'`              = amplitude * (1 + erf(arg))/2.0
    - `'logistic'`         = amplitude * [1 - 1/(1 + exp(arg))]

    where ``arg = (x - center)/sigma``.

    Note that ``sigma > 0`` gives a rising step, while ``sigma < 0`` gives
    a falling step.
    """
    out = sign(sigma)*(x - center)/max(tiny*tiny, abs(sigma))

    if form == 'erf':
        out = 0.5*(1 + erf(out))
    elif form == 'logistic':
        out = 1. - 1./(1. + exp(out))
    elif form in ('atan', 'arctan'):
        out = 0.5 + arctan(out)/pi
    elif form == 'linear':
        out = minimum(1, maximum(0, out + 0.5))
    else:
        msg = (f"Invalid value ('{form}') for argument 'form'; should be one "
               "of 'erf', 'logistic', 'atan', 'arctan', or 'linear'.")
        raise ValueError(msg)

    return amplitude*out


def rectangle(x, amplitude=1.0, center1=0.0, sigma1=1.0,
              center2=1.0, sigma2=1.0, form='linear'):
    """Return a rectangle function: step up, step down.
    Starts at 0.0, rises to `amplitude` (at `center1` with width `sigma1`),
    then drops to 0.0 (at `center2` with width `sigma2`) with `form`:
    - `'linear'` (default) = ramp_up + ramp_down
    - `'atan'`, `'arctan`' = amplitude*(atan(arg1) + atan(arg2))/pi
    - `'erf'`              = amplitude*(erf(arg1) + erf(arg2))/2.
    - `'logisitic'`        = amplitude*[1 - 1/(1 + exp(arg1)) - 1/(1+exp(arg2))]

    where ``arg1 = (x - center1)/sigma1`` and ``arg2 = -(x - center2)/sigma2``.

    Note that, unlike `step`,  ``sigma1 > 0`` and ``sigma2 > 0``, so that a
    rectangle can support a step up followed by a step down.
    Use a constant offset and adjust amplitude if that is what you need.

    See Also
    --------
    step

    """
    arg1 = (x - center1)/max(tiny, sigma1)
    arg2 = (center2 - x)/max(tiny, sigma2)

    if form == 'erf':
        out = 0.5*(erf(arg1) + erf(arg2))
    elif form == 'logistic':
        out = 1. - 1./(1. + exp(arg1)) - 1./(1. + exp(arg2))
    elif form in ('atan', 'arctan'):
        out = (arctan(arg1) + arctan(arg2))/pi
    elif form == 'linear':
        out = 0.5*(minimum(1, maximum(-1, arg1)) + minimum(1, maximum(-1, arg2)))
    else:
        msg = (f"Invalid value ('{form}') for argument 'form'; should be one "
               "of 'erf', 'logistic', 'atan', 'arctan', or 'linear'.")
        raise ValueError(msg)

    return amplitude*out


def exponential(x, amplitude=1, decay=1):
    """Return an exponential function.

    exponential(x, amplitude, decay) = amplitude * exp(-x/decay)

    """
    decay = not_zero(decay)
    return amplitude * exp(-x/decay)


def powerlaw(x, amplitude=1, exponent=1.0):
    """Return the powerlaw function.

    powerlaw(x, amplitude, exponent) = amplitude * x**exponent

    """
    return amplitude * x**exponent


def linear(x, slope=1.0, intercept=0.0):
    """Return a linear function.

    linear(x, slope, intercept) = slope * x + intercept

    """
    return slope * x + intercept


def parabolic(x, a=0.0, b=0.0, c=0.0):
    """Return a parabolic function.

    parabolic(x, a, b, c) = a * x**2 + b * x + c

    """
    return a * x**2 + b * x + c


def constant(x, c=0.0):
    """Return a constant array.

    constant(x, c) = c * ones_like(x)

    """
    return c * ones(shape(x))


def complex_constant(x, re=0.0, im=0.0):
    """Return a complex constant array.

    complex_constant(x, re, im) = (re + 1j*im) * ones_like(x)

    """
    return (re + 1j*im) * ones(shape(x))


def polynomial(x, c0=0.0, c1=0.0, c2=0.0, c3=0.0, c4=0.0, c5=0.0, c6=0.0,
               c7=0.0):
    """Return a polynomial function of degree up to 7.

    polynomial(x, c0, ..., c7) = c0 + c1*x + c2*x**2 + ... + c7*x**7

    """
    return polyval([c7, c6, c5, c4, c3, c2, c1, c0], x)


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
    """Return a 1-D B-spline, as used by `SplineModel`.

    spline_model(x, s0, ..., s99, knots, order) =
        splev(x, [knots, [s0, s1, ...], order])

    `knots` is the knot vector from `scipy.interpolate.splrep` and is
    required. `order` defaults to 3 (cubic).

    """
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
        raise ValueError("spline_model requires the spline 'knots'")
    if order is None:
        order = 3
    coefs = coefs[:len(knots)]
    coefs.extend([coefs[-1]]*(order+1))
    return splev(x, [knots, array(coefs), order])
