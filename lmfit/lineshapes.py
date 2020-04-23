"""Basic model line shapes and distribution functions."""

from numpy import (arctan, cos, exp, finfo, float64, isnan, log, pi, real, sin,
                   sqrt, where)
from scipy.special import erf, erfc
from scipy.special import gamma as gamfcn
from scipy.special import wofz

log2 = log(2)
s2pi = sqrt(2*pi)
spi = sqrt(pi)
s2 = sqrt(2.0)
tiny = finfo(float64).eps

functions = ('gaussian', 'lorentzian', 'voigt', 'pvoigt', 'moffat', 'pearson7',
             'breit_wigner', 'damped_oscillator', 'dho', 'logistic', 'lognormal',
             'students_t', 'expgaussian', 'doniach', 'donaich', 'skewed_gaussian',
             'skewed_voigt', 'thermal_distribution', 'step', 'rectangle',
             'exponential', 'powerlaw', 'linear', 'parabolic', 'sine',
             'expsine', 'split_lorentzian')


def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * exp(-(1.0*x-center)**2 / max(tiny, (2*sigma**2))))


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
    from the condition that the integral from ``-.inf`` to ``+.inf`` is equal
    to ``amplitude``.

    split_lorentzian(x, amplitude, center, sigma, sigma_r) =
        [2*amplitude / (pi* (sigma + sigma_r)] *
         {  sigma**2   * (x<center)  / [sigma**2  + (x - center)**2]
          + sigma_r**2 * (x>=center) / [sigma_r**2+ (x - center)**2] }

    """
    s = max(tiny, sigma)
    r = max(tiny, sigma_r)
    s2 = s*s
    r2 = r*r
    xc2 = (x-center)**2
    amp = 2*amplitude/(pi*(s+r))
    return amp*(s2*(x < center)/(s2+xc2) + r2*(x >= center)/(r2+xc2))


def voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None):
    """Return a 1-dimensional Voigt function.

    voigt(x, amplitude, center, sigma, gamma) =
        amplitude*wofz(z).real / (sigma*s2pi)

    see https://en.wikipedia.org/wiki/Voigt_profile

    """
    if gamma is None:
        gamma = sigma
    z = (x-center + 1j*gamma) / max(tiny, (sigma*s2))
    return amplitude*wofz(z).real / max(tiny, (sigma*s2pi))


def pvoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5):
    """Return a 1-dimensional pseudo-Voigt function.

    pvoigt(x, amplitude, center, sigma, fraction) =
       amplitude*(1-fraction)*gaussian(x, center, sigma_g) +
       amplitude*fraction*lorentzian(x, center, sigma)

    where sigma_g (the sigma for the Gaussian component) is

        sigma_g = sigma / sqrt(2*log(2)) ~= sigma / 1.17741

    so that the Gaussian and Lorentzian components have the
    same FWHM of 2*sigma.

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


def pearson7(x, amplitude=1.0, center=0.0, sigma=1.0, expon=1.0):
    """Return a Pearson7 lineshape.

    Using the wikipedia definition:
    pearson7(x, center, sigma, expon) =
        amplitude*(1+arg**2)**(-expon)/(sigma*beta(expon-0.5, 0.5))

    where arg = (x-center)/sigma
    and beta() is the beta function.

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


def dho(x, amplitude=1., center=0., sigma=1., gamma=1.0):
    """Return a Damped Harmonic Oscillator.

    Similar to version from PAN
    dho(x, amplitude, center, sigma, gamma) =
        amplitude*sigma*pi * (lm - lp) / (1.0 - exp(-x/gamma))

    where
        lm(x, center, sigma) = 1.0 / ((x-center)**2 + sigma**2)
        lp(x, center, sigma) = 1.0 / ((x+center)**2 + sigma**2)

    """
    bose = (1.0 - exp(-x/max(tiny, gamma)))
    if isinstance(bose, (int, float)):
        bose = max(tiny, bose)
    else:
        bose[where(isnan(bose))] = tiny
        bose[where(bose <= tiny)] = tiny

    lm = 1.0/((x-center)**2 + sigma**2)
    lp = 1.0/((x+center)**2 + sigma**2)
    return amplitude*sigma/pi*(lm - lp)/bose


def logistic(x, amplitude=1., center=0., sigma=1.):
    """Return a Logistic lineshape (yet another sigmoidal curve).

    logistic(x, amplitude, center, sigma) =
        = amplitude*(1.  - 1. / (1 + exp((x-center)/sigma)))

    """
    return amplitude*(1. - 1./(1. + exp((x-center)/max(tiny, sigma))))


def lognormal(x, amplitude=1.0, center=0., sigma=1):
    """Return a log-normal function.

    lognormal(x, amplitude, center, sigma)
        = (amplitude/(x*sigma*s2pi)) * exp(-(ln(x) - center)**2/ (2* sigma**2))

    """
    if isinstance(x, (int, float)):
        x = max(tiny, x)
    else:
        x[where(x <= tiny)] = tiny
    return ((amplitude/(x*max(tiny, sigma*s2pi))) *
            exp(-(log(x)-center)**2 / max(tiny, (2*sigma**2))))


def students_t(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return Student's t distribution.

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

    expgaussian(x, amplitude, center, sigma, gamma)
        = amplitude * (gamma/2) *
          exp[center*gamma + (gamma*sigma)**2/2 - gamma*x] *
          erfc[(center + gamma*sigma**2 - x)/(sqrt(2)*sigma)]

    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

    """
    gss = gamma*sigma*sigma
    arg1 = gamma*(center + gss/2.0 - x)
    arg2 = (center + gss - x)/max(tiny, (s2*sigma))
    return amplitude*(gamma/2) * exp(arg1) * erfc(arg2)


def doniach(x, amplitude=1.0, center=0, sigma=1.0, gamma=0.0):
    """Return a Doniach Sunjic asymmetric lineshape, used for photo-emission.

    doniach(x, amplitude, center, sigma, gamma) =
        amplitude / sigma^(1-gamma) *
        cos(pi*gamma/2 + (1-gamma) arctan((x-center)/sigma) /
        (sigma**2 + (x-center)**2)**[(1-gamma)/2]

    see http://www.casaxps.com/help_manual/line_shapes.htm

    """
    arg = (x-center)/max(tiny, sigma)
    gm1 = (1.0 - gamma)
    scale = amplitude/max(tiny, (sigma**gm1))
    return scale*cos(pi*gamma/2 + gm1*arctan(arg))/(1 + arg**2)**(gm1/2)


donaich = doniach   # for back-compat


def skewed_gaussian(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=0.0):
    """Return a Gaussian lineshape, skewed with error function.

    Equal to: gaussian(x, center, sigma)*(1+erf(beta*(x-center)))

    with beta = gamma/(sigma*sqrt(2))

    with  gamma < 0:  tail to low value of centroid
          gamma > 0:  tail to high value of centroid

    see https://en.wikipedia.org/wiki/Skew_normal_distribution

    """
    asym = 1 + erf(gamma*(x-center)/max(tiny, (s2*sigma)))
    return asym * gaussian(x, amplitude, center, sigma)


def skewed_voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None, skew=0.0):
    """Return a Voigt lineshape, skewed with error function.

    useful for ad-hoc Compton scatter profile

    with beta = skew/(sigma*sqrt(2))
    = voigt(x, center, sigma, gamma)*(1+erf(beta*(x-center)))

    skew < 0:  tail to low value of centroid
    skew > 0:  tail to high value of centroid

    see https://en.wikipedia.org/wiki/Skew_normal_distribution

    """
    beta = skew/max(tiny, (s2*sigma))
    asym = 1 + erf(beta*(x-center))
    return asym * voigt(x, amplitude, center, sigma, gamma=gamma)


def sine(x, amplitude=1.0, frequency=1.0, shift=0.0):
    """Return a sinusoidal function.

    sine(x, amplitude, frequency, shift):
       = amplitude * sin(x*frequency + shift)

    """
    return amplitude*sin(x*frequency + shift)


def expsine(x, amplitude=1.0, frequency=1.0, shift=0.0, decay=0.0):
    """Return an exponentially decaying sinusoidal function.

    expsine(x, amplitude, frequency, shift,  decay):
       = amplitude * sin(x*frequency + shift) * exp(-x*decay)

    """
    return amplitude*sin(x*frequency + shift) * exp(-x*decay)


def thermal_distribution(x, amplitude=1.0, center=0.0, kt=1.0, form='bose'):
    """Return a thermal distribution function.

    form = 'bose' (default) is the Bose-Einstein distribution
    thermal_distribution(x, amplitude=1.0, center=0.0, kt=1.0):
       = 1/(amplitude*exp((x - center)/kt) - 1)
    form = 'maxwell' is the Maxwell-Boltzmann distribution
    thermal_distribution(x, amplitude=1.0, center=0.0, kt=1.0):
       = 1/(amplitude*exp((x - center)/kt))
    form = 'fermi' is the Fermi-Dirac distribution
    thermal_distribution(x, amplitude=1.0, center=0.0, kt=1.0):
       = 1/(amplitude*exp((x - center)/kt) + 1)

    Notes:
    - ``kt`` should be defined in the same units as ``x``. (The Boltzmann
      constant is kB = 8.617e-5 eV/K).
    - set ``kt<0`` to implement the energy loss convention common in scattering
      research.

    see http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/disfcn.html

    """
    form = form.lower()
    if form.startswith('bose'):
        offset = -1
    elif form.startswith('maxwell'):
        offset = 0
    elif form.startswith('fermi'):
        offset = 1
    else:
        msg = "Invalid value ('%s') for argument 'form'; should be one of %s."\
              % (form, "'maxwell', 'fermi', or 'bose'")
        raise ValueError(msg)

    return real(1/(amplitude*exp((x - center)/kt) + offset + tiny*1j))


def step(x, amplitude=1.0, center=0.0, sigma=1.0, form='linear'):
    """Return a step function.

    starts at 0.0, ends at amplitude, with half-max at center, and
    rising with form:
      'linear' (default) = amplitude * min(1, max(0, arg))
      'atan', 'arctan'   = amplitude * (0.5 + atan(arg)/pi)
      'erf'              = amplitude * (1 + erf(arg))/2.0
      'logistic'         = amplitude * [1 - 1/(1 + exp(arg))]

    where arg = (x - center)/sigma

    """
    out = (x - center)/max(tiny, sigma)

    if form == 'erf':
        out = 0.5*(1 + erf(out))
    elif form == 'logistic':
        out = (1. - 1./(1. + exp(out)))
    elif form in ('atan', 'arctan'):
        out = 0.5 + arctan(out)/pi
    elif form == 'linear':
        out[where(out < 0)] = 0.0
        out[where(out > 1)] = 1.0
    else:
        msg = "Invalid value ('%s') for argument 'form'; should be one of %s."\
               % (form, "'erf', 'logistic', 'atan', 'arctan', or 'linear'")
        raise ValueError(msg)

    return amplitude*out


def rectangle(x, amplitude=1.0, center1=0.0, sigma1=1.0,
              center2=1.0, sigma2=1.0, form='linear'):
    """Return a rectangle function: step up, step down.

    (see step function)
    starts at 0.0, rises to amplitude (at center1 with width sigma1)
    then drops to 0.0 (at center2 with width sigma2) with form:
      'linear' (default) = ramp_up + ramp_down
      'atan', 'arctan'   = amplitude*(atan(arg1) + atan(arg2))/pi
      'erf'              = amplitude*(erf(arg1) + erf(arg2))/2.
      'logisitic'        = amplitude*[1 - 1/(1 + exp(arg1)) - 1/(1+exp(arg2))]

    where arg1 =  (x - center1)/sigma1
    and   arg2 = -(x - center2)/sigma2

    """
    arg1 = (x - center1)/max(tiny, sigma1)
    arg2 = (center2 - x)/max(tiny, sigma2)

    if form == 'erf':
        out = 0.5*(erf(arg1) + erf(arg2))
    elif form == 'logistic':
        out = (1. - 1./(1. + exp(arg1)) - 1./(1. + exp(arg2)))
    elif form in ('atan', 'arctan'):
        out = (arctan(arg1) + arctan(arg2))/pi
    elif form == 'linear':
        arg1[where(arg1 < 0)] = 0.0
        arg1[where(arg1 > 1)] = 1.0
        arg2[where(arg2 > 0)] = 0.0
        arg2[where(arg2 < -1)] = -1.0
        out = arg1 + arg2
    else:
        msg = "Invalid value ('%s') for argument 'form'; should be one of %s."\
               % (form, "'erf', 'logistic', 'atan', 'arctan', or 'linear'")
        raise ValueError(msg)

    return amplitude*out


def exponential(x, amplitude=1, decay=1):
    """Return an exponential function.

    x -> amplitude * exp(-x/decay)

    """
    decay = max(tiny, decay)
    return amplitude * exp(-x/decay)


def powerlaw(x, amplitude=1, exponent=1.0):
    """Return the powerlaw function.

    x -> amplitude * x**exponent

    """
    return amplitude * x**exponent


def linear(x, slope=1.0, intercept=0.0):
    """Return a linear function.

    x -> slope * x + intercept

    """
    return slope * x + intercept


def parabolic(x, a=0.0, b=0.0, c=0.0):
    """Return a parabolic function.

    x -> a * x**2 + b * x + c

    """
    return a * x**2 + b * x + c
