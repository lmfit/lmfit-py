#!/usr/bin/env python
"""
basic model line shapes and distribution functions
"""
from __future__ import division
from numpy import (pi, log, exp, sqrt, arctan, cos, where)
from numpy.testing import assert_allclose

from scipy.special import gamma as gamfcn
from scipy.special import gammaln, erf, erfc, wofz

log2 = log(2)
s2pi = sqrt(2*pi)
spi  = sqrt(pi)
s2   = sqrt(2.0)

functions = ('gaussian', 'lorentzian', 'voigt', 'pvoigt', 'pearson7',
             'breit_wigner', 'damped_oscillator', 'logistic', 'lognormal',
             'students_t', 'expgaussian', 'donaich', 'skewed_gaussian',
             'skewed_voigt', 'step', 'rectangle', 'erf', 'erfc', 'wofz',
             'gamma', 'gammaln', 'exponential', 'powerlaw', 'linear',
             'parabolic')

def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """1 dimensional gaussian:
    gaussian(x, amplitude, center, sigma)
    """
    return (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 /(2*sigma**2))

def lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """1 dimensional lorentzian
    lorentzian(x, amplitude, center, sigma)
    """
    return (amplitude/(1 + ((1.0*x-center)/sigma)**2) ) / (pi*sigma)

def voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    if gamma is None:
        gamma = sigma
    z = (x-center + 1j*gamma)/ (sigma*s2)
    return amplitude*wofz(z).real / (sigma*s2pi)

def pvoigt(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5):
    """1 dimensional pseudo-voigt:
    pvoigt(x, amplitude, center, sigma, fraction)
       = amplitude*(1-fraction)*gaussion(x, center, sigma_g) +
         amplitude*fraction*lorentzian(x, center, sigma)

    where sigma_g (the sigma for the Gaussian component) is

        sigma_g = sigma / sqrt(2*log(2)) ~= sigma / 1.17741

    so that the Gaussian and Lorentzian components have the
    same FWHM of 2*sigma.
    """
    sigma_g = sigma / sqrt(2*log2)
    return ((1-fraction)*gaussian(x, amplitude, center, sigma_g) +
               fraction*lorentzian(x, amplitude, center, sigma))

def pearson7(x, amplitude=1.0, center=0.0, sigma=1.0, expon=1.0):
    """pearson7 lineshape, using the wikipedia definition:

    pearson7(x, center, sigma, expon) =
      amplitude*(1+arg**2)**(-expon)/(sigma*beta(expon-0.5, 0.5))

    where arg = (x-center)/sigma
    and beta() is the beta function.
    """
    arg = (x-center)/sigma
    scale = amplitude * gamfcn(expon)/(gamfcn(0.5)*gamfcn(expon-0.5))
    return  scale*(1+arg**2)**(-expon)/sigma

def breit_wigner(x, amplitude=1.0, center=0.0, sigma=1.0, q=1.0):
    """Breit-Wigner-Fano lineshape:
       = amplitude*(q*sigma/2 + x - center)**2 / ( (sigma/2)**2 + (x - center)**2 )
    """
    gam = sigma/2.0
    return  amplitude*(q*gam + x - center)**2 / (gam*gam + (x-center)**2)

def damped_oscillator(x, amplitude=1.0, center=1., sigma=0.1):
    """amplitude for a damped harmonic oscillator
    amplitude/sqrt( (1.0 - (x/center)**2)**2 + (2*sigma*x/center)**2))
    """
    center = max(1.e-9, abs(center))
    return (amplitude/sqrt( (1.0 - (x/center)**2)**2 + (2*sigma*x/center)**2))

def logistic(x, amplitude=1., center=0., sigma=1.):
    """Logistic lineshape (yet another sigmoidal curve)
        = amplitude*(1.  - 1. / (1 + exp((x-center)/sigma)))
    """
    return amplitude*(1. - 1./(1. + exp((x-center)/sigma)))

def lognormal(x, amplitude=1.0, center=0., sigma=1):
    """log-normal function
    lognormal(x, center, sigma)
        = (amplitude/x) * exp(-(ln(x) - center)/ (2* sigma**2))
    """
    x[where(x<=1.e-19)] = 1.e-19
    return (amplitude/(x*sigma*s2pi)) * exp(-(log(x)-center)**2/ (2* sigma**2))

def students_t(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Student's t distribution:
        gamma((sigma+1)/2)   (1 + (x-center)**2/sigma)^(-(sigma+1)/2)
     =  -------------------------
        sqrt(sigma*pi)gamma(sigma/2)

    """
    s1  = (sigma+1)/2.0
    denom = (sqrt(sigma*pi)*gamfcn(sigma/2))
    return amplitude*(1 + (x-center)**2/sigma)**(-s1) * gamfcn(s1) / denom


def expgaussian(x, amplitude=1, center=0, sigma=1.0, gamma=1.0):
    """exponentially modified Gaussian

    = (gamma/2) exp[center*gamma + (gamma*sigma)**2/2 - gamma*x] *
                erfc[(center + gamma*sigma**2 - x)/(sqrt(2)*sigma)]

    http://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    """
    gss = gamma*sigma*sigma
    arg1 = gamma*(center +gss/2.0 - x)
    arg2 = (center + gss - x)/(s2*sigma)
    return amplitude*(gamma/2) * exp(arg1) * erfc(arg2)

def donaich(x, amplitude=1.0, center=0, sigma=1.0, gamma=0.0):
    """Doniach Sunjic asymmetric lineshape, used for photo-emission

    = amplitude* cos(pi*gamma/2 + (1-gamma) arctan((x-center)/sigma) /
                      (sigma**2 + (x-center)**2)**[(1-gamma)/2]

    see http://www.casaxps.com/help_manual/line_shapes.htm
    """
    arg = (x-center)/sigma
    gm1 = (1.0 - gamma)
    scale = amplitude/(sigma**gm1)
    return scale*cos(pi*gamma/2 + gm1*arctan(arg))/(1 + arg**2)**(gm1/2)

def skewed_gaussian(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=0.0):
    """Gaussian, skewed with error function, equal to

     gaussian(x, center, sigma)*(1+erf(beta*(x-center)))

    with beta = gamma/(sigma*sqrt(2))

    with  gamma < 0:  tail to low value of centroid
          gamma > 0:  tail to high value of centroid

    see http://en.wikipedia.org/wiki/Skew_normal_distribution
    """
    asym = 1 + erf(gamma*(x-center)/(s2*sigma))
    return asym * gaussian(x, amplitude, center, sigma)

def skewed_voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None, skew=0.0):
    """Skewed Voigt lineshape, skewed with error function
    useful for ad-hoc Compton scatter profile

    with beta = skew/(sigma*sqrt(2))
    = voigt(x, center, sigma, gamma)*(1+erf(beta*(x-center)))

    skew < 0:  tail to low value of centroid
    skew > 0:  tail to high value of centroid

    see http://en.wikipedia.org/wiki/Skew_normal_distribution
    """
    beta = skew/(s2*sigma)
    asym = 1 + erf(beta*(x-center))
    return asym * voigt(x, amplitude, center, sigma, gamma=gamma)

def step(x, amplitude=1.0, center=0.0, sigma=1.0, form='linear'):
    """step function:
    starts at 0.0, ends at amplitude, with half-max at center, and
    rising with form:
      'linear' (default) = amplitude * min(1, max(0, arg))
      'atan', 'arctan'   = amplitude * (0.5 + atan(arg)/pi)
      'erf'              = amplitude * (1 + erf(arg))/2.0
      'logistic'         = amplitude * [1 - 1/(1 + exp(arg))]

    where arg = (x - center)/sigma
    """
    if abs(sigma) <  1.e-13:
        sigma = 1.e-13

    out = (x - center)/sigma
    if form == 'erf':
        out = 0.5*(1 + erf(out))
    elif form.startswith('logi'):
        out = (1. - 1./(1. + exp(out)))
    elif form in ('atan', 'arctan'):
        out = 0.5 + arctan(out)/pi
    else:
        out[where(out < 0)] = 0.0
        out[where(out > 1)] = 1.0
    return amplitude*out

def rectangle(x, amplitude=1.0, center1=0.0, sigma1=1.0,
              center2=1.0, sigma2=1.0, form='linear'):
    """rectangle function: step up, step down  (see step function)
    starts at 0.0, rises to amplitude (at center1 with width sigma1)
    then drops to 0.0 (at center2 with width sigma2) with form:
      'linear' (default) = ramp_up + ramp_down
      'atan', 'arctan'   = amplitude*(atan(arg1) + atan(arg2))/pi
      'erf'              = amplitude*(erf(arg1) + erf(arg2))/2.
      'logisitic'        = amplitude*[1 - 1/(1 + exp(arg1)) - 1/(1+exp(arg2))]

    where arg1 =  (x - center1)/sigma1
    and   arg2 = -(x - center2)/sigma2
    """
    if abs(sigma1) <  1.e-13:
        sigma1 = 1.e-13
    if abs(sigma2) <  1.e-13:
        sigma2 = 1.e-13

    arg1 = (x - center1)/sigma1
    arg2 = (center2 - x)/sigma2
    if form == 'erf':
        out = 0.5*(erf(arg1) + erf(arg2))
    elif form.startswith('logi'):
        out = (1. - 1./(1. + exp(arg1)) - 1./(1. + exp(arg2)))
    elif form in ('atan', 'arctan'):
        out = (arctan(arg1) + arctan(arg2))/pi
    else:
        arg1[where(arg1 <  0)]  = 0.0
        arg1[where(arg1 >  1)]  = 1.0
        arg2[where(arg2 >  0)]  = 0.0
        arg2[where(arg2 < -1)] = -1.0
        out = arg1 + arg2
    return amplitude*out

def _erf(x):
    """error function.  = 2/sqrt(pi)*integral(exp(-t**2), t=[0, z])"""
    return erf(x)

def _erfc(x):
    """complented error function.  = 1 - erf(x)"""
    return erfc(x)

def _wofz(x):
    """fadeeva function for complex argument. = exp(-x**2)*erfc(-i*x)"""
    return wofz(x)

def _gamma(x):
    """gamma function"""
    return gamfcn(x)

def _gammaln(x):
    """log of absolute value of gamma function"""
    return gammaln(x)


def exponential(x, amplitude=1, decay=1):
    "x -> amplitude * exp(-x/decay)"
    return amplitude * exp(-x/decay)


def powerlaw(x, amplitude=1, exponent=1.0):
    "x -> amplitude * x**exponent"
    return amplitude * x**exponent


def linear(x, slope, intercept):
    "x -> slope * x + intercept"
    return slope * x + intercept


def parabolic(x, a, b, c):
    "x -> a * x**2 + b * x + c"
    return a * x**2 + b * x + c


def assert_results_close(actual, desired, rtol=1e-03, atol=1e-03,
                         err_msg='', verbose=True):
    """returns whether all parameter values in actual are close to
    those in desired"""
    for param_name, value in desired.items():
        assert_allclose(actual[param_name], value, rtol,
                        atol, err_msg, verbose)
