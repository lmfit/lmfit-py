#!/usr/bin/env python
"""
Some common lineshapes and distribution functions
"""

from numpy import (pi, log, exp, sqrt, arctan, cos, arange,
                   concatenate, convolve)
from numpy.testing import assert_allclose

from scipy.special import gamma, gammaln, beta, betaln, erf, erfc, wofz

log2 = log(2)
s2pi = sqrt(2*pi)
spi  = sqrt(pi)
s2   = sqrt(2.0)

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
       = amplitude*(1-fraction)*gaussion(x, center,sigma) +
         amplitude*fraction*lorentzian(x, center, sigma)
    """
    return ((1-fraction)*gaussian(x, amplitude, center, sigma) +
                fraction*lorentzian(x, amplitude, center, sigma))

def pearson7(x, amplitude=1.0, center=0.0, sigma=1.0, expon=0.5):
    """pearson7 lineshape, according to NIST StRD
    though it seems wikpedia gives a different formula...
    pearson7(x, center, sigma, expon)
    """
    scale = amplitude * gamma(expon) * (sqrt((2**(1/expon) -1)) /
                                        (gamma(expon-0.5)) / (sigma*spi))
    return scale / (1 + ( ((1.0*x-center)/sigma)**2) * (2**(1/expon) -1) )**expon

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
    return (amplitude/(x*sigma*s2pi)) * exp(-(log(x) - center)**2/ (2* sigma**2))

def students_t(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Student's t distribution:
        gamma((sigma+1)/2)   (1 + (x-center)**2/sigma)^(-(sigma+1)/2)
     =  -------------------------
        sqrt(sigma*pi)gamma(sigma/2)

    """
    s1  = (sigma+1)/2.0
    denom = (sqrt(sigma*pi)*gamma(sigma/2))
    return amplitude*(1 + (x-center)**2/sigma)**(-s1) * gamma(s1) / denom


def exgaussian(x, amplitude=1, center=0, sigma=1.0, gamma=1.0):
    """exponentially modified Gaussian

    = (gamma/2) exp[center*gamma + (gamma*sigma)**2/2 - gamma*x] *
                erfc[(center + gamma*sigma**2 - x)/(sqrt(2)*sigma)]

    http://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    """
    gss = gamma*sigma*sigma
    arg1 = gamma*(center +gss/2.0 - x)
    arg2 = (center + gss - x)/s2
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
    return (1 + erf(beta*(x-center)))*voigt(x, amplitude, center, sigma, gamma=gamma)

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
    return gamma(x)

def _gammaln(x):
    """log of absolute value of gamma function"""
    return gammaln(x)



normalized_gaussian = gaussian


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


def assert_results_close(actual, desired, rtol=1e-03, atol=1e-03,b
                         err_msg='', verbose=True):
    for param_name, value in desired.items():
        print 'Assert ', param_name, actual[param_name], value
        assert_allclose(actual[param_name], value, rtol, atol, err_msg, verbose)
