"""Utility mathematical functions and common lineshapes for minimizer
"""
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import gamma

log2 = np.log(2)
pi = np.pi


def gaussian(x, height, center, sigma):
    "x -> height * exp(-(x - center)**2 / (const*sigma**2))"
    const = 2  # for future generalization to N dimensions
    return height * np.exp(-(x - center)**2 / (const*sigma**2))


def normalized_gaussian(x, center, sigma):
    "x -> 1/(sigma*sqrt(2*pi)) * exp(-(x - center)**2 / (const*sigma**2))"
    const = 2  # for future generalization to N dimensions
    normalization = 1/(sigma*np.sqrt(const*pi))
    return normalization * np.exp(-(x - center)**2 / (const*sigma**2))


def exponential(x, amplitude, decay):
    "x -> amplitude * exp(-x/decay)"
    return amplitude * np.exp(-x/decay)


def powerlaw(x, coefficient, exponent):
    "x -> coefficient * x**exponent"
    return coefficient * x**exponent


def linear(x, slope, intercept):
    "x -> slope * x + intercept"
    return slope * x + intercept


def parabolic(x, a, b, c):
    "x -> a * x**2 + b * x + c"
    return a * x**2 + b * x + c


def loren(x, amp, cen, wid):
    "lorentzian function: wid = half-width at half-max"
    return (amp / (1 + ((x-cen)/wid)**2))


def loren_area(x, amp, cen, wid):
    "scaled lorenztian function: wid = half-width at half-max"
    return loren(x, amp, cen, wid) / (pi*wid)


def pvoigt(x, amp, cen, wid, frac):
    """pseudo-voigt function:
    (1-frac)*gauss(amp, cen, wid) + frac*loren(amp, cen, wid)"""
    return amp * (gauss(x, (1-frac), cen, wid) +
                  loren(x, frac, cen, wid))


def pvoigt_area(x, amp, cen, wid, frac):
    """scaled pseudo-voigt function:
    (1-frac)*gauss_area(amp, cen, wid) + frac*loren_are(amp, cen, wid)"""

    return amp * (gauss_area(x, (1-frac), cen, wid) +
                  loren_area(x, frac,     cen, wid))


def pearson7(x, amp, cen, wid, expon):
    """pearson peak function """
    xp = 1.0 * expon
    return amp / (1 + (((x-cen)/wid)**2) * (2**(1/xp) - 1))**xp


def pearson7_area(x, amp, cen, wid, expon):
    """scaled pearson peak function """
    xp = 1.0 * expon
    scale = gamma(xp) * np.sqrt((2**(1/xp) - 1)) / (gamma(xp-0.5))
    return scale * pearson7(x, amp, cen, wid, xp) / (wid * np.sqrt(pi))


def assert_results_close(actual, desired, rtol=1e-03, atol=1e-03,
                         err_msg='', verbose=True):
    for param_name, value in desired.items():
        assert_allclose(actual[param_name], value, rtol, atol,
                        err_msg, verbose)
