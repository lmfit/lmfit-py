"""Utility mathematical functions and common lineshapes for minimizer
"""
import numpy
import scipy
from scipy.special import gamma

CUSTOM_FUNCTIONS = {}

log2 = numpy.log(2)
pi = numpy.pi

def gauss(x, amp, cen, wid):
    "gaussian function: wid = half-width at half-max"
    return amp * numpy.exp(-log2 * (x-cen) **2 / wid**2)

def loren(x, amp, cen, wid):
    "lorenztian function: wid = half-width at half-max"
    return (amp  / (1 + ((x-cen)/wid)**2))

def gauss_area(x, amp, cen, wid):
    "scaled gaussian function: wid = half-width at half-max"
    return numpy.sqrt(log2/pi) * gauss(x, amp, cen, wid) / wid

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
    return amp / (1 + ( ((x-cen)/wid)**2) * (2**(1/xp) -1) )**xp


def pearson7_area(x, amp, cen, wid, expon):
    """scaled pearson peak function """
    xp = 1.0 * expon
    scale = gamma(xp) * sqrt((2**(1/xp) - 1)) / (gamma(xp-0.5))
    return scale * pearson7(x, amp, cen, wid, xp) / (wid*sqrt(pi))

    return scale * pearson7(x, amp, cen, sigma, expon) / (sigma*sqrt(pi))

CUSTOM_FUNCTIONS = {'gauss': gauss, 'gauss_area': gauss_area,
                    'loren': loren, 'loren_area': loren_area,
                    'pvoigt': pvoigt, 'pvoigt_area': pvoigt_area,
                    'pearson7': pearson7, 'pearson7_area': pearson7_area}


