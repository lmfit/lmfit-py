"""Basic analytical convolutions between lineshapes as defined in lmft built-ins."""

import numpy as np

from .lineshapes import (gaussian, lorentzian, pvoigt, voigt)


def conv_lorentzian_lorentzian(left, right, params, **kwargs):
    r"""Convolution between two Lorentzians.

    .. math::

        a_1 \mathcal{L}_{\sigma_1, center_1}
        \otimes a_2 \mathcal{L}_{sigma_2, center_2} =
            a_1 a_2 \mathcal{L}_{\sigma_1 + \sigma_2, center_1 + center_2}

    """
    lp = left.make_funcargs(params, **kwargs)
    rp = right.make_funcargs(params, **kwargs)
    amplitude = lp['amplitude'] * rp['amplitude']
    sigma = lp['sigma'] + rp['sigma']
    center = lp['center'] + rp['center']

    return lorentzian(lp['x'], amplitude, center, sigma)


def conv_delta_lorentzian(left, right, params, **kwargs):
    r"""Convolution between a Dirac delta and a Lorentzian.

    .. math::

        a_{\delta} \delta(x - center) \otimes a_L \mathcal{L}_{\sigma, center_L} =
            a_{\delta} a_L \mathcal{L}_{\sigma, center}

    """
    # set left to delta component if not so
    if left.func.__name__ == 'lorentzian':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, **kwargs)
    rp = right.make_funcargs(params, **kwargs)

    amplitude = lp['amplitude'] * rp['amplitude']
    sigma = rp['sigma']
    center = lp['center']

    return lorentzian(lp['x'], amplitude, center, sigma)


def conv_delta_gaussian(left, right, params, **kwargs):
    r"""Convolution between a Dirac delta and a Lorentzian.

    .. math::

        a_{\delta} \delta(x - center) \otimes a_G G_{\sigma, center_G} =
            a_{\delta} a_G G_{\sigma, center}

    """
    # set left to delta component if not so
    if left.func.__name__ == 'gaussian':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, **kwargs)
    rp = right.make_funcargs(params, **kwargs)

    amplitude = lp['amplitude'] * rp['amplitude']
    sigma = rp['sigma']
    center = lp['center']

    return gaussian(lp['x'], amplitude, center, sigma)


def conv_gaussian_lorentzian(left, right, params, **kwargs):
    """Convolution of a Gaussian and a Lorentzian.

    Results in a Voigt profile as defined in lineshapes.

    """
    # set left to Gaussian component if not so
    if left.func.__name__ == 'lorentzian':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, **kwargs)
    rp = right.make_funcargs(params, **kwargs)

    amplitude = lp['amplitude'] * rp['amplitude']
    center = rp['center'] + lp['center']
    sigma = lp['sigma']
    gamma = rp['sigma']

    return voigt(lp['x'], amplitude, center, sigma, gamma)


def conv_gaussian_gaussian(left, right, params, **kwargs):
    r"""Convolution between two Gaussians.

    .. math::

        a_1 G_{\sigma_1, center_1} \otimes a_2 G_{\sigma_2, center_2} =
            a_1 * a_2 G_{\sigma_1 + \sigma2, center_1 + center_2}

    """
    lp = left.make_funcargs(params, **kwargs)
    rp = right.make_funcargs(params, **kwargs)

    amplitude = lp['amplitude'] * rp['amplitude']
    center = lp['center'] + rp['center']
    sigma = lp['sigma'] + rp['sigma']

    return gaussian(lp['x'], amplitude, center, sigma)


def conv_delta_pvoigt(left, right, params, **kwargs):
    r"""Convolution between a Dirac delta and a pseudo-Voigt profile.

    .. math::

        a_1 \delta(x - center) \otimes
        a_2 p\mathcal{V}_{\sigma, center_{\mathcal{V}}, fraction} =
            a_1 a_2 p\mathcal{V}_{\sigma, center, fraction}

    """
    # set left to delta component if not so
    if left.func.__name__ == 'pvoigt':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, **kwargs)
    rp = right.make_funcargs(params, **kwargs)

    amplitude = lp['amplitude'] * rp['amplitude']
    center = lp['center']
    sigma = rp['sigma']
    fraction = rp['fraction']

    return pvoigt(lp['x'], amplitude, center, sigma, fraction)


def conv_lorentzian_pvoigt(left, right, params, **kwargs):
    """Convolution between a Lorentzian and a pseudo-Voigt profile.

    .. math::

        a_L \\mathcal{L}_{\\sigma_L, center_L} \\otimes
        a_V p\\mathcal{V}_{\\sigma, center, fraction} =
            a_L a_V \\left[ (1 - fraction)
                \\mathcal{V}_{\\sigma_g, \\sigma, center + center_L}
                + fraction
                    \\mathcal{L}_{\\sigma + \\sigma_L, center + center_L}
            \\right]

    where :math:`p\\mathcal{V}` is the pseudo-Voigt, :math:`\\mathcal{V}` is
    a Voigt profile, :math:`\\sigma_g = \\frac{\\sigma}{\\sqrt(2 log(2)} and
    :math:`\\mathcal{L}` is a Lorentzian.

    """
    # set left to lorentzian component if not so
    if left.func.__name__ == 'pvoigt':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, **kwargs)
    rp = right.make_funcargs(params, **kwargs)

    amplitude = lp['amplitude'] * rp['amplitude']
    center = lp['center'] + rp['center']
    gamma = lp['sigma']
    sigma_l = lp['sigma'] + rp['sigma']
    sigma_v = rp['sigma'] / np.sqrt(2 * np.log(2))
    fraction = rp['fraction']

    x = lp['x']

    out = ((1 - fraction) * voigt(x, amplitude, center, sigma_v, gamma)
           + fraction * lorentzian(x, amplitude, center, sigma_l))

    return out


def conv_gaussian_pvoigt(left, right, params, **kwargs):
    """Convolution between a Gaussian and a pseudo-Voigt profile.

    .. math::

        a_L G_{\\sigma_G, center_G} \\otimes
        a_V p\\mathcal{V}_{\\sigma, center, fraction} =
            a_G a_V \\left[fraction
                \\mathcal{V}_{\\sigma_G, \\sigma, center + center_G}
                + (1 - fraction)
                    G_{\\sigma_g + \\sigma_G, center + center_G}
            \\right]

    where :math:`p\\mathcal{V}` is the pseudo-Voigt, :math:`\\mathcal{V}` is
    a Voigt profile, and :math:`\\sigma_g = \\frac{\\sigma}{\\sqrt(2 log(2)}.

    """
    # set left to gaussian component if not so
    if left.func.__name__ == 'pvoigt':
        tmp = right
        right = left
        left = tmp

    lp = left.make_funcargs(params, **kwargs)
    rp = right.make_funcargs(params, **kwargs)

    amplitude = lp['amplitude'] * rp['amplitude']
    center = lp['center'] + rp['center']
    gamma = rp['sigma']
    sigma_G = lp['sigma'] + rp['sigma'] / np.sqrt(2 * np.log(2))
    sigma_V = lp['sigma']
    fraction = rp['fraction']

    x = lp['x']

    out = (fraction * voigt(x, amplitude, center, sigma_V, gamma)
           + (1 - fraction) * gaussian(x, amplitude, center, sigma_G))

    return out
