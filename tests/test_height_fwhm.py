#!/usr/bin/env python
from __future__ import (print_function)

from lmfit import lineshapes, models
import inspect
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve


def test_height_fwhm(lineshape, model):
    """Check height and fwhm parameters"""
    pars = model.guess(y, x=x)
    out = model.fit(y, pars, x=x)
    height = out.params['height'].value
    fwhm = out.params['fwhm'].value
    args = {key: out.best_values[key] for key in
            inspect.getargspec(lineshape)[0] if key is not 'x'}
    args.update({'center': 0})
    height2 = lineshape(0, **args)
    print(height, 'diff=', height - height2)
    func = lambda x:  lineshape(x, **args) - 0.5*height
    fwhm2 = fsolve(func, fwhm)[0]*2
    print(fwhm, 'diff=', fwhm - fwhm2)


if __name__ == '__main__':
    mu = 0
    variance = 1
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 6*sigma, mu + 6*sigma, 100)
    y = norm.pdf(x, mu, 1)
    print('voigt')
    test_height_fwhm(lineshapes.voigt, models.VoigtModel())
    print('pvoigt')
    test_height_fwhm(lineshapes.pvoigt, models.PseudoVoigtModel())
    print('pearson7')
    test_height_fwhm(lineshapes.pearson7, models.Pearson7Model())
