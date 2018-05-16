#!/usr/bin/env python
from __future__ import (print_function)

from lmfit import lineshapes, models
import inspect
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import sys

import os


def check_height_fwhm(x, y, lineshape, model, with_plot=False, report=False):
    """Check height and fwhm parameters"""
    pars = model.guess(y, x=x)
    out = model.fit(y, pars, x=x)
    if report:
        print(out.fit_report())

    # account for functions whose centers are not mu
    mu = out.params['center'].value
    if lineshape is lineshapes.lognormal:
        cen = np.exp(mu - out.params['sigma']**2)
    else:
        cen = mu
    # get arguments for lineshape
    args = {key: out.best_values[key] for key in
            inspect.getargspec(lineshape)[0] if key is not 'x'}
    # output format for assertion errors
    fmt = ("Program calculated values and real values do not match!\n"
           "{:^20s}{:^20s}{:^20s}{:^20s}\n"
           "{:^20s}{:^20f}{:^20f}{:^20f}")

    if 'height' in out.params:
        height_pro = out.params['height'].value
        height_act = lineshape(cen, **args)
        diff = height_act - height_pro

        assert abs(diff) < 0.001, fmt.format(model._name, 'Actual',
                'program', 'Diffrence', 'Height', height_act, height_pro, diff)

        if 'fwhm' in out.params:
            fwhm_pro = out.params['fwhm'].value
            func = lambda x:  lineshape(x, **args) - 0.5*height_act
            ret = fsolve(func, [cen - fwhm_pro/4, cen + fwhm_pro/2])
            # print(ret)
            fwhm_act = ret[1] - ret[0]
            diff = fwhm_act - fwhm_pro

            assert abs(diff) < 0.5, fmt.format(model._name, 'Actual',
                    'program', 'Diffrence', 'FWHM', fwhm_act, fwhm_pro, diff)

    print(model._name, 'OK')

def test_peak_like():
    # mu = 0
    # variance = 1.0
    # sigma = np.sqrt(variance)
    # x = np.linspace(mu - 20*sigma, mu + 20*sigma, 100.0)
    # y = norm.pdf(x, mu, 1)
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), '..', 'examples', 'test_peak.dat'))
    x = data[:, 0]
    y = data[:, 1]
    check_height_fwhm(x, y, lineshapes.voigt, models.VoigtModel())
    check_height_fwhm(x, y, lineshapes.pvoigt, models.PseudoVoigtModel())
    check_height_fwhm(x, y, lineshapes.pearson7, models.Pearson7Model())
    check_height_fwhm(x, y, lineshapes.moffat, models.MoffatModel())
    check_height_fwhm(x, y, lineshapes.students_t, models.StudentsTModel())
    check_height_fwhm(x, y, lineshapes.breit_wigner, models.BreitWignerModel())
    check_height_fwhm(x, y, lineshapes.damped_oscillator, models.DampedOscillatorModel())
    check_height_fwhm(x, y, lineshapes.dho, models.DampedHarmonicOscillatorModel())
    check_height_fwhm(x, y, lineshapes.expgaussian, models.ExponentialGaussianModel())
    check_height_fwhm(x, y, lineshapes.skewed_gaussian, models.SkewedGaussianModel())
    check_height_fwhm(x, y, lineshapes.donaich, models.DonaichModel())
    x=x-9 # Lognormal will only fit peaks with centers < 1
    check_height_fwhm(x, y, lineshapes.lognormal, models.LognormalModel())

if __name__ == '__main__':
    test_peak_like()
