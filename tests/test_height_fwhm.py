#!/usr/bin/env python
from __future__ import (print_function)

from lmfit import lineshapes, models
import inspect
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from matplotlib import pyplot as plt


def test_height_fwhm(x, y, lineshape, model, plot=False, report=False):
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

        assert abs(diff) < 0.001, fmt.format(model._name,'Actual',
                'program', 'Diffrence', 'Height', height_pro, height_act, diff)

        if 'fwhm' in out.params:
            fwhm_pro = out.params['fwhm'].value
            func = lambda x:  lineshape(x, **args) - 0.5*height_act
            ret = fsolve(func, [cen - fwhm_pro/2, cen + fwhm_pro/2])
            fwhm_act = ret[1] - ret[0]
            diff = fwhm_act - fwhm_pro

            assert abs(diff) < 0.05, fmt.format(model._name,'Actual',
                    'program', 'Diffrence', 'FWHM', fwhm_pro, fwhm_act, diff)

    print(model._name, 'OK')
    if plot:
        fig = plt.figure()
        out.plot(fig=fig)
        plt.show()


if __name__ == '__main__':
    mu = 0
    variance = 1.0
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 20*sigma, mu + 20*sigma, 100.0)
    y = norm.pdf(x, mu, 1)
    test_height_fwhm(x, y, lineshapes.voigt, models.VoigtModel())
    test_height_fwhm(x, y, lineshapes.pvoigt, models.PseudoVoigtModel())
    test_height_fwhm(x, y, lineshapes.pearson7, models.Pearson7Model())
    test_height_fwhm(x, y, lineshapes.moffat, models.MoffatModel())
    test_height_fwhm(x, y, lineshapes.students_t, models.StudentsTModel())
    test_height_fwhm(x, y, lineshapes.breit_wigner, models.BreitWignerModel())
    test_height_fwhm(x+1, y, lineshapes.lognormal, models.LognormalModel())
    test_height_fwhm(x, y, lineshapes.damped_oscillator,
                     models.DampedOscillatorModel())
    test_height_fwhm(x+1, y, lineshapes.dho,
                     models.DampedHarmonicOscillatorModel())
    test_height_fwhm(x, y, lineshapes.expgaussian,
                     models.ExponentialGaussianModel())
    test_height_fwhm(x, y, lineshapes.skewed_gaussian,
                     models.SkewedGaussianModel())
    test_height_fwhm(x, y, lineshapes.donaich,
                     models.DonaichModel())
