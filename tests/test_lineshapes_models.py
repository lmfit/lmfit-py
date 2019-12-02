"""Tests for lineshape functions and built-in models."""

import inspect
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import fsolve

from lmfit import lineshapes, models

if sys.version_info[0] == 2:
    inspect_args = inspect.getargspec
elif sys.version_info[0] == 3:
    inspect_args = inspect.getfullargspec


def check_height_fwhm(x, y, lineshape, model):
    """Check height and fwhm parameters."""
    pars = model.guess(y, x=x)
    out = model.fit(y, pars, x=x)

    # account for functions whose centers are not mu
    mu = out.params['center'].value
    if lineshape is lineshapes.lognormal:
        cen = np.exp(mu - out.params['sigma']**2)
    else:
        cen = mu

    # get arguments for lineshape
    args = {key: out.best_values[key] for key in
            inspect_args(lineshape)[0] if key != 'x'}

    # output format for assertion errors
    fmt = ("Program calculated values and real values do not match!\n"
           "{:^20s}{:^20s}{:^20s}{:^20s}\n"
           "{:^20s}{:^20f}{:^20f}{:^20f}")

    if 'height' in out.params:
        height_pro = out.params['height'].value
        height_act = lineshape(cen, **args)
        diff = height_act - height_pro

        assert abs(diff) < 0.001, fmt.format(model._name, 'Actual', 'program',
                                             'Difference', 'Height',
                                             height_act, height_pro, diff)

        if 'fwhm' in out.params:
            fwhm_pro = out.params['fwhm'].value
            func = lambda x: lineshape(x, **args) - 0.5*height_act
            ret = fsolve(func, [cen - fwhm_pro/4, cen + fwhm_pro/2])
            fwhm_act = ret[1] - ret[0]
            diff = fwhm_act - fwhm_pro

            assert abs(diff) < 0.5, fmt.format(model._name, 'Actual',
                                               'program', 'Difference',
                                               'FWHM', fwhm_act, fwhm_pro,
                                               diff)


def test_height_fwhm_calculation(peakdata):
    """Test for correctness of height and FWHM calculation."""
    # mu = 0
    # variance = 1.0
    # sigma = np.sqrt(variance)
    # x = np.linspace(mu - 20*sigma, mu + 20*sigma, 100.0)
    # y = norm.pdf(x, mu, 1)
    x = peakdata[0]
    y = peakdata[1]
    check_height_fwhm(x, y, lineshapes.voigt, models.VoigtModel())
    check_height_fwhm(x, y, lineshapes.pvoigt, models.PseudoVoigtModel())
    check_height_fwhm(x, y, lineshapes.pearson7, models.Pearson7Model())
    check_height_fwhm(x, y, lineshapes.moffat, models.MoffatModel())
    check_height_fwhm(x, y, lineshapes.students_t, models.StudentsTModel())
    check_height_fwhm(x, y, lineshapes.breit_wigner, models.BreitWignerModel())
    check_height_fwhm(x, y, lineshapes.damped_oscillator,
                      models.DampedOscillatorModel())
    check_height_fwhm(x, y, lineshapes.dho,
                      models.DampedHarmonicOscillatorModel())
    check_height_fwhm(x, y, lineshapes.expgaussian,
                      models.ExponentialGaussianModel())
    check_height_fwhm(x, y, lineshapes.skewed_gaussian,
                      models.SkewedGaussianModel())
    check_height_fwhm(x, y, lineshapes.donaich, models.DonaichModel())
    x = x-9  # Lognormal will only fit peaks with centers < 1
    check_height_fwhm(x, y, lineshapes.lognormal, models.LognormalModel())


@pytest.mark.parametrize("lineshape", lineshapes.functions)
def test_finite_output_lineshape(lineshape):
    """Test for finite output of lineshape functions."""
    x = np.linspace(0, 100)

    # no need to test the lineshapes below
    if lineshape in ['linear', 'exponential', 'sine', 'expsine', 'powerlaw',
                     'parabolic', 'erf', 'erfc', 'wofz', 'gamma', 'gammaln']:
        return None

    elif lineshape in ['gaussian', 'lorentzian', 'damped_oscillator',
                       'logistic', 'lognormal', 'students_t']:
        func_args = (x, 1.0, x.size/2.0, 0.0)
    elif lineshape in ['split_lorentzian', 'voigt', 'pvoigt', 'dho',
                       'expgaussian', 'donaich', 'skewed_gaussian']:
        func_args = (x, 1.0, x.size/2.0, 0.0, 0.0)
    elif lineshape in ['moffat', 'pearson7', 'breit_wigner']:
        func_args = (x, 1.0, x.size/2.0, 0.0, 1.0)
    elif lineshape in ['skewed_voigt']:
        func_args = (x, 1.0, x.size/2.0, 0.0, 0.0, 0.0)
    elif lineshape == 'step':
        func_args = (x, 1.0, x.size/2.0, 0.0, 'linear')
    elif lineshape == 'rectangle':
        func_args = (x, 1.0, x.size/2.0, 0.0, x.size/2.0, 0.0, 'linear')

    ls = getattr(lineshapes, lineshape)
    out = ls(*func_args)
    assert np.all(np.isfinite(out))


def test_height_and_fwhm_expression_evalution_in_builtin_models():
    """Assert models do not throw an ZeroDivisionError."""
    mod = models.GaussianModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9)
    params.update_constraints()

    mod = models.LorentzianModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9)
    params.update_constraints()

    mod = models.SplitLorentzianModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, sigma_r=1.0)
    params.update_constraints()

    mod = models.VoigtModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, gamma=1.0)
    params.update_constraints()

    mod = models.PseudoVoigtModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, fraction=0.5)
    params.update_constraints()

    mod = models.MoffatModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, beta=0.0)
    params.update_constraints()

    mod = models.Pearson7Model()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, expon=1.0)
    params.update_constraints()

    mod = models.StudentsTModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9)
    params.update_constraints()

    mod = models.BreitWignerModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, q=0.0)
    params.update_constraints()

    mod = models.LognormalModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9)
    params.update_constraints()

    mod = models.DampedOscillatorModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9)
    params.update_constraints()

    mod = models.DampedHarmonicOscillatorModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, gamma=0.0)
    params.update_constraints()

    mod = models.ExponentialGaussianModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, gamma=0.0)
    params.update_constraints()

    mod = models.SkewedGaussianModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, gamma=0.0)
    params.update_constraints()

    mod = models.SkewedVoigtModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, gamma=0.0,
                             skew=0.0)
    params.update_constraints()

    mod = models.DonaichModel()
    params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, gamma=0.0)
    params.update_constraints()

    mod = models.StepModel()
    for f in ('linear', 'arctan', 'erf', 'logistic'):
        params = mod.make_params(amplitude=1.0, center=0.0, sigma=0.9, form=f)
        params.update_constraints()

    mod = models.RectangleModel()
    for f in ('linear', 'arctan', 'erf', 'logistic'):
        params = mod.make_params(amplitude=1.0, center1=0.0, sigma1=0.0,
                                 center2=0.0, sigma2=0.0, form=f)
        params.update_constraints()


def test_guess_modelparams():
    x = np.linspace(-10, 10, 501)

    mod = models.ConstantModel()
    y = 6.0 + x*0.005
    pars = mod.guess(y)
    assert_allclose(pars['c'].value, 6.0, rtol=0.01)

    mod = models.ComplexConstantModel(prefix='f_')
    y = 6.0 + x*0.005 + (4.0 - 0.02*x)*1j
    pars = mod.guess(y)
    assert_allclose(pars['f_re'].value, 6.0, rtol=0.01)
    assert_allclose(pars['f_im'].value, 4.0, rtol=0.01)

    mod = models.QuadraticModel(prefix='g_')
    y = -0.2 + 3.0*x + 0.005*x**2
    pars = mod.guess(y, x=x)
    assert_allclose(pars['g_a'].value, 0.005, rtol=0.01)
    assert_allclose(pars['g_b'].value, 3.0, rtol=0.01)
    assert_allclose(pars['g_c'].value, -0.2, rtol=0.01)

    mod = models.PolynomialModel(4, prefix='g_')
    y = -0.2 + 3.0*x + 0.005*x**2 - 3.3e-6*x**3 + 1.e-9*x**4
    pars = mod.guess(y, x=x)
    assert_allclose(pars['g_c0'].value, -0.2, rtol=0.01)
    assert_allclose(pars['g_c1'].value, 3.0, rtol=0.01)
    assert_allclose(pars['g_c2'].value, 0.005, rtol=0.1)
    assert_allclose(pars['g_c3'].value, -3.3e-6, rtol=0.1)
    assert_allclose(pars['g_c4'].value, 1.e-9, rtol=0.1)

    mod = models.GaussianModel(prefix='g_')
    y = lineshapes.gaussian(x, amplitude=2.2, center=0.25, sigma=1.3)
    y += np.random.normal(size=len(x), scale=0.004)
    pars = mod.guess(y, x=x)
    assert_allclose(pars['g_amplitude'].value, 3, rtol=2)
    assert_allclose(pars['g_center'].value, 0.25, rtol=1)
    assert_allclose(pars['g_sigma'].value, 1.3, rtol=1)

    mod = models.LorentzianModel(prefix='l_')
    pars = mod.guess(y, x=x)
    assert_allclose(pars['l_amplitude'].value, 3, rtol=2)
    assert_allclose(pars['l_center'].value, 0.25, rtol=1)
    assert_allclose(pars['l_sigma'].value, 1.3, rtol=1)

    mod = models.SplitLorentzianModel(prefix='s_')
    pars = mod.guess(y, x=x)
    assert_allclose(pars['s_amplitude'].value, 3, rtol=2)
    assert_allclose(pars['s_center'].value, 0.25, rtol=1)
    assert_allclose(pars['s_sigma'].value, 1.3, rtol=1)
    assert_allclose(pars['s_sigma_r'].value, 1.3, rtol=1)

    mod = models.VoigtModel(prefix='l_')
    pars = mod.guess(y, x=x)
    assert_allclose(pars['l_amplitude'].value, 3, rtol=2)
    assert_allclose(pars['l_center'].value, 0.25, rtol=1)
    assert_allclose(pars['l_sigma'].value, 1.3, rtol=1)

    mod = models.SkewedVoigtModel(prefix='l_')
    pars = mod.guess(y, x=x)
    assert_allclose(pars['l_amplitude'].value, 3, rtol=2)
    assert_allclose(pars['l_center'].value, 0.25, rtol=1)
    assert_allclose(pars['l_sigma'].value, 1.3, rtol=1)


def test_splitlorentzian_prefix():
    mod1 = models.SplitLorentzianModel()
    par1 = mod1.make_params(amplitude=1.0, center=0.0, sigma=0.9, sigma_r=1.3)
    par1.update_constraints()

    mod2 = models.SplitLorentzianModel(prefix='prefix_')
    par2 = mod2.make_params(amplitude=1.0, center=0.0, sigma=0.9, sigma_r=1.3)
    par2.update_constraints()
