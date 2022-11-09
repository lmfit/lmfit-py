"""Tests of ModelResult.eval_uncertainty()"""
import os

import numpy as np
from numpy.testing import assert_allclose

from lmfit.lineshapes import gaussian
from lmfit.models import ExponentialModel, GaussianModel, LinearModel


def get_linearmodel(slope=0.8, intercept=0.5, noise=1.5):
    # create data to be fitted
    np.random.seed(88)
    x = np.linspace(0, 10, 101)
    y = intercept + x*slope
    y = y + np.random.normal(size=len(x), scale=noise)

    model = LinearModel()
    params = model.make_params(intercept=intercept, slope=slope)

    return x, y, model, params


def get_gaussianmodel(amplitude=1.0, center=5.0, sigma=1.0, noise=0.1):
    # create data to be fitted
    np.random.seed(7392)
    x = np.linspace(-20, 20, 201)
    y = gaussian(x, amplitude, center=center, sigma=sigma)
    y = y + np.random.normal(size=len(x), scale=noise)

    model = GaussianModel()
    params = model.make_params(amplitude=amplitude/5.0,
                               center=center-1.0,
                               sigma=sigma*2.0)
    return x, y, model, params


def test_linear_constant_intercept():
    x, y, model, params = get_linearmodel(slope=4, intercept=-10)

    params['intercept'].vary = False

    ret = model.fit(y, params, x=x)

    dely = ret.eval_uncertainty(sigma=1)
    slope_stderr = ret.params['slope'].stderr

    assert_allclose(dely.min(), 0, rtol=1.e-2)
    assert_allclose(dely.max(), slope_stderr*x.max(), rtol=1.e-2)
    assert_allclose(dely.mean(), slope_stderr*x.mean(), rtol=1.e-2)


def test_linear_constant_slope():
    x, y, model, params = get_linearmodel(slope=-4, intercept=2.3)

    params['slope'].vary = False

    ret = model.fit(y, params, x=x)

    dely = ret.eval_uncertainty(sigma=1)

    intercept_stderr = ret.params['intercept'].stderr

    assert_allclose(dely.min(), intercept_stderr, rtol=1.e-2)
    assert_allclose(dely.max(), intercept_stderr, rtol=1.e-2)


def test_gauss_sigmalevel():
    """Test that dely increases as sigma increases."""
    x, y, model, params = get_gaussianmodel(amplitude=50.0, center=4.5,
                                            sigma=0.78, noise=0.1)
    ret = model.fit(y, params, x=x)

    dely_sigma1 = ret.eval_uncertainty(sigma=1)
    dely_sigma2 = ret.eval_uncertainty(sigma=2)
    dely_sigma3 = ret.eval_uncertainty(sigma=3)

    assert dely_sigma3.mean() > 1.5*dely_sigma2.mean()
    assert dely_sigma2.mean() > 1.5*dely_sigma1.mean()


def test_gauss_noiselevel():
    """Test that dely increases as expected with changing noise level."""
    lonoise = 0.05
    hinoise = 10*lonoise
    x, y, model, params = get_gaussianmodel(amplitude=20.0, center=2.1,
                                            sigma=1.0, noise=lonoise)
    ret1 = model.fit(y, params, x=x)
    dely_lonoise = ret1.eval_uncertainty(sigma=1)

    x, y, model, params = get_gaussianmodel(amplitude=20.0, center=2.1,
                                            sigma=1.0, noise=hinoise)
    ret2 = model.fit(y, params, x=x)
    dely_hinoise = ret2.eval_uncertainty(sigma=1)

    assert_allclose(dely_hinoise.mean(), 10*dely_lonoise.mean(), rtol=1.e-2)


def test_component_uncertainties():
    "test dely_comps"
    y, x = np.loadtxt(os.path.join(os.path.dirname(__file__), '..',
                                   'examples', 'NIST_Gauss2.dat')).T
    model = (GaussianModel(prefix='g1_') +
             GaussianModel(prefix='g2_') +
             ExponentialModel(prefix='bkg_'))

    params = model.make_params(bkg_amplitude=100, bkg_decay=80,
                               g1_amplitude=3000,
                               g1_center=100,
                               g1_sigma=10,
                               g2_amplitude=3000,
                               g2_center=150,
                               g2_sigma=10)

    result = model.fit(y, params, x=x)
    comps = result.eval_components(x=x)
    dely = result.eval_uncertainty(sigma=3)

    assert 'g1_' in comps
    assert 'g2_' in comps
    assert 'bkg_' in comps
    assert dely.mean() > 0.8
    assert dely.mean() < 2.0
    assert result.dely_comps['g1_'].mean() > 0.5
    assert result.dely_comps['g1_'].mean() < 1.5
    assert result.dely_comps['g2_'].mean() > 0.5
    assert result.dely_comps['g2_'].mean() < 1.5
    assert result.dely_comps['bkg_'].mean() > 0.5
    assert result.dely_comps['bkg_'].mean() < 1.5
