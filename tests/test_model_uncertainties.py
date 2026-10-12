"""Tests of ModelResult.eval_uncertainty()"""
import os

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.stats import t

from lmfit.lineshapes import gaussian
from lmfit.model import Model
from lmfit.models import ExponentialModel, GaussianModel, LinearModel


@pytest.mark.parametrize('array_output', [False, True])
def test_python_complex_scalar_uncertainty(array_output):
    """Evaluate a scalar complex model with the existing complex-pair convention."""
    def constant(x, a, b):
        value = complex(a, b)
        return np.asarray(value) if array_output else value

    rng = np.random.default_rng(123)
    data = rng.normal(size=20) + 1j*rng.normal(size=20)
    result = Model(constant).fit(data, x=np.arange(20), a=0, b=0)
    scale = t.ppf(0.975, result.ndata-result.nvarys)
    expected = scale*np.sqrt(result.covar[0, 0] + 1j*result.covar[1, 1])
    actual = result.eval_uncertainty(sigma=0.95)
    assert_allclose(actual, expected, rtol=1e-8)
    assert np.shape(actual) == ()


@pytest.mark.parametrize('layout', ['C', 'F', 'strided', 'reversed'])
def test_complex_array_uncertainty_layout(layout):
    """Compare complex array propagation with an analytic Jacobian."""
    def gain(x, a, b):
        value = (a + 1j*b)*x
        if layout == 'F':
            return np.asfortranarray(value)
        if layout == 'strided':
            return np.stack([value, value], axis=-1)[..., 0]
        if layout == 'reversed':
            return value[..., ::-1].copy()[..., ::-1]
        return value

    rng = np.random.default_rng(123)
    x = (np.linspace(-1, 1, 15) + 1j*np.linspace(0.2, 1.2, 15)).reshape(3, 5)
    data = (0.8 + 0.3j)*x + 0.03*(rng.normal(size=x.shape) + 1j*rng.normal(size=x.shape))
    result = Model(gain).fit(data, x=x, a=0.7, b=0.2)
    jr = np.stack([x.real, -x.imag], axis=-1)
    ji = np.stack([x.imag, x.real], axis=-1)
    variance = (np.einsum('...i,ij,...j->...', jr, result.covar, jr)
                + 1j*np.einsum('...i,ij,...j->...', ji, result.covar, ji))
    scale = t.ppf(0.975, result.ndata-result.nvarys)
    actual = result.eval_uncertainty(sigma=0.95)
    assert_allclose(actual, scale*np.sqrt(variance), rtol=1e-8)
    assert_allclose(result.dely_predicted, scale*np.sqrt(variance + result.redchi), rtol=1e-8)
    assert actual.shape == x.shape


@pytest.mark.parametrize('scalar', [False, True])
def test_real_component_of_complex_model_uncertainty(scalar):
    """Real components contribute only to the real derivative channels."""
    def gain(x, a, b):
        return (a + 1j*b)*x

    def offset(x, c):
        return float(c) if scalar else np.full(x.shape, c, dtype=float)

    rng = np.random.default_rng(123)
    x = (np.linspace(-1, 1, 15) + 1j*np.linspace(0.2, 1.2, 15)).reshape(3, 5)
    data = (0.8 + 0.3j)*x + 0.2 + 0.03*(rng.normal(size=x.shape) + 1j*rng.normal(size=x.shape))
    model = Model(gain, prefix='gain_') + Model(offset, prefix='bias_')
    result = model.fit(data, x=x, gain_a=0.7, gain_b=0.2, bias_c=0.1)
    result.eval_uncertainty(sigma=0.95)
    scale = t.ppf(0.975, result.ndata-result.nvarys)
    expected_bias = np.full(x.shape, scale*np.sqrt(result.covar[2, 2]))
    assert_allclose(result.dely_comps['bias_'], expected_bias, rtol=1e-8)


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


def test_scalar_independent_vars():
    """Github #951"""
    mr = LinearModel().fit(data=[1, 2, 4], x=[1, 2, 3])

    assert_allclose(mr.eval_uncertainty(x=1.2), np.array([0.60629034]))
    assert_allclose(mr.eval_uncertainty(x=np.array(1.2j)), np.array([1.15903153+0.17475665j]))


def test_multidim_model():
    """test models that return a multi-dimension output"""
    def fit(x, p=2):
        return np.array([x, p*x])
    model = Model(fit)

    mr = model.fit(data=np.array([[1, 2, 3], [2, 3, 5]]), x=np.array([1, 2, 3]))

    assert_allclose(mr.eval_uncertainty(x=np.array([3, 4])), np.array([[0, 0], [0.18432744, 0.24576991]]))
    assert_allclose(mr.eval_uncertainty(x=np.array([3j, 4j])), np.array([[0, 0], [0.13033918+0.13033918j, 0.17378557+0.17378557j]]))
