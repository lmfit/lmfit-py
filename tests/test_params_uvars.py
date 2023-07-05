"""Tests for the Iteration Callback Function."""
import os

import numpy as np

from lmfit.models import ExponentialModel, GaussianModel

y, x = np.loadtxt(os.path.join(os.path.dirname(__file__), '..',
                               'examples', 'NIST_Gauss2.dat')).T

nist_model = (ExponentialModel(prefix='exp_')
              + GaussianModel(prefix='g1_')
              + GaussianModel(prefix='g2_'))


def test_uvars_calc_afterfit():
    """test using uvars by hand"""
    pars = nist_model.make_params(g1_center=110, g1_amplitude=4000,
                                  g1_sigma=25, g2_center=160,
                                  g2_amplitude=2000, g2_sigma=20,
                                  exp_amplitude=50, exp_decay=75)

    out = nist_model.fit(y, pars, x=x)

    assert out.nfev > 25
    assert out.errorbars
    assert out.chisqr > 1000
    assert out.chisqr < 3000

    u_g1amp = out.uvars['g1_amplitude']
    u_g2amp = out.uvars['g2_amplitude']

    # stderr in area of Gauss1 + Gauss2 in quadrature, ignoring correlations
    area_stderr_quad = np.sqrt(out.params['g1_amplitude'].stderr**2 +
                               out.params['g2_amplitude'].stderr**2)
    assert area_stderr_quad > 53
    assert area_stderr_quad < 58

    # stderr in area of Gauss1 + Gauss2 including correlations: smaller.
    area_stderr_uvars = (u_g1amp + u_g2amp).std_dev
    assert area_stderr_uvars > 43
    assert area_stderr_uvars < 48


def test_uvars_calc_with_param_def():
    """test using uvars with a defined constraint parameter during the fit"""
    pars = nist_model.make_params(g1_center=110, g1_amplitude=4000,
                                  g1_sigma=25, g2_center=160,
                                  g2_amplitude=2000, g2_sigma=20,
                                  exp_amplitude=50, exp_decay=75)

    pars.add('area', expr='g1_amplitude + g2_amplitude')
    out = nist_model.fit(y, pars, x=x)

    assert out.nfev > 25
    assert out.errorbars
    assert out.chisqr > 1000
    assert out.chisqr < 3000

    post_area = out.params['area']

    assert post_area.value > 6500
    assert post_area.value < 7000

    assert post_area.stderr > 43
    assert post_area.stderr < 48


def test_uvars_calc_post_fit_method():
    """test using uvars and Model.post_fit method"""

    def post_fit(result):
        "example post fit function"
        result.params.add('post_area', expr='g1_amplitude + g2_amplitude')

    pars = nist_model.make_params(g1_center=110, g1_amplitude=4000,
                                  g1_sigma=25, g2_center=160,
                                  g2_amplitude=2000, g2_sigma=20,
                                  exp_amplitude=50, exp_decay=75)

    _model = (ExponentialModel(prefix='exp_')
              + GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_'))

    _model.post_fit = post_fit
    out = _model.fit(y, pars, x=x)

    assert out.nfev > 25
    assert out.errorbars
    assert out.chisqr > 1000
    assert out.chisqr < 3000

    post_area = out.params['post_area']

    assert post_area.value > 6500
    assert post_area.value < 7000

    assert post_area.stderr > 43
    assert post_area.stderr < 48
