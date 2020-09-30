"""Tests for convolutions"""

from itertools import product

import numpy as np
from numpy.testing import assert_array_almost_equal

import pytest

from lmfit import CompositeModel
from lmfit.models import (LorentzianModel, GaussianModel,
                          PseudoVoigtModel, DeltaModel)
import lmfit.convolutions as conv


def get_models(prefix=None):
    """Set up the models with the associated convolutions."""
    lorentz = LorentzianModel(prefix=prefix)
    lorentz.convolutions = dict(
        lorentzian=conv.conv_lorentzian_lorentzian,
        gaussian=conv.conv_gaussian_lorentzian,
        pvoigt=conv.conv_lorentzian_pvoigt,
        delta=conv.conv_delta_lorentzian)

    gauss = GaussianModel(prefix=prefix)
    gauss.convolutions = dict(
        gaussian=conv.conv_gaussian_gaussian,
        lorentzian=conv.conv_gaussian_lorentzian,
        pvoigt=conv.conv_gaussian_pvoigt,
        delta=conv.conv_delta_gaussian)

    pvoigt = PseudoVoigtModel(prefix=prefix)
    pvoigt.convolutions = dict(
        gaussian=conv.conv_gaussian_pvoigt,
        lorentzian=conv.conv_lorentzian_pvoigt,
        delta=conv.conv_delta_pvoigt)

    delta = DeltaModel(prefix=prefix)
    pvoigt.convolutions = dict(
        gaussian=conv.conv_delta_gaussian,
        lorentzian=conv.conv_delta_lorentzian,
        pvoigt=conv.conv_delta_pvoigt)

    return (gauss, lorentz, delta, pvoigt)


@pytest.mark.parametrize(
    "modelPair", product(get_models('left_'), get_models('right_')))
def test_comp_analytic_numeric(modelPair):
    X = np.linspace(-10, 10, 1000)

    analytic = (modelPair[0] << modelPair[1]).eval(x=X)
    numeric = CompositeModel(
        *modelPair, lambda *pair: np.convolve(*pair, mode='same')).eval(x=X)

    # normalize both results
    assert_array_almost_equal(analytic / analytic.sum(),
                              numeric / numeric.sum(),
                              decimal=2)
