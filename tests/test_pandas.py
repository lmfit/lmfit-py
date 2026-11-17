"""Tests for using data in pandas.[DataFrame|Series]."""
import os

import numpy as np
import pytest

import lmfit

pandas = pytest.importorskip('pandas')


def test_pandas_guess_from_peak():
    """Regression test for failure in guess_from_peak with pandas (GH #629)."""
    data = pandas.read_csv(os.path.join(os.path.dirname(__file__), '..',
                                        'examples', 'peak.csv'))
    xdat, ydat = np.loadtxt(os.path.join(os.path.dirname(__file__), '..',
                                         'examples', 'peak.csv'),
                            unpack=True, skiprows=1, delimiter=',')

    model = lmfit.models.LorentzianModel()
    guess_pd = model.guess(data['y'], x=data['x'])
    guess = model.guess(ydat, x=xdat)

    assert guess_pd == guess


@pytest.mark.parametrize('model_class',
                         [lmfit.models.LorentzianModel,
                          lmfit.models.GaussianModel,
                          lmfit.models.VoigtModel])
def test_pandas_guess_from_peak_negative(model_class):
    """Test guess_from_peak with negative=True on pandas.Series."""
    data = pandas.read_csv(os.path.join(os.path.dirname(__file__), '..',
                                        'examples', 'peak.csv'))
    xdat, ydat = np.loadtxt(os.path.join(os.path.dirname(__file__), '..',
                                         'examples', 'peak.csv'),
                            unpack=True, skiprows=1, delimiter=',')

    model = model_class()
    guess_pd = model.guess(-data['y'], x=data['x'], negative=True)
    guess = model.guess(-ydat, x=xdat, negative=True)

    assert guess_pd == guess


@pytest.mark.parametrize('negative', [False, True])
def test_pandas_guess_from_peak_nondefault_index(negative):
    """Test guess_from_peak with filtered and decreasing pandas.Series."""
    data = pandas.read_csv(os.path.join(os.path.dirname(__file__), '..',
                                        'examples', 'peak.csv'))
    sub = data[data['x'] > 5.0]
    y_pd = -sub['y'] if negative else sub['y']

    x_arr = sub['x'].to_numpy()
    y_arr = y_pd.to_numpy()

    model = lmfit.models.LorentzianModel()
    guess_sub = model.guess(y_pd, x=sub['x'], negative=negative)
    guess_rev = model.guess(y_pd.iloc[::-1], x=sub['x'].iloc[::-1],
                            negative=negative)
    guess_np = model.guess(y_arr, x=x_arr, negative=negative)

    assert guess_sub == guess_np
    assert guess_rev == guess_np


def test_pandas_Voigt_model():
    """Regression test for Series.real reported in GH Issues 727."""
    data = pandas.read_csv(os.path.join(os.path.dirname(__file__), '..',
                                        'examples', 'peak.csv'))
    model = lmfit.models.VoigtModel()
    params = model.make_params()
    fit = model.fit(data['y'], params, x=data['x'])

    assert fit.success
