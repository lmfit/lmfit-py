import os

import numpy as np
import pytest

import lmfit


@pytest.fixture
def minimizer_Alpine02():
    """Return a lmfit Minimizer object for the Alpine02 function."""
    def residual_Alpine02(params):
        x0 = params['x0'].value
        x1 = params['x1'].value
        return np.prod(np.sqrt(x0) * np.sin(x0)) * np.prod(np.sqrt(x1) *
                                                           np.sin(x1))

    # create Parameters and set initial values and bounds
    pars = lmfit.Parameters()
    pars.add_many(('x0', 1., True, 0.0, 10.0),
                  ('x1', 1., True, 0.0, 10.0))

    mini = lmfit.Minimizer(residual_Alpine02, pars)
    return mini


@pytest.fixture
def peakdata():
    """Return the peak-like test data."""
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), '..',
                                   'examples', 'test_peak.dat'))
    return data.T
