import numpy as np
from nose.tools import assert_true
from lmfit.lineshapes import gaussian
from lmfit.models import GaussianModel


def test_default_inputs_gauss():

    area = 1
    cen = 0
    std = 0.2
    x = np.arange(-3, 3, 0.01)
    y = gaussian(x, area, cen, std)

    g = GaussianModel()

    fit_option1 = {'maxfev': 5000, 'xtol': 1e-2}
    result1 = g.fit(y, x=x, amplitude=1, center=0, sigma=0.5, fit_kws=fit_option1)

    fit_option2 = {'maxfev': 5000, 'xtol': 1e-6}
    result2 = g.fit(y, x=x, amplitude=1, center=0, sigma=0.5, fit_kws=fit_option2)

    assert_true(result1.values!=result2.values)
    return
