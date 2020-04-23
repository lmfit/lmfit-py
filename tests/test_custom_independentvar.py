import numpy as np

from lmfit.lineshapes import gaussian
from lmfit.models import Model


class Stepper:
    def __init__(self, start, stop, npts):
        self.start = start
        self.stop = stop
        self.npts = npts

    def get_x(self):
        return np.linspace(self.start, self.stop, self.npts)


def gaussian_mod(obj, amplitude, center, sigma):
    return gaussian(obj.get_x(), amplitude, center, sigma)


def test_custom_independentvar():
    """Tests using a non-trivial object as an independent variable."""
    npts = 501
    xmin = 1
    xmax = 21
    cen = 8
    obj = Stepper(xmin, xmax, npts)
    y = gaussian(obj.get_x(), amplitude=3.0, center=cen, sigma=2.5)
    y += np.random.normal(scale=0.2, size=npts)

    gmod = Model(gaussian_mod)

    params = gmod.make_params(amplitude=2, center=5, sigma=8)
    out = gmod.fit(y, params, obj=obj)

    assert out.nvarys == 3
    assert out.nfev > 10
    assert out.chisqr > 1
    assert out.chisqr < 100
    assert out.params['sigma'].value < 3
    assert out.params['sigma'].value > 2
    assert out.params['center'].value > xmin
    assert out.params['center'].value < xmax
    assert out.params['amplitude'].value > 1
    assert out.params['amplitude'].value < 5
