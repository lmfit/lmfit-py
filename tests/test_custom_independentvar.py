import numpy as np

from lmfit.lineshapes import gaussian
from lmfit.model import coerce_arraylike
from lmfit.models import Model


class Stepper:
    def __init__(self, start, stop, npts):
        self.start = start
        self.stop = stop
        self.npts = npts

    def get_x(self):
        return np.linspace(self.start, self.stop, self.npts)


class CustomObject:
    """A minimal user-defined object that cannot be coerced to a float."""

    def __init__(self, value):
        self.value = value


def gaussian_mod(obj, amplitude, center, sigma):
    return gaussian(obj.get_x(), amplitude, center, sigma)


def gaussian_with_objlist(x, amplitude, center, sigma, custom_object_list=None):
    """Gaussian model that also accepts a list of custom objects as an
    independent variable; the list is not used in the calculation."""
    return gaussian(x, amplitude, center, sigma)


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


def test_list_of_custom_objects_independentvar():
    """A list of custom (non-float) objects used as an independent variable
    must not raise a TypeError in ``coerce_arraylike`` (GitHub issue #1040).

    ``numpy.isrealobj`` returns True for an object-dtype array, so the old
    code tried to cast the list of objects to float64 and raised.
    """
    npts = 501
    xmin = 1
    xmax = 21
    cen = 8
    x = np.linspace(xmin, xmax, npts)
    custom_object_list = [CustomObject(1), CustomObject(2), CustomObject(5)]

    y = gaussian(x, amplitude=3.0, center=cen, sigma=2.5)
    y += np.random.normal(scale=0.2, size=npts)

    gmod = Model(gaussian_with_objlist,
                 independent_vars=['x', 'custom_object_list'])
    params = gmod.make_params(amplitude=2, center=5, sigma=8)

    # before the fix this raised:
    #   TypeError: float() argument must be a string or a real number ...
    out = gmod.fit(y, params, x=x, custom_object_list=custom_object_list)

    # the list of custom objects must be passed through unchanged
    assert out.userkws['custom_object_list'] is custom_object_list

    assert out.nvarys == 3
    assert out.nfev > 10
    assert out.params['sigma'].value > 2
    assert out.params['sigma'].value < 3
    assert out.params['center'].value > xmin
    assert out.params['center'].value < xmax
    assert out.params['amplitude'].value > 1
    assert out.params['amplitude'].value < 5


def test_coerce_arraylike_dtypes():
    """``coerce_arraylike`` casts real-numeric and complex array-likes but
    leaves non-numeric objects (and scalars) unchanged (GitHub issue #1040)."""
    # real numeric lists/tuples/arrays -> float64 ndarray
    for seq in ([1, 2, 3], (1.0, 2.0, 3.0), np.arange(4)):
        res = coerce_arraylike(seq)
        assert isinstance(res, np.ndarray)
        assert res.dtype == np.float64
        assert np.allclose(res, np.asarray(seq, dtype=np.float64))

    # complex list -> complex128 ndarray
    res = coerce_arraylike([1 + 2j, 3 + 4j])
    assert isinstance(res, np.ndarray)
    assert res.dtype == np.complex128

    # list of custom, non-float objects -> returned unchanged (issue #1040)
    objs = [CustomObject(1), CustomObject(2)]
    assert coerce_arraylike(objs) is objs

    # scalars and plain objects pass through unchanged
    assert coerce_arraylike(5) == 5
    assert coerce_arraylike('a string') == 'a string'
