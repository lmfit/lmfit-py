import numpy as np

import lmfit
from lmfit.lineshapes import gaussian
from lmfit.models import GaussianModel, SplineModel


def _isclose(name, expected_value, fit_value, atol, rtol):
    """isclose with error message"""
    assert np.isclose(expected_value, fit_value, atol=atol, rtol=rtol), \
           f"bad value for {name}: expected {expected_value}, got {fit_value}."


def check_fit(model, params, x, y, test_values, noise_scale=1.e-3, atol=0.1, rtol=0.05):
    """Checks that a model fits noisy data well

    Parameters
    -----------
    model:  model to use
    par:  parameters to use
    x:      x data
    y:      y data
    test_values: dict of 'true values'
    noise_scale: float, optional
           The standard deviation of noise that is added to the test data.
    atol: float, optional
           Absolute tolerance for considering fit parameters close to the
           parameters test data was generated with.
    rtol: float, optional
           Relative tolerance for considering fit parameters close to the
           parameters test data was generated with.

    Returns
    -------
      fit result

    Raises
    -------
       AssertionError
          Any fit parameter that is not close to the parameter used to
          generate the test data raises this error.
    """
    y += np.random.normal(scale=noise_scale, size=len(y))
    result = model.fit(y, params, x=x)
    fit_values = result.best_values
    for name, test_val in test_values.items():
        _isclose(name, test_val, fit_values[name], atol, rtol)
    return result


def testLinear():
    mod = lmfit.models.LinearModel()
    x = np.linspace(-1, 1, 201)
    y = 10*x + 2
    params = mod.make_params(intercept=1, slope=2)
    check_fit(mod, params, x, y, dict(intercept=2, slope=10))


def testQuadratic():
    mod = lmfit.models.QuadraticModel()
    x = np.linspace(-1, 1, 201)
    y = 0.3*x*x + 10*x + 2
    params = mod.make_params(a=0, b=5, c=1)
    check_fit(mod, params, x, y, dict(a=0.3, b=10, c=2))


def testSine_partialperiod():
    mod = lmfit.models.SineModel()
    x = np.linspace(-1, 1, 201)
    pars = dict(amplitude=1.5, frequency=0.9, shift=0.4)

    y = pars['amplitude']*np.sin(x*pars['frequency'] + pars['shift'])

    params = mod.make_params(amplitude=1, frequency=1, shift=-0.2)
    check_fit(mod, params, x, y, pars)


def testSineWithLine():
    mod = lmfit.models.SineModel() + lmfit.models.LinearModel()
    x = np.linspace(-5, 5, 501)
    pars = dict(amplitude=5.3, frequency=3.8, shift=0.1, intercept=8.2, slope=0.2)

    y = pars['amplitude']*np.sin(x*pars['frequency'] + pars['shift'])
    y += pars['intercept'] + x * pars['slope']

    params = mod.make_params(amplitude=10, frequency=4.5, shift=0.1,
                             intercept=10, slope=0)

    check_fit(mod, params, x, y, pars, noise_scale=0.02)


def testSineManyShifts():
    mod = lmfit.models.SineModel() + lmfit.models.LinearModel()
    x = np.linspace(-5, 5, 501)
    pars = dict(amplitude=5.3, frequency=3.8, intercept=8.2, slope=0.2)

    for shift in (0.1, 0.5, 1.0, 1.5):
        pars['shift'] = shift
        y = pars['amplitude']*np.sin(x*pars['frequency'] + pars['shift'])
        y += pars['intercept'] + x*pars['slope']

        params = mod.make_params(amplitude=10, frequency=4.5, shift=0.8,
                                 intercept=10, slope=0)

        check_fit(mod, params, x, y, pars, noise_scale=0.02)


def testSineModel_guess():
    mod = lmfit.models.SineModel()
    x = np.linspace(-10, 10, 201)
    pars = dict(amplitude=1.5, frequency=0.5, shift=0.4)

    y = pars['amplitude']*np.sin(x*pars['frequency'] + pars['shift'])

    params = mod.guess(y, x=x)
    assert params['amplitude'] > 0.5
    assert params['amplitude'] < 5.0
    assert params['frequency'] > 0.1
    assert params['frequency'] < 1.5
    assert params['shift'] > 0.0
    assert params['shift'] < 1.0


def testSineModel_make_params():
    mod = lmfit.models.SineModel()
    pars = mod.make_params(amplitude=1.5, frequency=0.5,
                           shift=dict(value=0.4, min=-10, max=10))

    assert pars['amplitude'].value > 1.4
    assert pars['amplitude'].value < 1.6
    assert pars['amplitude'].min == 0
    assert pars['amplitude'].max > 1e99
    assert pars['shift'].value > 0.35
    assert pars['shift'].value < 0.45
    assert pars['shift'].min < -9.5
    assert pars['shift'].max > 9.5


def testSplineModel():
    x = np.linspace(0, 25, 501)
    y = gaussian(x, amplitude=10, center=16.2, sigma=0.8) + 3 + 0.03*x + np.sin(x/4)

    model = GaussianModel(prefix='peak_')
    params = model.make_params(amplitude=8, center=16, sigma=1)

    knot_xvals = np.array([1, 3, 5, 7, 9, 11, 13, 19, 21, 23, 25])

    bkg = SplineModel(prefix='bkg_', xknots=knot_xvals)
    params.update(bkg.guess(y, x))

    model = model + bkg

    pars = dict(peak_amplitude=10, peak_center=16.2, peak_sigma=0.8)
    check_fit(model, params, x, y, pars, noise_scale=0.05)
