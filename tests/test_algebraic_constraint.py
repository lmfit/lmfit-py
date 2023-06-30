"""Tests for algebraic parameter constraints."""
import numpy as np
import pytest

from lmfit import Minimizer, Model, Parameters
from lmfit.lineshapes import gaussian, lorentzian


@pytest.fixture
def minimizer():
    """Return the Minimizer object."""
    def residual(pars, x, sigma=None, data=None):
        """Define objective function."""
        yg = gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g'])
        yl = lorentzian(x, pars['amp_l'], pars['cen_l'], pars['wid_l'])

        model = yg + yl + pars['line_off'] + x * pars['line_slope']

        if data is None:
            return model
        if sigma is None:
            return model - data
        return (model-data) / sigma

    # generate synthetic data
    n = 601
    xmin = 0.
    xmax = 20.0
    x = np.linspace(xmin, xmax, n)

    data = (gaussian(x, 21, 8.1, 1.2) + lorentzian(x, 10, 9.6, 2.4) +
            np.random.normal(scale=0.23, size=n) + x*0.5)

    # create initial Parameters
    pars = Parameters()
    pars.add(name='amp_g', value=10)
    pars.add(name='cen_g', value=9)
    pars.add(name='wid_g', value=1)
    pars.add(name='amp_tot', value=20)
    pars.add(name='amp_l', expr='amp_tot - amp_g')
    pars.add(name='cen_l', expr='1.5+cen_g')
    pars.add(name='wid_l', expr='2*wid_g')
    pars.add(name='line_slope', value=0.0)
    pars.add(name='line_off', value=0.0)

    sigma = 0.021  # estimate of data error (for all data points)

    mini = Minimizer(residual, pars, fcn_args=(x,), fcn_kws={'sigma': sigma,
                                                             'data': data})

    return mini


def test_algebraic_constraints(minimizer):
    """Test algebraic constraints."""
    result = minimizer.minimize(method='leastsq')

    pfit = result.params
    assert pfit['cen_l'].value == 1.5 + pfit['cen_g'].value
    assert pfit['amp_l'].value == pfit['amp_tot'].value - pfit['amp_g'].value
    assert pfit['wid_l'].value == 2.0 * pfit['wid_g'].value


def test_algebraic_constraints_function(minimizer):
    """Test constraints with a user-defined function added to symbol table."""
    def width_func(wpar):
        return 2.5*wpar

    minimizer.params._asteval.symtable['wfun'] = width_func
    minimizer.params.add(name='wid_l', expr='wfun(wid_g)')
    result = minimizer.minimize(method='leastsq')

    pfit = result.params
    assert pfit['cen_l'].value == 1.5 + pfit['cen_g'].value
    assert pfit['amp_l'].value == pfit['amp_tot'].value - pfit['amp_g'].value
    assert pfit['wid_l'].value == 2.5 * pfit['wid_g'].value


def test_constraints_function_call():
    """Test a constraint with simple function call in Model class."""

    x = np.array([1723, 1773, 1823, 1523, 1773, 1033.03078, 1042.98077,
                  1047.90937, 1053.95899, 1057.94906, 1063.13788, 1075.74218,
                  1086.03102])
    y = np.array([0.79934, -0.31876, -0.46852, 0.05, -0.21, 11.1708, 10.31844,
                  9.73069, 9.21319, 9.12457, 9.05243, 8.66407, 8.29664])

    def VFT(T, ninf=-3, A=5e3, T0=800):
        return ninf + A/(T-T0)

    vftModel = Model(VFT)
    vftModel.set_param_hint('D', vary=False, expr=r'A*log(10)/T0')
    result = vftModel.fit(y, T=x)

    assert 2600.0 < result.params['A'].value < 2650.0
    assert 7.0 < result.params['D'].value < 7.5


def test_constraints(minimizer):
    """Test changing of algebraic constraints."""
    result = minimizer.minimize(method='leastsq')

    pfit = result.params
    assert pfit['cen_l'].value == 1.5 + pfit['cen_g'].value
    assert pfit['amp_l'].value == pfit['amp_tot'].value - pfit['amp_g'].value
    assert pfit['wid_l'].value == 2.0*pfit['wid_g'].value

    # now, change fit slightly and re-run
    minimizer.params['wid_l'].expr = '1.25*wid_g'
    result = minimizer.minimize(method='leastsq')
    pfit = result.params

    assert pfit['cen_l'].value == 1.5 + pfit['cen_g'].value
    assert pfit['amp_l'].value == pfit['amp_tot'].value - pfit['amp_g'].value
    assert pfit['wid_l'].value == 1.25*pfit['wid_g'].value
