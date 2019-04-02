import numpy as np

from lmfit import Minimizer, Parameters, conf_interval, ci_report, fit_report
from lmfit.models import GaussianModel
from lmfit.lineshapes import gaussian

np.random.seed(0)


def test_reports_created():
    """do a simple Model fit but with all the bells-and-whistles
    and verify that the reports are created
    """
    x = np.linspace(0, 12, 601)
    data = gaussian(x, amplitude=36.4, center=6.70, sigma=0.88)
    data = data + np.random.normal(size=len(x), scale=3.2)
    model = GaussianModel()
    params = model.make_params(amplitude=50, center=5, sigma=2)

    params['amplitude'].min = 0
    params['sigma'].min = 0
    params['sigma'].brute_step = 0.001

    result = model.fit(data, params, x=x)

    report = result.fit_report()
    assert(len(report) > 500)

    html_params = result.params._repr_html_()
    assert(len(html_params) > 500)

    html_report = result._repr_html_()
    assert(len(html_report) > 1000)


def test_ci_report():
    """test confidence interval report"""

    def residual(pars, x, data=None):
        argu = (x*pars['decay'])**2
        shift = pars['shift']
        if abs(shift) > np.pi/2:
            shift = shift - np.sign(shift)*np.pi
        model = pars['amp']*np.sin(shift + x/pars['period']) * np.exp(-argu)
        if data is None:
            return model
        return model - data

    p_true = Parameters()
    p_true.add('amp', value=14.0)
    p_true.add('period', value=5.33)
    p_true.add('shift', value=0.123)
    p_true.add('decay', value=0.010)

    n = 2500
    xmin = 0.
    xmax = 250.0
    x = np.linspace(xmin, xmax, n)
    data = residual(p_true, x) + np.random.normal(scale=0.7215, size=n)

    fit_params = Parameters()
    fit_params.add('amp', value=13.0)
    fit_params.add('period', value=2)
    fit_params.add('shift', value=0.0)
    fit_params.add('decay', value=0.02)

    mini = Minimizer(residual, fit_params, fcn_args=(x,),
                     fcn_kws={'data': data})
    out = mini.leastsq()
    report = fit_report(out)
    assert(len(report) > 500)

    ci, tr = conf_interval(mini, out, trace=True)
    report = ci_report(ci)
    assert(len(report) > 250)
