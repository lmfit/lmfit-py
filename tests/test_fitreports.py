import numpy as np

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
