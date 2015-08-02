import numpy as np
from lmfit import Parameters, minimize, report_fit
from lmfit.models import LinearModel, GaussianModel
from lmfit.lineshapes import gaussian

def per_iteration(pars, iter, resid, *args, **kws):
    """iteration callback, will abort at iteration 23
    """
    # print( iter, ', '.join(["%s=%.4f" % (p.name, p.value) for p in pars.values()]))
    return iter == 23

def test_itercb():
    x = np.linspace(0, 20, 401)
    y = gaussian(x, amplitude=24.56, center=7.6543, sigma=1.23)
    y = y  - .20*x + 3.333 + np.random.normal(scale=0.23,  size=len(x))
    mod = GaussianModel(prefix='peak_') + LinearModel(prefix='bkg_')

    pars = mod.make_params(peak_amplitude=21.0,
                           peak_center=7.0,
                           peak_sigma=2.0,
                           bkg_intercept=2,
                           bkg_slope=0.0)

    out = mod.fit(y, pars, x=x, iter_cb=per_iteration)

    assert(out.nfev == 23)
    assert(out.aborted)
    assert(not out.errorbars)
    assert(not out.success)
