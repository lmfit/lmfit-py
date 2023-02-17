# <examples/doc_with_itercb.py>
import matplotlib.pyplot as plt
from numpy import linspace, random

from lmfit.lineshapes import gaussian
from lmfit.models import GaussianModel, LinearModel


def per_iteration(pars, iteration, resid, *args, **kws):
    print(" ITER ", iteration, [f"{p.name} = {p.value:.5f}" for p in pars.values()])


x = linspace(0., 20, 401)
y = gaussian(x, amplitude=24.56, center=7.6543, sigma=1.23)
random.seed(2021)
y = y - .20*x + 3.333 + random.normal(scale=0.23, size=x.size)

mod = GaussianModel(prefix='peak_') + LinearModel(prefix='bkg_')

pars = mod.make_params(peak_amplitude=dict(value=3.0, min=0),
                       peak_center=dict(value=6.0, min=0, max=20),
                       peak_sigma=2.0,
                       bkg_intercept=0,
                       bkg_slope=0)

out = mod.fit(y, pars, x=x, iter_cb=per_iteration)

plt.plot(x, y, '--')

print(f'Nfev = {out.nfev}')
print(out.fit_report())

plt.plot(x, out.best_fit, '-', label='best fit')
plt.legend()
plt.show()
# <end examples/doc_with_itercb.py>
