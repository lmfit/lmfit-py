# <examples/doc_builtinmodels_splinemodel.py>
import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import GaussianModel, SplineModel

data = np.loadtxt('test_splinepeak.dat')
x = data[:, 0]
y = data[:, 1]

plt.plot(x, y, label='data')

model = GaussianModel(prefix='peak_')
params = model.make_params(amplitude=dict(value=8, min=0),
                           center=dict(value=16, min=5, max=25),
                           sigma=dict(value=1, min=0))

# make a background spline with knots evenly spaced over the background,
# but sort of skipping over where the peak is
knot_xvals3 = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
knot_xvals2 = np.array([1, 3, 5, 7, 9, 11, 13,   16,   19, 21, 23, 25])  # noqa: E241
knot_xvals1 = np.array([1, 3, 5, 7, 9, 11, 13,         19, 21, 23, 25])  # noqa: E241

bkg = SplineModel(prefix='bkg_', xknots=knot_xvals1)
params.update(bkg.guess(y, x))

model = model + bkg

plt.plot(x, model.eval(params, x=x), label='initial')

out = model.fit(y, params, x=x)
print(out.fit_report(min_correl=0.3))
comps = out.eval_components()

plt.plot(x, out.best_fit, label='best fit')
plt.plot(x, comps['bkg_'], label='background')
plt.plot(x, comps['peak_'], label='peak')

knot_yvals = np.array([o.value for o in out.params.values() if o.name.startswith('bkg')])
plt.plot(knot_xvals1, knot_yvals, 'o', color='black', label='spline knots values')
plt.legend()
plt.show()


#   knot positions         | peak amplitude
#  11, 13, 19, 21          |  12.223  0.295
#  11, 13, 16, 19, 21      |  11.746  0.594
#  11, 13, 15, 17, 19, 21  |  12.052  0.872


plt.plot(x, y, 'o', label='data')

for nknots in (10, 15, 20, 25, 30):
    model = SplineModel(prefix='bkg_', xknots=np.linspace(0, 25, nknots))
    params = model.guess(y, x)
    out = model.fit(y, params, x=x)
    plt.plot(x, out.best_fit, label=f'best-fit ({nknots} knots)')
plt.legend()
plt.show()

# <end examples/doc_builtinmodels_splinemodel.py>
