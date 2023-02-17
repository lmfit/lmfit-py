# <examples/doc_model_two_components.py>
import matplotlib.pyplot as plt
from numpy import exp, loadtxt, pi, sqrt

from lmfit import Model

data = loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1] + 0.25*x - 1.0


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))


def line(x, slope, intercept):
    """a line"""
    return slope*x + intercept


mod = Model(gaussian) + Model(line)
pars = mod.make_params(amp=5, cen=5, wid={'value': 1, 'min': 0},
                       slope=0, intercept=1)

result = mod.fit(y, pars, x=x)
print(result.fit_report())

plt.plot(x, y, 'o')
plt.plot(x, result.init_fit, '--', label='initial fit')
plt.plot(x, result.best_fit, '-', label='best fit')
plt.legend()
plt.show()
# <end examples/doc_model_two_components.py>
