import matplotlib.pyplot as plt
import numpy as np

from lmfit import Model, Parameters


def my_poly(x, prefix='coef_', **params):
    """here we use parameters that start with the 'prefix'
    and use the remaining part of the parameter name to give the
    polynomial order.

    That is, with prefix='coef_', the parameter name 'coef_0'
    will have the 0-order value, and 'coef_1' will have the linear value.
    """
    val = 0.0
    plen = len(prefix)
    for pname, pval in params.items():
        if pname.startswith(prefix):
            i = int(pname[plen:])
            val += pval*x**i
    return val


my_model = Model(my_poly)

# Parameter names and starting values
params = Parameters()
params.add('coef_0', value=10.0)
params.add('coef_1', value=1.0)
params.add('coef_2', value=0.1)
params.add('coef_3', value=0.01)
params.add('coef_4', value=0.001)

np.random.seed(10)

x = np.linspace(-40, 40, 161)
y = -30.4 + 2.8*x - 0.5*x*x - 0.015 * x**3 + 2.e-4*x**4
y = y + np.random.normal(size=len(y), scale=4)

out = my_model.fit(y, params, prefix='coef_', x=x, weights=0.2)

print(out.fit_report())

plt.plot(x, y, label='data')
plt.plot(x, out.best_fit, label='fit')
plt.legend()
plt.show()
