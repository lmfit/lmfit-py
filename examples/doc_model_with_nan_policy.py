# <examples/doc_model_with_nan_policy.py>
import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import GaussianModel

data = np.loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]

y[44] = np.nan
y[65] = np.nan

# nan_policy = 'raise'
# nan_policy = 'propagate'
nan_policy = 'omit'

gmodel = GaussianModel()
result = gmodel.fit(y, x=x, amplitude=5, center=6, sigma=1,
                    nan_policy=nan_policy)

print(result.fit_report())

# make sure nans are removed for plotting:
x_ = x[np.where(np.isfinite(y))]
y_ = y[np.where(np.isfinite(y))]

plt.plot(x_, y_, 'o')
plt.plot(x_, result.init_fit, '--', label='initial fit')
plt.plot(x_, result.best_fit, '-', label='best fit')
plt.legend()
plt.show()
# <end examples/doc_model_with_nan_policy.py>
