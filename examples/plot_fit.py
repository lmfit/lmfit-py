import lmfit
import numpy as np
from matplotlib import pyplot as plt

# construct data
x = np.linspace(-4, 4)
y = np.exp(-x**2)

noise = np.random.randn(x.size) * 0.1
y += noise

# define model and fit
model = lmfit.models.GaussianModel()
model.guess(y, x=x)
fit = model.fit(y, x=x, weights=1/noise**2)

fig = fit.plot()

# customize plot post-factum
ax_residuals, ax_fit = fig.get_axes()
ax_residuals.set_title('The residuals')
ax_fit.set_title('An example gaussian fit')
ax_residuals.get_legend().set_visible(False)
ax_fit.get_legend().texts[0].set_size('small')

plt.show()
