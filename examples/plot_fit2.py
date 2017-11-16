#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np

import lmfit

# construct data
x = np.linspace(-4, 4)
y = np.exp(-x**2)

noise = np.random.randn(x.size) * 0.1
y += noise

# define model and fit
model_gaussian = lmfit.models.GaussianModel()
model_gaussian.guess(y, x=x)
fit_gaussian = model_gaussian.fit(y, x=x, weights=1/noise**2)

# plot the with with customization
fig, gridspec = fit_gaussian.plot(fig_kws=dict(figsize=[8, 7]),
                                  ax_fit_kws=dict(title='The gaussian fit'),
                                  initfmt='k:', datafmt='ks',
                                  fit_kws=dict(lw=2, color='red'),
                                  data_kws=dict(ms=8, markerfacecolor='white'))

fig.set_tight_layout(True)
plt.show()
