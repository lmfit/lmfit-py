# <examples/doc_model_composite.py>
import matplotlib.pyplot as plt
import numpy as np

from lmfit import CompositeModel, Model
from lmfit.lineshapes import gaussian, step

# create data from broadened step
x = np.linspace(0, 10, 201)
y = step(x, amplitude=12.5, center=4.5, sigma=0.88, form='erf')
np.random.seed(0)
y = y + np.random.normal(scale=0.35, size=x.size)


def jump(x, mid):
    """Heaviside step function."""
    o = np.zeros(x.size)
    imid = max(np.where(x <= mid)[0])
    o[imid:] = 1.0
    return o


def convolve(arr, kernel):
    """Simple convolution of two arrays."""
    npts = min(arr.size, kernel.size)
    pad = np.ones(npts)
    tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts) / 2)
    return out[noff:noff+npts]


# create Composite Model using the custom convolution operator
mod = CompositeModel(Model(jump), Model(gaussian), convolve)

# create parameters for model.  Note that 'mid' and 'center' will be highly
# correlated. Since 'mid' is used as an integer index, it will be very
# hard to fit, so we fix its value
pars = mod.make_params(amplitude=dict(value=1, min=0),
                       center=3.5,
                       sigma=dict(value=1.5, min=0),
                       mid=dict(value=4, vary=False))

# fit this model to data array y
result = mod.fit(y, params=pars, x=x)

print(result.fit_report())

# generate components
comps = result.eval_components(x=x)

# plot results
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

axes[0].plot(x, y, 'bo')
axes[0].plot(x, result.init_fit, 'k--', label='initial fit')
axes[0].plot(x, result.best_fit, 'r-', label='best fit')
axes[0].legend()

axes[1].plot(x, y, 'bo')
axes[1].plot(x, 10*comps['jump'], 'k--', label='Jump component')
axes[1].plot(x, 10*comps['gaussian'], 'r-', label='Gaussian component')
axes[1].legend()

plt.show()
# <end examples/doc_model_composite.py>
