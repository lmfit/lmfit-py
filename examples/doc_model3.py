#!/usr/bin/env python
#<examples/model_doc3.py>

import numpy as np
from lmfit import Model, CompositeModel
from lmfit.lineshapes import step, gaussian

import matplotlib.pyplot as plt

# create data from broadened step
npts = 201
x = np.linspace(0, 10, npts)
y = step(x, amplitude=12.5, center=4.5, sigma=0.88, form='erf')
y = y + np.random.normal(size=npts, scale=0.35)

def jump(x, mid):
    "heaviside step function"
    o = np.zeros(len(x))
    imid = max(np.where(x<=mid)[0])
    o[imid:] = 1.0
    return o

def convolve(arr, kernel):
    # simple convolution of two arrays
    npts = min(len(arr), len(kernel))
    pad  = np.ones(npts)
    tmp  = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out  = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts)/2)
    return out[noff:noff+npts]
#
# create Composite Model using the custom convolution operator
mod  = CompositeModel(Model(jump), Model(gaussian), convolve)

pars = mod.make_params(amplitude=1, center=3.5, sigma=1.5, mid=5.0)

# 'mid' and 'center' should be completely correlated, and 'mid' is
# used as an integer index, so a very poor fit variable:
pars['mid'].vary = False

# fit this model to data array y
result =  mod.fit(y, params=pars, x=x)

print(result.fit_report())

plot_components = False

# plot results
plt.plot(x, y,         'bo')
if plot_components:
    # generate components
    comps = result.eval_components(x=x)
    plt.plot(x, 10*comps['jump'], 'k--')
    plt.plot(x, 10*comps['gaussian'], 'r-')
else:
    plt.plot(x, result.init_fit, 'k--')
    plt.plot(x, result.best_fit, 'r-')
plt.show()
# #<end examples/model_doc3.py>
