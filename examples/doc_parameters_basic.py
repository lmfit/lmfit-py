# <examples/doc_parameters_basic.py>
import numpy as np

from lmfit import Minimizer, Parameters, create_params, report_fit

# create data to be fitted
x = np.linspace(0, 15, 301)
np.random.seed(2021)
data = (5.0 * np.sin(2.0*x - 0.1) * np.exp(-x*x*0.025) +
        np.random.normal(size=x.size, scale=0.2))


# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """Model a decaying sine wave and subtract data."""
    amp = params['amp']
    shift = params['shift']
    omega = params['omega']
    decay = params['decay']
    model = amp * np.sin(x*omega + shift) * np.exp(-x*x*decay)
    return model - data


# create a set of Parameters
params = Parameters()
params.add('amp', value=10, min=0)
params.add('decay', value=0.1)
params.add('shift', value=0.0, min=-np.pi/2., max=np.pi/2.)
params.add('omega', value=3.0)

# ... or use
params = create_params(amp=dict(value=10, min=0),
                       decay=0.1,
                       omega=3,
                       shift=dict(value=0, min=-np.pi/2, max=np.pi/2))

# do fit, here with the default leastsq algorithm
minner = Minimizer(fcn2min, params, fcn_args=(x, data))
result = minner.minimize()

# calculate final result
final = data + result.residual

# write error report
report_fit(result)

# try to plot results
try:
    import matplotlib.pyplot as plt
    plt.plot(x, data, '+')
    plt.plot(x, final)
    plt.show()
except ImportError:
    pass
# <end of examples/doc_parameters_basic.py>
