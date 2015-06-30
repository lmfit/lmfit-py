#
# example fitting to multiple (simulated) data sets
#
import numpy as np
from lmfit import minimize, Parameters, fit_report
from lmfit.lineshapes import gaussian

def gauss_dataset(params, i, x):
    """calc gaussian from params for data set i
    using simple, hardwired naming convention"""
    amp = params['amp_%i' % (i+1)].value
    cen = params['cen_%i' % (i+1)].value
    sig = params['sig_%i' % (i+1)].value
    return gaussian(x, amp, cen, sig)

def objective(params, x, data):
    """ calculate total residual for fits to several data sets held
    in a 2-D array, and modeled by Gaussian functions"""
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - gauss_dataset(params, i, x)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()

def test_multidatasets():
    # create 5 datasets
    x  = np.linspace( -1, 2, 151)
    data = []
    for i in np.arange(5):
        amp  =  2.60 + 1.50*np.random.rand()
        cen  = -0.20 + 1.50*np.random.rand()
        sig  =  0.25 + 0.03*np.random.rand()
        dat  = gaussian(x, amp, cen, sig) + \
               np.random.normal(size=len(x), scale=0.1)
        data.append(dat)

    # data has shape (5, 151)
    data = np.array(data)
    assert(data.shape) == (5, 151)

    # create 5 sets of parameters, one per data set
    pars = Parameters()
    for iy, y in enumerate(data):
        pars.add( 'amp_%i' % (iy+1), value=0.5, min=0.0,  max=200)
        pars.add( 'cen_%i' % (iy+1), value=0.4, min=-2.0,  max=2.0)
        pars.add( 'sig_%i' % (iy+1), value=0.3, min=0.01, max=3.0)

    # but now constrain all values of sigma to have the same value
    # by assigning sig_2, sig_3, .. sig_5 to be equal to sig_1
    for iy in (2, 3, 4, 5):
        pars['sig_%i' % iy].expr='sig_1'

    # run the global fit to all the data sets
    out = minimize(objective, pars, args=(x, data))

    assert(len(pars) == 15)
    assert(out.nvarys == 11)
    assert(out.nfev  > 15)
    assert(out.chisqr > 1.0)
    assert(pars['amp_1'].value > 0.1)
    assert(pars['sig_1'].value > 0.1)
    assert(pars['sig_2'].value == pars['sig_1'].value)

    ## plot the data sets and fits
    #  plt.figure()
    #  for i in range(5):
    #      y_fit = gauss_dataset(pars, i, x)
    #      plt.plot(x, data[i, :], 'o', x, y_fit, '-')
    #  plt.show()

if __name__ == '__main__':
    test_multidatasets()
