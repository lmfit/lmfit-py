import numpy as np
from lmfit import minimize, Parameters, Parameter, report_fit
from lmfit_testutils import assert_paramval, assert_paramattr


def test_basic():
    # create data to be fitted
    x = np.linspace(0, 15, 301)
    data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
            np.random.normal(size=len(x), scale=0.2) )

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        amp = params['amp'].value
        shift = params['shift'].value
        omega = params['omega'].value
        decay = params['decay'].value

        model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
        return model - data

    # create a set of Parameters
    params = Parameters()
    params.add('amp',   value= 10,  min=0)
    params.add('decay', value= 0.1)
    params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
    params.add('omega', value= 3.0)

    # do fit, here with leastsq model
    result = minimize(fcn2min, params, args=(x, data))

    # calculate final result
    final = data + result.residual

    # report_fit(result)

    assert(result.nfev >   5)
    assert(result.nfev < 500)
    assert(result.chisqr > 1)
    assert(result.nvarys == 4)
    assert_paramval(result.params['amp'],   5.03, tol=0.05)
    assert_paramval(result.params['omega'], 2.0, tol=0.05)


if __name__ == '__main__':
    test_basic()
