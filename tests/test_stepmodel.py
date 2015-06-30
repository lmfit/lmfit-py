import numpy as np
from lmfit import fit_report
from lmfit.models import StepModel, ConstantModel
from lmfit_testutils import assert_paramval, assert_paramattr

def get_data():
    x  = np.linspace(0, 10, 201)
    dat = np.ones_like(x)
    dat[:48] = 0.0
    dat[48:77] = np.arange(77-48)/(77.0-48)
    dat = dat +  5e-2*np.random.randn(len(x))
    dat = 110.2 * dat + 12.0
    return x, dat

def test_stepmodel_linear():
    x, y = get_data()
    stepmod = StepModel(form='linear')
    const = ConstantModel()
    pars = stepmod.guess(y, x)
    pars = pars + const.make_params(c=3*y.min())
    mod = stepmod + const

    out = mod.fit(y, pars, x=x)

    assert(out.nfev > 5)
    assert(out.nvarys == 4)
    assert(out.chisqr > 1)
    assert(out.params['c'].value > 3)
    assert(out.params['center'].value > 1)
    assert(out.params['center'].value < 4)
    assert(out.params['sigma'].value > 0.5)
    assert(out.params['sigma'].value < 3.5)
    assert(out.params['amplitude'].value > 50)


def test_stepmodel_erf():
    x, y = get_data()
    stepmod = StepModel(form='linear')
    const = ConstantModel()
    pars = stepmod.guess(y, x)
    pars = pars + const.make_params(c=3*y.min())
    mod = stepmod + const

    out = mod.fit(y, pars, x=x)

    assert(out.nfev > 5)
    assert(out.nvarys == 4)
    assert(out.chisqr > 1)
    assert(out.params['c'].value > 3)
    assert(out.params['center'].value > 1)
    assert(out.params['center'].value < 4)
    assert(out.params['amplitude'].value > 50)
    assert(out.params['sigma'].value > 0.2)
    assert(out.params['sigma'].value < 1.5)

if __name__ == '__main__':
    # test_stepmodel_linear()
    test_stepmodel_erf()
