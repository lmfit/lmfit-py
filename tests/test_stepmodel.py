import numpy as np

from lmfit.models import ConstantModel, RectangleModel, StepModel


def get_data():
    np.random.seed(2021)
    x = np.linspace(0, 10, 201)
    dat = np.ones_like(x)
    dat[:48] = 0.0
    dat[48:77] = np.arange(77-48)/(77.0-48)
    dat = dat + 5e-2*np.random.randn(len(x))
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

    assert out.nfev > 5
    assert out.nvarys == 4
    assert out.chisqr > 1
    assert out.params['c'].value > 3
    assert out.params['center'].value > 1
    assert out.params['center'].value < 4
    assert out.params['sigma'].value > 0.5
    assert out.params['sigma'].value < 3.5
    assert out.params['amplitude'].value > 50


def test_stepmodel_erf():
    x, y = get_data()
    stepmod = StepModel(form='linear')
    const = ConstantModel()
    pars = stepmod.guess(y, x)
    pars = pars + const.make_params(c=3*y.min())
    mod = stepmod + const

    out = mod.fit(y, pars, x=x)

    assert out.nfev > 5
    assert out.nvarys == 4
    assert out.chisqr > 1
    assert out.params['c'].value > 3
    assert out.params['center'].value > 1
    assert out.params['center'].value < 4
    assert out.params['amplitude'].value > 50
    assert out.params['sigma'].value > 0.2
    assert out.params['sigma'].value < 1.5


def test_stepmodel_stepdown():
    x = np.linspace(0, 50, 201)
    y = np.ones_like(x)
    y[129:] = 0.0
    y[109:129] = 1.0 - np.arange(20)/20
    y = y + 5e-2*np.random.randn(len(x))
    stepmod = StepModel(form='linear')
    pars = stepmod.guess(y, x)

    out = stepmod.fit(y, pars, x=x)

    assert out.nfev > 10
    assert out.nvarys == 3
    assert out.chisqr > 0.2
    assert out.chisqr < 5.0
    assert out.params['center'].value > 28
    assert out.params['center'].value < 32
    assert out.params['amplitude'].value > 0.5
    assert out.params['sigma'].value < -2.0
    assert out.params['sigma'].value > -8.0


def test_rectangle():
    x = np.linspace(0, 50, 201)
    y = np.ones_like(x) * 2.5
    y[:33] = 0.0
    y[162:] = 0.0
    y[33:50] = 2.5*np.arange(50-33)/(50-33)
    y[155:162] = 2.5 * (1 - np.arange(162-155)/(162-155))
    y = y + 5e-2*np.random.randn(len(x))
    stepmod = RectangleModel(form='linear')
    pars = stepmod.guess(y, x)

    out = stepmod.fit(y, pars, x=x)

    assert out.nfev > 10
    assert out.nvarys == 5
    assert out.chisqr > 0.2
    assert out.chisqr < 5.0
    assert out.params['center1'].value > 8
    assert out.params['center1'].value < 14
    assert out.params['amplitude'].value > 2.0
    assert out.params['amplitude'].value < 10.0
    assert out.params['sigma1'].value > 1.0
    assert out.params['sigma1'].value < 5.0
    assert out.params['sigma2'].value > 0.3
    assert out.params['sigma2'].value < 2.5
