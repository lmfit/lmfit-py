"""Tests for saving/loading Models and ModelResults."""
import os
import time

import numpy as np
from numpy.testing import assert_allclose
import pytest

from lmfit import Parameters
import lmfit.jsonutils
from lmfit.lineshapes import gaussian, lorentzian
from lmfit.model import (Model, ModelResult, load_model, load_modelresult,
                         save_model, save_modelresult)
from lmfit.models import ExponentialModel, GaussianModel, VoigtModel
from lmfit_testutils import assert_between, assert_param_between

y, x = np.loadtxt(os.path.join(os.path.dirname(__file__), '..',
                               'examples', 'NIST_Gauss2.dat')).T

SAVE_MODEL = 'model_1.sav'
SAVE_MODELRESULT = 'modelresult_1.sav'


def clear_savefile(fname):
    """Remove save files so that tests start fresh."""
    try:
        os.unlink(fname)
    except OSError:
        pass


def wait_for_file(fname, timeout=10):
    """Check whether file is created within certain amount of time."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        if os.path.exists(fname):
            return True
        time.sleep(0.05)
    return False


def create_model_params(x, y):
    """Create the model and parameters."""
    exp_mod = ExponentialModel(prefix='exp_')
    params = exp_mod.guess(y, x=x)

    gauss1 = GaussianModel(prefix='g1_')
    params.update(gauss1.make_params())

    gauss2 = GaussianModel(prefix='g2_')
    params.update(gauss2.make_params())

    params['g1_center'].set(value=105, min=75, max=125)
    params['g1_sigma'].set(value=15, min=3)
    params['g1_amplitude'].set(value=2000, min=10)

    params['g2_center'].set(value=155, min=125, max=175)
    params['g2_sigma'].set(value=15, min=3)
    params['g2_amplitude'].set(value=2000, min=10)

    model = gauss1 + gauss2 + exp_mod
    return model, params


def check_fit_results(result):
    """Check the result of optimization."""
    assert result.nvarys == 8
    assert_between(result.chisqr, 1000, 1500)
    assert_between(result.aic, 400, 450)

    pars = result.params
    assert_param_between(pars['exp_decay'], 90, 92)
    assert_param_between(pars['exp_amplitude'], 98, 101)

    assert_param_between(pars['g1_sigma'], 16, 17)
    assert_param_between(pars['g1_center'], 106, 109)
    assert_param_between(pars['g1_amplitude'], 4100, 4500)
    assert_param_between(pars['g1_fwhm'], 38, 42)
    assert_param_between(pars['g1_height'], 100, 103)

    assert_param_between(pars['g2_sigma'], 10, 15)
    assert_param_between(pars['g2_center'], 150, 160)
    assert_param_between(pars['g2_amplitude'], 2100, 2900)
    assert_param_between(pars['g2_fwhm'], 30, 34)
    assert_param_between(pars['g2_height'], 70, 75)


@pytest.mark.parametrize("dill", [False, True])
def test_save_load_model(dill):
    """Save/load Model with/without dill."""
    if dill:
        pytest.importorskip("dill")
    else:
        lmfit.jsonutils.HAS_DILL = False

    # create/save Model and perform some tests
    model, _pars = create_model_params(x, y)
    save_model(model, SAVE_MODEL)

    file_exists = wait_for_file(SAVE_MODEL, timeout=10)
    assert file_exists

    with open(SAVE_MODEL, 'r') as fh:
        text = fh.read()
    assert 1000 < len(text) < 2500

    # load the Model, perform fit and assert results
    saved_model = load_model(SAVE_MODEL)
    params = saved_model.make_params()

    params['exp_decay'].set(100)
    params['exp_amplitude'].set(100)
    params['g1_center'].set(105, min=75, max=125)
    params['g1_sigma'].set(15, min=3)
    params['g1_amplitude'].set(2000, min=10)

    params['g2_center'].set(155, min=125, max=175)
    params['g2_sigma'].set(15, min=3)
    params['g2_amplitude'].set(2000, min=10)

    result = saved_model.fit(y, params, x=x)
    check_fit_results(result)

    clear_savefile(SAVE_MODEL)


@pytest.mark.parametrize("dill", [False, True])
def test_save_load_modelresult(dill):
    """Save/load ModelResult with/without dill."""
    if dill:
        pytest.importorskip("dill")
    else:
        lmfit.jsonutils.HAS_DILL = False

    # create model, perform fit, save ModelResult and perform some tests
    model, params = create_model_params(x, y)
    result = model.fit(y, params, x=x)
    save_modelresult(result, SAVE_MODELRESULT)

    file_exists = wait_for_file(SAVE_MODELRESULT, timeout=10)
    assert file_exists

    text = ''
    with open(SAVE_MODELRESULT, 'r') as fh:
        text = fh.read()
    assert_between(len(text), 8000, 25000)

    # load the saved ModelResult from file and compare results
    result_saved = load_modelresult(SAVE_MODELRESULT)
    check_fit_results(result_saved)

    clear_savefile(SAVE_MODEL)


def test_saveload_modelresult_attributes():
    """Test for restoring all attributes of the ModelResult."""
    model, params = create_model_params(x, y)
    result = model.fit(y, params, x=x)
    save_modelresult(result, SAVE_MODELRESULT)

    time.sleep(0.25)
    file_exists = wait_for_file(SAVE_MODELRESULT, timeout=10)
    assert file_exists
    time.sleep(0.25)

    loaded = load_modelresult(SAVE_MODELRESULT)

    assert len(result.data) == len(loaded.data)
    assert_allclose(result.data, loaded.data)

    for pname in result.params.keys():
        assert_allclose(result.init_params[pname].value,
                        loaded.init_params[pname].value)

    clear_savefile(SAVE_MODELRESULT)


def test_saveload_modelresult_exception():
    """Make sure the proper exceptions are raised when needed."""
    model, _pars = create_model_params(x, y)
    save_model(model, SAVE_MODEL)

    with pytest.raises(AttributeError, match=r'needs saved ModelResult'):
        load_modelresult(SAVE_MODEL)
    clear_savefile(SAVE_MODEL)


def test_saveload_modelresult_roundtrip():
    """Test for modelresult.loads()/dumps() and repeating that"""
    def mfunc(x, a, b):
        return a * (x-b)

    model = Model(mfunc)
    params = model.make_params(a=0.0, b=3.0)

    xx = np.linspace(-5, 5, 201)
    yy = 0.5 * (xx - 0.22) + np.random.normal(scale=0.01, size=len(xx))

    result1 = model.fit(yy, params, x=xx)

    result2 = ModelResult(model, Parameters())
    result2.loads(result1.dumps(), funcdefs={'mfunc': mfunc})

    result3 = ModelResult(model, Parameters())
    result3.loads(result2.dumps(), funcdefs={'mfunc': mfunc})

    assert result3 is not None
    assert_param_between(result2.params['a'], 0.48, 0.52)
    assert_param_between(result2.params['b'], 0.20, 0.25)
    assert_param_between(result3.params['a'], 0.48, 0.52)
    assert_param_between(result3.params['b'], 0.20, 0.25)


def test_saveload_usersyms():
    """Test save/load of modelresult with non-trivial user symbols,
    this example uses a VoigtModel, wheree `wofz()` is used in a
    constraint expression"""
    x = np.linspace(0, 20, 501)
    y = gaussian(x, 1.1, 8.5, 2) + lorentzian(x, 1.7, 8.5, 1.5)
    np.random.seed(20)
    y = y + np.random.normal(size=len(x), scale=0.025)

    model = VoigtModel()
    pars = model.guess(y, x=x)
    result = model.fit(y, pars, x=x)

    savefile = 'tmpvoigt_modelresult.sav'
    save_modelresult(result, savefile)

    assert_param_between(result.params['sigma'], 0.7, 2.1)
    assert_param_between(result.params['center'], 8.4, 8.6)
    assert_param_between(result.params['height'], 0.2, 1.0)

    time.sleep(0.25)
    result2 = load_modelresult(savefile)

    assert_param_between(result2.params['sigma'], 0.7, 2.1)
    assert_param_between(result2.params['center'], 8.4, 8.6)
    assert_param_between(result2.params['height'], 0.2, 1.0)
