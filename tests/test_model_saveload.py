"""Tests for saving/loading Models and ModelResults."""

import json
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
from lmfit.models import (ExponentialModel, ExpressionModel, GaussianModel,
                          VoigtModel)

y, x = np.loadtxt(os.path.join(os.path.dirname(__file__), '..',
                               'examples', 'NIST_Gauss2.dat')).T

SAVE_MODEL = 'model_1.sav'
SAVE_MODELRESULT = 'modelresult_1.sav'

MODELRESULT_LMFIT_1_0 = 'gauss_modelresult_lmfit100.sav'


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
    assert_allclose(result.chisqr, 1247.528209, rtol=1.0e-5)
    assert_allclose(result.aic, 417.864631, rtol=1.0e-5)

    pars = result.params
    assert_allclose(pars['exp_decay'], 90.950886, rtol=1.0e-5)
    assert_allclose(pars['exp_amplitude'], 99.018328, rtol=1.0e-5)

    assert_allclose(pars['g1_sigma'], 16.672575, rtol=1.0e-5)
    assert_allclose(pars['g1_center'], 107.030954, rtol=1.0e-5)
    assert_allclose(pars['g1_amplitude'], 4257.773192, rtol=1.0e-5)
    assert_allclose(pars['g1_fwhm'], 39.260914, rtol=1.0e-5)
    assert_allclose(pars['g1_height'], 101.880231, rtol=1.0e-5)

    assert_allclose(pars['g2_sigma'], 13.806948, rtol=1.0e-5)
    assert_allclose(pars['g2_center'], 153.270101, rtol=1.0e-5)
    assert_allclose(pars['g2_amplitude'], 2493.417703, rtol=1.0e-5)
    assert_allclose(pars['g2_fwhm'], 32.512878, rtol=1.0e-5)
    assert_allclose(pars['g2_height'], 72.045593, rtol=1.0e-5)


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

    with open(SAVE_MODEL) as fh:
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
    with open(SAVE_MODELRESULT) as fh:
        text = fh.read()
    assert 12000 < len(text) < 60000  # depending on whether dill is present

    # load the saved ModelResult from file and compare results
    result_saved = load_modelresult(SAVE_MODELRESULT)
    assert result_saved.residual is not None
    check_fit_results(result_saved)

    clear_savefile(SAVE_MODEL)


def test_load_legacy_modelresult():
    """Load legacy ModelResult."""
    fname = os.path.join(os.path.dirname(__file__), MODELRESULT_LMFIT_1_0)
    result_saved = load_modelresult(fname)
    assert result_saved is not None


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


@pytest.mark.parametrize("method", ['leastsq', 'nelder', 'powell', 'cobyla',
                                    'bfgs', 'lbfgsb', 'differential_evolution',
                                    'brute', 'basinhopping', 'ampgo', 'shgo',
                                    'dual_annealing'])
def test_saveload_modelresult_roundtrip(method):
    """Test for modelresult.loads()/dumps() and repeating that."""
    def mfunc(x, a, b):
        return a * (x-b)

    model = Model(mfunc)
    params = model.make_params(a=0.1, b=3.0)
    params['a'].set(min=.01, max=1, brute_step=0.01)
    params['b'].set(min=.01, max=3.1, brute_step=0.01)

    np.random.seed(2020)
    xx = np.linspace(-5, 5, 201)
    yy = 0.5 * (xx - 0.22) + np.random.normal(scale=0.01, size=xx.size)

    result1 = model.fit(yy, params=params, x=xx, method=method)

    result2 = ModelResult(model, Parameters())
    result2.loads(result1.dumps(), funcdefs={'mfunc': mfunc})

    result3 = ModelResult(model, Parameters())
    result3.loads(result2.dumps(), funcdefs={'mfunc': mfunc})

    assert result3 is not None
    assert_allclose(result2.params['a'], 0.5, rtol=1.0e-2)
    assert_allclose(result2.params['b'], 0.22, rtol=1.0e-2)
    assert_allclose(result3.params['a'], 0.50, rtol=1.0e-2)
    assert_allclose(result3.params['b'], 0.22, rtol=1.0e-2)


def test_saveload_modelresult_expression_model():
    """Test for ModelResult.loads()/dumps() for ExpressionModel.

    * make sure that the loaded ModelResult has `init_params` and `init_fit`.

    """
    savefile = 'expr_modres.txt'
    x = np.linspace(-10, 10, 201)
    amp, cen, wid = 3.4, 1.8, 0.5

    y = amp * np.exp(-(x-cen)**2 / (2*wid**2)) / (np.sqrt(2*np.pi)*wid)
    y = y + np.random.normal(size=x.size, scale=0.01)

    gmod = ExpressionModel("amp * exp(-(x-cen)**2 /(2*wid**2))/(sqrt(2*pi)*wid)")
    result = gmod.fit(y, x=x, amp=5, cen=5, wid=1)
    save_modelresult(result, savefile)
    time.sleep(0.25)

    result2 = load_modelresult(savefile)

    assert result2 is not None
    assert result2.residual is not None
    assert result2.init_fit is not None
    assert_allclose((result2.init_fit - result.init_fit).sum() + 1.00, 1.00,
                    rtol=1.0e-2)
    os.unlink(savefile)


def test_saveload_usersyms():
    """Test save/load of ModelResult with non-trivial user symbols.

    This example uses a VoigtModel, where `wofz()` is used in a constraint
    expression.

    """
    x = np.linspace(0, 20, 501)
    y = gaussian(x, 1.1, 8.5, 2) + lorentzian(x, 1.7, 8.5, 1.5)
    np.random.seed(20)
    y = y + np.random.normal(size=len(x), scale=0.025)

    model = VoigtModel()
    pars = model.guess(y, x=x)
    result = model.fit(y, pars, x=x)

    savefile = 'tmpvoigt_modelresult.sav'
    save_modelresult(result, savefile)

    assert_allclose(result.params['sigma'], 1.075487, rtol=1.0e-5)
    assert_allclose(result.params['center'], 8.489738, rtol=1.0e-5)
    assert_allclose(result.params['height'], 0.557778, rtol=1.0e-5)

    time.sleep(0.25)
    result2 = load_modelresult(savefile)

    assert result2.residual is not None
    assert_allclose(result2.params['sigma'], 1.075487, rtol=1.0e-5)
    assert_allclose(result2.params['center'], 8.489738, rtol=1.0e-5)
    assert_allclose(result2.params['height'], 0.557778, rtol=1.0e-5)


def test_modelresult_summary():
    """Test summary() method of ModelResult.
    """
    x = np.linspace(0, 20, 501)
    y = gaussian(x, 1.1, 8.5, 2) + lorentzian(x, 1.7, 8.5, 1.5)
    np.random.seed(20)
    y = y + np.random.normal(size=len(x), scale=0.025)

    model = VoigtModel()
    pars = model.guess(y, x=x)
    result = model.fit(y, pars, x=x)

    summary = result.summary()

    assert isinstance(summary, dict)

    for attr in ('ndata', 'nvarys', 'nfree', 'chisqr', 'redchi', 'aic',
                 'bic', 'rsquared', 'nfev', 'max_nfev', 'aborted',
                 'errorbars', 'success', 'message', 'lmdif_message', 'ier',
                 'nan_policy', 'scale_covar', 'calc_covar', 'ci_out',
                 'col_deriv', 'flatchain', 'call_kws', 'var_names',
                 'user_options', 'kws', 'init_values', 'best_values'):
        val = summary.get(attr, '__INVALID__')
        assert val != '__INVALID__'

    assert len(json.dumps(summary)) > 100
