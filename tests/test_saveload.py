import os
import time

import numpy as np

import lmfit.jsonutils
from lmfit.model import (load_model, load_modelresult, save_model,
                         save_modelresult)
from lmfit.models import ExponentialModel, GaussianModel
from lmfit_testutils import assert_between, assert_param_between

try:
    import dill
    DILL_AVAILABLE = True
except ImportError:
    DILL_AVAILABLE = False

dat = np.loadtxt(os.path.join(os.path.dirname(__file__), '..', 'examples',
                              'NIST_Gauss2.dat'))
XDAT = dat[:, 1]
YDAT = dat[:, 0]

SAVE_MODEL = 'model_1.sav'
SAVE_MODELRESULT = 'modelresult_1.sav'


def clear_savefile(fname):
    """remove save files so that tests are fresh"""
    try:
        os.unlink(fname)
    except OSError:
        pass


def create_model_params(x, y):
    exp_mod = ExponentialModel(prefix='exp_')
    params = exp_mod.guess(y, x=x)

    gauss1 = GaussianModel(prefix='g1_')
    params.update(gauss1.make_params())

    gauss2 = GaussianModel(prefix='g2_')
    params.update(gauss2.make_params())

    params['g1_center'].set(105, min=75, max=125)
    params['g1_sigma'].set(15, min=3)
    params['g1_amplitude'].set(2000, min=10)

    params['g2_center'].set(155, min=125, max=175)
    params['g2_sigma'].set(15, min=3)
    params['g2_amplitude'].set(2000, min=10)

    model = gauss1 + gauss2 + exp_mod
    return model, params


def check_fit_results(result):
    assert(result.nvarys == 8)
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


def wait_for_file(fname, timeout=10):
    end_time = time.time() + timeout
    while time.time() < end_time:
        if os.path.exists(fname):
            return True
        time.sleep(0.05)
    return False


def do_save_model(with_dill=True):
    x, y = XDAT, YDAT
    if DILL_AVAILABLE and with_dill:
        lmfit.jsonutils.HAS_DILL = True
    else:
        lmfit.jsonutils.HAS_DILL = False

    model, params = create_model_params(x, y)

    save_model(model, SAVE_MODEL)
    file_exists = wait_for_file(SAVE_MODEL, timeout=10)
    assert(file_exists)

    text = ''
    with open(SAVE_MODEL, 'r') as fh:
        text = fh.read()
    assert_between(len(text), 1000, 2500)


def do_load_model():
    x, y = XDAT, YDAT
    model = load_model(SAVE_MODEL)
    params = model.make_params()

    params['exp_decay'].set(100)
    params['exp_amplitude'].set(100)
    params['g1_center'].set(105, min=75, max=125)
    params['g1_sigma'].set(15, min=3)
    params['g1_amplitude'].set(2000, min=10)

    params['g2_center'].set(155, min=125, max=175)
    params['g2_sigma'].set(15, min=3)
    params['g2_amplitude'].set(2000, min=10)

    result = model.fit(y, params, x=x)
    check_fit_results(result)


def do_save_modelresult(with_dill=True):
    x, y = XDAT, YDAT
    if DILL_AVAILABLE and with_dill:
        lmfit.jsonutils.HAS_DILL = True
    else:
        lmfit.jsonutils.HAS_DILL = False

    model, params = create_model_params(x, y)

    result = model.fit(y, params, x=x)
    save_modelresult(result, SAVE_MODELRESULT)
    file_exists = wait_for_file(SAVE_MODELRESULT, timeout=10)
    assert(file_exists)

    text = ''
    with open(SAVE_MODELRESULT, 'r') as fh:
        text = fh.read()
    assert_between(len(text), 8000, 25000)


def do_load_modelresult():
    result = load_modelresult(SAVE_MODELRESULT)
    check_fit_results(result)


def test_saveload_model_withdill():
    clear_savefile(SAVE_MODEL)
    do_save_model(with_dill=True)
    do_load_model()


def test_saveload_model_nodill():
    clear_savefile(SAVE_MODEL)
    do_save_model(with_dill=False)
    do_load_model()


def test_saveload_modelresult_withdill():
    clear_savefile(SAVE_MODELRESULT)
    do_save_modelresult(with_dill=True)
    do_load_modelresult()


def test_saveload_modelresult_nodill():
    clear_savefile(SAVE_MODELRESULT)
    do_save_modelresult(with_dill=False)
    do_load_modelresult()


def test_saveload_modelresult_items():
    clear_savefile(SAVE_MODELRESULT)
    x, y = XDAT, YDAT
    model, params = create_model_params(x, y)
    result = model.fit(y, params, x=x)

    save_modelresult(result, SAVE_MODELRESULT)

    time.sleep(0.25)
    file_exists = wait_for_file(SAVE_MODELRESULT, timeout=10)

    assert(file_exists)
    time.sleep(0.25)

    loaded = load_modelresult(SAVE_MODELRESULT)
    assert(len(result.data) == len(loaded.data))
    assert(abs(max(result.data) - max(loaded.data)) < 0.001)

    for pname in result.params.keys():
        assert(abs(result.init_params[pname].value -
                   loaded.init_params[pname].value) < 0.001)
