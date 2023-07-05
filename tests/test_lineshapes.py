"""Tests for lineshape functions."""

import inspect

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

import lmfit
from lmfit.lineshapes import not_zero, tiny


@pytest.mark.parametrize(
    "value, expected_result",
    [(1, 1.0), (-1, -1.0), (0, tiny), (-0, -tiny)])
def test_not_zero(value, expected_result):
    """Test that not_zero gives the expected results"""
    assert_almost_equal(not_zero(value), expected_result)


@pytest.mark.parametrize("lineshape", lmfit.lineshapes.functions)
def test_no_ZeroDivisionError_and_finite_output(lineshape):
    """Tests for finite output and ZeroDivisionError is not raised."""
    xvals = np.linspace(0, 10, 100)

    func = getattr(lmfit.lineshapes, lineshape)
    assert callable(func)
    sig = inspect.signature(func)

    # set the following function arguments:
    #   x = xvals
    #   center = 0.5*(max(xvals)-min(xvals))
    #   center1 = 0.25*(max(xvals)-min(xvals))
    #   center2 = 0.75*(max(xvals)-min(xvals))
    #   form = default value (i.e., 'linear' or 'bose')
    xvals_mid_range = xvals.mean()
    zero_pars = [par_name for par_name in sig.parameters.keys() if par_name
                 not in ('x', 'form')]

    for par_zero in zero_pars:
        fnc_args = []
        for par in sig.parameters.keys():
            if par == 'x':
                fnc_args.append(xvals)
            elif par == 'center':
                fnc_args.append(0.5*xvals_mid_range)
            elif par == 'center1':
                fnc_args.append(0.25*xvals_mid_range)
            elif par == 'center2':
                fnc_args.append(0.75*xvals_mid_range)
            elif par == par_zero:
                fnc_args.append(0.0)
            else:
                fnc_args.append(sig.parameters[par].default)

        fnc_output = func(*fnc_args)
        assert len(xvals) == len(fnc_output)
        assert np.all(np.isfinite(fnc_output))


@pytest.mark.parametrize("lineshape", lmfit.lineshapes.functions)
def test_x_float_value(lineshape):
    """Test lineshapes when x is not an array but a float."""
    xval = 7.0

    func = getattr(lmfit.lineshapes, lineshape)
    sig = inspect.signature(func)

    fnc_args = [xval]

    for par in [par_name for par_name in sig.parameters.keys()
                if par_name != 'x']:
        fnc_args.append(sig.parameters[par].default)

    fnc_output = func(*fnc_args)
    assert isinstance(fnc_output, float)


rising_form = ['erf', 'logistic', 'atan', 'arctan', 'linear', 'unknown']


@pytest.mark.parametrize("form", rising_form)
@pytest.mark.parametrize("lineshape", ['step', 'rectangle'])
def test_form_argument_step_rectangle(form, lineshape):
    """Test 'form' argument for step- and rectangle-functions."""
    xvals = np.linspace(0, 10, 100)

    func = getattr(lmfit.lineshapes, lineshape)
    sig = inspect.signature(func)

    fnc_args = [xvals]
    for par in [par_name for par_name in sig.parameters.keys()
                if par_name != 'x']:
        if par == 'form':
            fnc_args.append(form)
        else:
            fnc_args.append(sig.parameters[par].default)

    if form == 'unknown':
        msg = r"Invalid value .* for argument .*; should be one of .*"
        with pytest.raises(ValueError, match=msg):
            func(*fnc_args)
    else:
        fnc_output = func(*fnc_args)
        assert len(fnc_output) == len(xvals)


@pytest.mark.parametrize('form', rising_form)
@pytest.mark.parametrize('lineshape', ['step', 'rectangle'])
def test_value_step_rectangle(form, lineshape):
    """Test values at mu1/mu2 for step- and rectangle-functions."""
    func = getattr(lmfit.lineshapes, lineshape)
    # at position mu1 we should be at A/2
    assert_almost_equal(func(0), 0.5)
    # for a rectangular shape we have the same at mu2
    if lineshape == 'rectangle':
        assert_almost_equal(func(1), 0.5)


thermal_form = ['bose', 'maxwell', 'fermi', 'Bose-Einstein', 'unknown']


@pytest.mark.parametrize("form", thermal_form)
def test_form_argument_thermal_distribution(form):
    """Test 'form' argument for thermal_distribution function."""
    xvals = np.linspace(0, 10, 100)

    func = lmfit.lineshapes.thermal_distribution
    sig = inspect.signature(lmfit.lineshapes.thermal_distribution)

    fnc_args = [xvals]
    for par in [par_name for par_name in sig.parameters.keys()
                if par_name != 'x']:
        if par == 'form':
            fnc_args.append(form)
        else:
            fnc_args.append(sig.parameters[par].default)

    if form == 'unknown':
        msg = r"Invalid value .* for argument .*; should be one of .*"
        with pytest.raises(ValueError, match=msg):
            func(*fnc_args)
    else:
        fnc_output = func(*fnc_args)
        assert len(fnc_output) == len(xvals)
