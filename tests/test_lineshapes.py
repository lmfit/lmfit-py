"""Tests for lineshape functions."""

import inspect

import numpy as np
import pytest

import lmfit

lineshapes_functions = [fnc_name for fnc_name in lmfit.lineshapes.functions if
                        fnc_name not in ('gamma', 'gammaln', 'wofz', 'erf',
                                         'erfc')]


@pytest.mark.parametrize("lineshape", lineshapes_functions)
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


@pytest.mark.parametrize("lineshape", lineshapes_functions)
def test_x_float_value(lineshape):
    """Test lineshapes when x is not an array but a float."""
    xval = 7.0

    func = getattr(lmfit.lineshapes, lineshape)
    sig = inspect.signature(func)

    fnc_args = [xval]

    for par in [par_name for par_name in sig.parameters.keys()
                if par_name != 'x']:
        fnc_args.append(sig.parameters[par].default)

    if lineshape in ('step', 'rectangle'):
        msg = r"'float' object does not support item assignment"
        with pytest.raises(TypeError, match=msg):
            fnc_output = func(*fnc_args)
    else:
        fnc_output = func(*fnc_args)
        assert isinstance(fnc_output, float)
