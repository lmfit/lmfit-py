"""Tests for the print/report functions."""
import numpy as np
import pytest

import lmfit
from lmfit import (Minimizer, Parameters, ci_report, conf_interval, fit_report,
                   report_ci, report_fit)
from lmfit.lineshapes import gaussian
from lmfit.models import GaussianModel
from lmfit.printfuncs import (alphanumeric_sort, correl_table,
                              fitreport_html_table, getfloat_attr, gformat)

np.random.seed(0)


@pytest.fixture
def params():
    """Return a lmfit.Parameters class with initial values."""
    pars = Parameters()
    pars.add_many(('a1', 4), ('b', -20.0), ('c1', 3), ('a', 10.0), ('a2', 5),
                  ('b10', 6), ('d', None), ('b01', 8), ('e', 9), ('aa1', 10))
    return pars


@pytest.fixture
def fitresult():
    """Return a ModelResult after fitting a randomized Gaussian data set."""
    x = np.linspace(0, 12, 601)
    data = gaussian(x, amplitude=36.4, center=6.70, sigma=0.88)
    data = data + np.random.normal(x.size, scale=3.2)

    model = GaussianModel()
    params = model.make_params(amplitude=50, center=5, sigma=2)

    params['amplitude'].min = 1
    params['amplitude'].max = 100.0
    params['sigma'].min = 0
    params['sigma'].brute_step = 0.001

    result = model.fit(data, params, x=x)
    return result


@pytest.fixture
def confidence_interval():
    """Return the result of the confidence interval (ci) calculation."""
    def residual(pars, x, data=None):
        argu = (x*pars['decay'])**2
        shift = pars['shift']
        if abs(shift) > np.pi/2:
            shift = shift - np.sign(shift)*np.pi
        model = pars['amp']*np.sin(shift + x/pars['period']) * np.exp(-argu)
        if data is None:
            return model
        return model - data

    p_true = Parameters()
    p_true.add_many(('amp', 14.0), ('period', 5.33), ('shift', 0.123),
                    ('decay', 0.010))

    x = np.linspace(0.0, 250.0, 2500)
    data = residual(p_true, x) + np.random.normal(scale=0.7215, size=x.size)

    fit_params = Parameters()
    fit_params.add_many(('amp', 13.0), ('period', 2), ('shift', 0.0),
                        ('decay', 0.02))

    mini = Minimizer(residual, fit_params, fcn_args=(x,),
                     fcn_kws={'data': data})
    out = mini.leastsq()
    ci = conf_interval(mini, out)
    return ci


def test_alphanumeric_sort(params):
    """Test alphanumeric sort of the parameters."""
    sorted_params = sorted(params, key=alphanumeric_sort)
    expected = ['a', 'a1', 'a2', 'aa1', 'b', 'b01', 'b10', 'c1', 'd', 'e']
    assert sorted_params == expected


test_data_getfloat_attr = [('a', 'value', '10.0000000'),
                           ('b', 'value', '-20.0000000'),
                           ('c1', 'value', '3'),
                           ('d', 'value', '-inf'),
                           ('e', 'non_existent_attr', 'unknown'),
                           ('aa1', 'test', '(20+5j)')]


@pytest.mark.parametrize("par, attr, expected", test_data_getfloat_attr)
def test_getfloat_attr(params, par, attr, expected):
    """Test getfloat_attr function."""
    if par == 'aa1':
        # add an attribute that is not None, float, or int
        # This will never occur for Parameter values, but this function is
        # also used on the MinimizerResult/ModelResult where it could happen.
        params['aa1'].test = 20+5j

    output = getfloat_attr(params[par], attr)
    assert output == expected

    if par == 'a':
        assert len(output) == 10  # leading blank for pos values is stripped
    elif par == 'b':
        assert len(output) == 11
    elif par == 'c1':
        assert len(output) == 1


test_data_gformat = [(-1.25, '-1.25000000'), (1.25, ' 1.25000000'),
                     (-1234567890.1234567890, '-1.2346e+09'),
                     (1234567890.1234567890, ' 1.2346e+09'),
                     (12345.67890e150, ' 1.235e+154')]


@pytest.mark.parametrize("test_input, expected", test_data_gformat)
def test_gformat(test_input, expected):
    """Test gformat function."""
    output = gformat(test_input)
    assert output == expected


def test_reports_created(fitresult):
    """Verify that the fit reports are created and all headers are present."""
    report_headers = ['[[Model]]', '[[Fit Statistics]]', '[[Variables]]',
                      '[[Correlations]] (unreported correlations are < 0.100)']

    report = fitresult.fit_report()
    assert len(report) > 500
    for header in report_headers:
        assert header in report

    report1 = fit_report(fitresult)
    for header in report_headers[1:]:
        assert header in report1

    html_params = fitresult.params._repr_html_()
    assert len(html_params) > 500
    assert 'brute' in html_params
    assert 'standard error' in html_params
    assert 'relative error' in html_params

    html_report = fitresult._repr_html_()
    assert len(html_report) > 1000
    for header in ('Model', 'Fit Statistics', 'Parameters', 'Correlations'):
        assert header in html_report


def test_fitreports_init_values(fitresult):
    """Verify that initial values are displayed as expected."""
    fitresult.params['sigma'].init_value = None
    report_split = fitresult.fit_report().split('\n')
    indx = [i for i, val in enumerate(report_split) if 'sigma' in val][0]
    assert '(init = ?)' in report_split[indx]

    indx_center = [i for i, val in enumerate(report_split) if
                   'center:' in val][0]
    indx_amplitude = [i for i, val in enumerate(report_split) if
                      'amplitude:' in val][0]
    for indx, init_val in zip([indx_center, indx_amplitude], [5, 50]):
        assert f'(init = {init_val})' in report_split[indx]


def test_fitreports_min_correl(fitresult):
    """Verify that only correlation >= min_correl are displayed."""
    report = fitresult.fit_report(min_correl=0.6)
    assert '[[Correlation]]' not in report

    html_report = fitresult._repr_html_(min_correl=0.6)
    assert 'Correlation' not in html_report


def test_fitreports_show_corre(fitresult):
    """Verify that correlation are not shown when show_correl=False."""
    report = fitresult.fit_report(show_correl=False)
    assert '[[Correlation]]' not in report

    html_report = fitresult._repr_html_(show_correl=False)
    assert 'Correlation' not in html_report


def test_fitreports_sort_pars(fitresult):
    """Test sorting of parameters in the fit report."""
    # not sorted
    report_split = fitresult.fit_report(sort_pars=False).split('\n')
    indx_vars = report_split.index('[[Variables]]')
    first_par = list(fitresult.params.keys())[0]
    assert first_par in report_split[indx_vars+1]

    # sorted using default alphanumeric sort
    report_split = fitresult.fit_report(sort_pars=True).split('\n')
    indx_vars = report_split.index('[[Variables]]')
    assert 'amplitude' in report_split[indx_vars+1]

    # sorted using custom sorting algorithm: length of variable name
    def sort_length(s):
        return len(s)

    report_split = fitresult.fit_report(sort_pars=sort_length).split('\n')
    indx_vars = report_split.index('[[Variables]]')
    assert 'fwhm' in report_split[indx_vars+1]


def test_correl_table(fitresult, capsys):
    """Verify that ``correl_table`` is not empty."""
    table_lines = correl_table(fitresult.params).split('\n')
    nvarys = fitresult.nvarys

    assert len(table_lines) == nvarys+4
    assert len(table_lines[5]) > nvarys*10


def test_fit_report_correl_table(fitresult, capsys):
    """Verify that ``correl_table`` is not empty."""
    out = fitresult.fit_report(correl_mode='table')
    assert '[[Correlations]]' in out
    assert '----+' in out


def test_report_fit(fitresult, capsys):
    """Verify that the fit report is printed when using report_fit."""
    # report_fit with MinimizerResult/ModelResult as argument gives full
    # output of fitting results (except for [[Model]])
    report_fit(fitresult)
    report_headers = ['[[Fit Statistics]]', '[[Variables]]',
                      '[[Correlations]] (unreported correlations are < 0.100)']
    captured = capsys.readouterr()
    for header in report_headers:
        assert header in captured.out

    # report_fit with Parameter set as argument gives [[Variables]] and
    # [[Correlations]]
    report_fit(fitresult)
    report_headers = ['[[Variables]]',
                      '[[Correlations]] (unreported correlations are < 0.100)']
    captured = capsys.readouterr()
    for header in report_headers:
        assert header in captured.out


def test_report_leastsq_no_errorbars(fitresult):
    """Verify correct message when uncertainties could not be estimated."""
    # general warning is shown
    fitresult.errorbars = False
    report = fitresult.fit_report()
    assert 'Warning: uncertainties could not be estimated:' in report

    # parameter is at initial value
    fitresult.params['amplitude'].value = 50.0
    report = fitresult.fit_report()
    assert 'amplitude:  at initial value' in report

    # parameter is at boundary max/min
    fitresult.params['amplitude'].value = 100.0
    report = fitresult.fit_report()
    assert 'amplitude:  at boundary' in report

    fitresult.params['amplitude'].value = 1.0
    report = fitresult.fit_report()
    assert 'amplitude:  at boundary' in report


def test_report_no_errorbars_no_numdifftools(fitresult):
    """Verify message without numdifftools and not using leastsq/least_squares."""
    fitresult.fit(method='nelder')
    lmfit.printfuncs.HAS_NUMDIFFTOOLS = False
    fitresult.errorbars = False
    report = fitresult.fit_report()
    msg = 'this fitting method does not natively calculate uncertainties'
    assert msg in report
    assert 'numdifftools' in report


def test_report_no_errorbars_with_numdifftools_no_init_value(fitresult):
    """No TypeError for parameters without initial value when no errorbars.

    Verify that for parameters without an init_value the fit_report() function
    does not raise a TypeError when comparing if a parameter is at its initial
    value (if HAS_NUMDIFFTOOLS is True and result.errorbars is False).

    See GitHub Issue 578: https://github.com/lmfit/lmfit-py/issues/578

    """
    fitresult.fit(method='nelder')
    lmfit.printfuncs.HAS_NUMDIFFTOOLS = True
    fitresult.errorbars = False
    fitresult.params['amplitude'].init_value = None
    report = fitresult.fit_report()
    assert 'Warning: uncertainties could not be estimated:' in report


def test_report_fixed_parameter(fitresult):
    """Verify that a fixed parameter is shown correctly."""
    fitresult.params['center'].vary = False
    report_split = fitresult.fit_report().split('\n')
    indx = [i for i, val in enumerate(report_split) if 'center' in val][0]
    assert '(fixed)' in report_split[indx]


def test_report_expression_parameter(fitresult):
    """Verify that a parameter with expression is shown correctly."""
    report_split = fitresult.fit_report().split('\n')
    indices = [i for i, val in enumerate(report_split) if
               'fwhm' in val or 'height' in val]
    for indx in indices:
        assert '==' in report_split[indx]

    html_params = fitresult.params._repr_html_()
    assert 'expression' in html_params


def test_report_modelpars(fitresult):
    """Verify that model_values are shown when modelpars are given."""
    model = GaussianModel()
    params = model.make_params(amplitude=35, center=7, sigma=0.9)
    report_split = fitresult.fit_report(modelpars=params).split('\n')
    indices = [i for i, val in enumerate(report_split) if
               ('sigma:' in val or 'center:' in val or 'amplitude:' in val)]
    for indx in indices:
        assert 'model_value' in report_split[indx]


def test_report_parvalue_non_numeric(fitresult):
    """Verify that a non-numeric value is handled gracefully."""
    fitresult.params['center'].value = None
    fitresult.params['center'].stderr = None
    report = fitresult.fit_report()
    assert len(report) > 50


def test_report_zero_value_spercent(fitresult):
    """Verify that ZeroDivisionError in spercent calc. gives empty string."""
    fitresult.params['center'].value = 0
    fitresult.params['center'].stderr = 0.1
    report_split = fitresult.fit_report().split('\n')
    indx = [i for i, val in enumerate(report_split) if 'center:' in val][0]
    assert '%' not in report_split[indx]
    assert '%' in report_split[indx+1]

    html_params_split = fitresult.params._repr_html_().split('<tr>')
    indx = [i for i, val in enumerate(html_params_split) if 'center' in val][0]
    assert '%' not in html_params_split[indx]
    assert '%' in html_params_split[indx+1]


@pytest.mark.skipif(not lmfit.minimizer.HAS_EMCEE, reason="requires emcee v3")
def test_spercent_html_table():
    """Regression test for GitHub Issue #768."""
    np.random.seed(2021)
    x = np.random.uniform(size=100)
    y = x + 0.1 * np.random.uniform(size=x.size)

    def res(par, x, y):
        return y - par['k'] * x + par['b']

    params = lmfit.Parameters()
    params.add('b', 0, vary=False)
    params.add('k', 1)

    fitter = lmfit.Minimizer(res, params, fcn_args=(x, y))
    fit_res = fitter.minimize(method='emcee', steps=5)
    fitreport_html_table(fit_res)


def test_ci_report(confidence_interval):
    """Verify that the CI report is created when using ci_report."""
    report = ci_report(confidence_interval)
    assert len(report) > 250

    for par in confidence_interval.keys():
        assert par in report

    for interval in ['99.73', '95.45', '68.27', '_BEST_']:
        assert interval in report


def test_report_ci(confidence_interval, capsys):
    """Verify that the CI report is printed when using report_ci."""
    report_ci(confidence_interval)
    captured = capsys.readouterr()

    assert len(captured.out) > 250

    for par in confidence_interval.keys():
        assert par in captured.out

    for interval in ['99.73', '95.45', '68.27', '_BEST_']:
        assert interval in captured.out


def test_ci_report_with_offset(confidence_interval):
    """Verify output of CI report when using with_offset."""
    report_split = ci_report(confidence_interval,
                             with_offset=True).split('\n')  # default
    amp_values = [abs(float(val)) for val in report_split[1].split()[2:]]
    assert np.all(np.less(np.delete(amp_values, 3), 0.2))

    report_split = ci_report(confidence_interval,
                             with_offset=False).split('\n')
    amp_values = [float(val) for val in report_split[1].split()[2:]]
    assert np.all(np.greater(amp_values, 13))


@pytest.mark.parametrize("ndigits", [3, 5, 7])
def test_ci_report_with_ndigits(confidence_interval, ndigits):
    """Verify output of CI report when specifying ndigits."""
    report_split = ci_report(confidence_interval, ndigits=ndigits).split('\n')
    period_values = list(report_split[2].split()[2:])
    length = [len(val.split('.')[-1]) for val in period_values]
    assert np.all(np.equal(length, ndigits))
