"""
Examples of how to use global_fit.py, formulated as unit tests.

Tested on lmfit version 1.2.2 (python 3.10.10) and 1.3.3 (python 3.13.3).

To run all of them in an ipython console and see the diagrams, you can run
    import pytest
    pytest.main(['tests/test_global_fit.py'])
"""
from matplotlib import pyplot as plt
import numpy as np
import pytest

from lmfit import models
from lmfit.global_fit import (multi_constrain, multi_fit, multi_guess,
                              multi_make_params, repeat_model)

# Tests that need xarray are skipped if unavailable, by pytest.importorskip('xarray')


# ---- Five guassians like in example_fit_multi_datasets.html
@pytest.fixture
def five_gaussians():
    # Create five simulated Gaussian data sets
    np.random.seed(2021)
    x = np.linspace(-1, 2, 151)
    five_signals = []
    for _ in np.arange(5):
        amp = 0.60 + 9.50 * np.random.rand()
        cen = -0.20 + 1.20 * np.random.rand()
        sig = 0.25 + 0.03 * np.random.rand()
        # For equality with the web example, rescale amp by the normalization in lineshapes.gaussian
        amp *= (np.sqrt(2 * np.pi) * sig)
        dat = models.gaussian(x, amp, cen, sig) + np.random.normal(size=x.size, scale=0.1)
        five_signals.append(dat)
    return x, five_signals


def test_five_gaussians_list(five_gaussians):
    x, five_signals = five_gaussians
    print("=== List like https://lmfit.github.io/lmfit-py/examples/example_fit_multi_datasets.html")
    # In comparison with example_fit_multi_datasets.html, 11 lines of custom objective function
    # and suffix handling was shortened to 2 lines here. Since this implementation uses
    # models.GaussianModel and prefixes instead of suffixes, the parameter names are not exactly
    # as on the web page and the amplitude is defined differently.
    # For an even shorter variant, see test_five_gaussians_list_using_repeat_and_guess().
    prefixes = [f'p{i}_' for i in range(1, 6)]  # May not start with digits. 'p' as in 'peak'.
    five_models = [models.GaussianModel(prefix=p) for p in prefixes]
    # five_signals = np.array(five_signals)  # Would also work, but is not necessary

    # Create all the parameters
    fit_params = multi_make_params(five_models)
    # Set initial values and constraints (here not using any model-defaults, param_hints or guess())
    for i, m in enumerate(five_models):
        fit_params[m.prefix + 'amplitude'].set(value=0.5, min=0.0, max=200)
        fit_params[m.prefix + 'center'].set(value=0.4, min=-2.0, max=2.0)
        fit_params[m.prefix + 'sigma'].set(value=0.3, min=0.01, max=3.0)
        if i != 0:
            # Constrain sigma to be equal for all peaks, setting an `expr` to reuse the first sigma.
            fit_params[m.prefix + 'sigma'].set(expr=prefixes[0] + 'sigma')
    fit_params.pretty_print()

    # Run the global fit and show the fitting result, now using Model.fit instead of minimize()
    out = multi_fit(five_models, five_signals, fit_params, x=x)
    print(out.fit_report())

    plt.figure(1, clear=True)
    for label, model in enumerate(five_models):
        plotted = plt.plot(x, model.eval(out.params, x=x), label=label)
        plt.plot(x, five_signals[label], '.:', color=plotted[0].get_color())
    plt.legend()

    assert out.ndata == 5 * len(x)
    assert out.nvarys == 11  # Number of fitted variables
    assert out.rsquared >= 0.98
    # The ordering between labels and peaks is not well defined, could change by optimizer details.
    for p in out.params.values():
        if p.name.endswith('center') and p.value < 0:  # Found the leftmost peak
            assert out.params[p.name[:3] + 'amplitude'].value == pytest.approx(4.594, abs=0.02)
    assert out.params['p1_center'].value == pytest.approx(0.6805, abs=0.01)
    assert out.params['p3_center'].value == pytest.approx(-0.0826, abs=0.01)
    # The constant initial values are far from the fitted
    assert out.params['p3_center'].value != pytest.approx(fit_params['p3_center'].value, abs=0.02)
    assert 'p1_fwhm' in fit_params
    assert 'p1_fwhm' in out.params
    assert 'p1_fwhm' not in out.var_names  # Not a fitted parameter, computed from p2_sigma
    assert 'p1_fwhm' not in out.model._param_root_names
    assert 'p1_sigma' in out.var_names
    assert 'p1_sigma' in out.model._param_root_names
    assert 'p2_sigma' not in out.var_names  # Due to sharing

    # Check that the evaluation works in list-mode
    print(out.model)
    fitted_eval_all = out.eval()
    fitted_eval_1 = five_models[1].eval(out.params, x=x)
    assert np.array_equal(out.multi_eval()[1], fitted_eval_1)
    assert np.array_equal(fitted_eval_all[len(x):2 * len(x)], fitted_eval_1)
    assert fitted_eval_1[40] == pytest.approx(0.166, abs=0.001)
    assert fitted_eval_all[len(x) + 40] == fitted_eval_1[40]

    demo_params = fit_params.copy()
    assert demo_params[prefixes[4] + 'sigma'].value == 0.3
    assert out.params[prefixes[4] + 'sigma'].value == pytest.approx(0.2576, abs=0.001)
    # Other sigma instances are given by expressions and will follow this change in p1_sigma
    demo_params[prefixes[0] + 'sigma'].set(value=0.332)
    assert demo_params[prefixes[4] + 'sigma'].value == 0.332
    initial_eval_all = out.model.eval(demo_params, x=x)
    assert initial_eval_all[len(x) + 40] == pytest.approx(0.117, abs=0.001)
    assert np.array_equal(initial_eval_all[len(x):2 * len(x)], initial_eval_all[0:len(x)])
    # Make some parameter differ between the five instances
    demo_params[prefixes[1] + 'center'].set(value=0.42)
    initial_eval_0 = five_models[0].eval(demo_params, x=x)
    initial_eval_1 = five_models[1].eval(demo_params, x=x)
    initial_eval_all = out.model.eval(demo_params, x=x)
    assert np.array_equal(initial_eval_all[0:len(x)], initial_eval_0)
    assert np.array_equal(initial_eval_all[len(x):2 * len(x)], initial_eval_1)
    assert not np.array_equal(initial_eval_all[len(x):2 * len(x)], initial_eval_all[0:len(x)])


def test_five_gaussians_list_using_repeat_and_guess(five_gaussians):
    x, five_signals = five_gaussians
    print("=== List-example variant using multi_guess()")
    prefixes = [f'p{i}_' for i in range(1, 6)]  # May not start with digits. 'p' as in 'peak'.
    dict_of_models = repeat_model(models.GaussianModel, prefixes)
    assert list(dict_of_models.keys()) == [f'p{i}' for i in range(1, 6)]
    # The prefixes can be generated automatically
    dict_of_models = repeat_model(models.GaussianModel, range(1, 6), numeric_prefix='p')
    assert list(dict_of_models.keys()) == [f'p{i}' for i in range(1, 6)]
    five_models = list(dict_of_models.values())  # To work in list mode below
    assert five_models[2].prefix == prefixes[2]

    # Customize constraints (will be respected by multi_guess() which chooses initial values)
    hints = {}
    for i, m in enumerate(five_models):
        hints[m.prefix + 'amplitude'] = {'min': 0.0, 'max': 200}
        hints[m.prefix + 'center'] = {'min': -2.0, 'max': 2.0}
        hints[m.prefix + 'sigma'] = {'min': 0.01, 'max': 3.0}
        if i != 0:
            # Constrain sigma to be equal for all peaks, setting an `expr` to reuse the first sigma.
            hints[m.prefix + 'sigma']['expr'] = prefixes[0] + 'sigma'
    fit_params = multi_guess(five_models, five_signals, param_hints=hints, x=x)
    print("\nInitial guess:")
    fit_params.pretty_print()

    # Run the global fit and show the fitting result, now using Model.fit instead of minimize()
    out = multi_fit(five_models, five_signals, fit_params, x=x)
    print(out.fit_report())

    plt.figure(2, clear=True)
    for label, model in enumerate(five_models):
        plotted = plt.plot(x, model.eval(out.params, x=x), label=label)
        plt.plot(x, five_signals[label], '.:', color=plotted[0].get_color())
    plt.legend()

    assert out.ndata == 5 * len(x)
    assert out.nvarys == 11  # Number of fitted variables
    assert out.rsquared >= 0.98
    # The ordering between labels and peaks is not well defined, could change by optimizer details.
    for p in out.params.values():
        if p.name.endswith('center') and p.value < 0:  # Found the leftmost peak
            assert out.params[p.name[:3] + 'amplitude'].value == pytest.approx(4.594, abs=0.02)
    assert out.params['p1_center'].value == pytest.approx(0.6805, abs=0.01)
    assert out.params['p3_center'].value == pytest.approx(-0.0826, abs=0.01)
    # Since the initival values were guessed from the data, they should be close but not exact
    assert out.params['p3_center'].value == pytest.approx(fit_params['p3_center'].value, abs=0.02)
    assert out.params['p3_center'].value != pytest.approx(fit_params['p3_center'].value, abs=0.001)


def test_five_gaussians_dict(five_gaussians):
    x, five_signals = five_gaussians
    print("=== Dict like https://lmfit.github.io/lmfit-py/examples/example_fit_multi_datasets.html")
    labels = [f'p{i}_' for i in range(1, 6)]  # May not start with digits. 'p' as in 'peak'.
    dict_of_models = {label: models.GaussianModel(prefix=label) for label in labels}
    dict_of_signals = dict(zip(labels, five_signals))

    # Create all the parameters
    fit_params = multi_make_params(dict_of_models)
    # Set initial values and constraints (here not using any model-defaults, param_hints or guess())
    for label, m in dict_of_models.items():
        fit_params[m.prefix + 'amplitude'].set(value=0.5, min=0.0, max=200)
        fit_params[m.prefix + 'center'].set(value=0.4, min=-2.0, max=2.0)
        fit_params[m.prefix + 'sigma'].set(value=0.3, min=0.01, max=3.0)
        if m.prefix != labels[0]:
            # Constrain sigma to be equal for all peaks, setting an `expr` to reuse the first sigma.
            fit_params[m.prefix + 'sigma'].set(expr=labels[0] + 'sigma')
    fit_params.pretty_print()

    # Run the global fit and show the fitting result, now using Model.fit instead of minimize()
    out = multi_fit(dict_of_models, dict_of_signals, fit_params, x=x)
    print(out.fit_report())

    plt.figure(3, clear=True)
    for label, model in dict_of_models.items():
        plotted = plt.plot(x, model.eval(out.params, x=x), label=label)
        plt.plot(x, dict_of_signals[label], '.:', color=plotted[0].get_color())
    plt.legend()

    assert out.ndata == 5 * len(x)
    assert out.nvarys == 11  # Number of fitted variables
    assert out.rsquared >= 0.98
    # The ordering between labels and peaks is not well defined, could change by optimizer details.
    for p in out.params.values():
        if p.name.endswith('center') and p.value < 0:  # Found the leftmost peak
            assert out.params[p.name[:3] + 'amplitude'].value == pytest.approx(4.594, abs=0.02)
    assert out.params['p1_center'].value == pytest.approx(0.6805, abs=0.01)
    assert out.params['p3_center'].value == pytest.approx(-0.0826, abs=0.01)
    # The constant initial values are far from the fitted
    assert out.params['p3_center'].value != pytest.approx(fit_params['p3_center'].value, abs=0.02)

    # Check that the evaluation works in dict-mode
    print(out.model)
    fitted_eval_1 = dict_of_models[labels[1]].eval(out.params, x=x)
    assert np.array_equal(out.multi_eval()[labels[1]], fitted_eval_1)
    assert np.array_equal(out.eval()[len(x):2 * len(x)], fitted_eval_1)

    demo_params = fit_params.copy()
    # Other sigma instances are given by expressions and will follow this change in p1_sigma
    demo_params[labels[0] + 'sigma'].set(value=0.332)
    initial_eval_all = out.model.eval(demo_params, x=x)
    assert np.array_equal(initial_eval_all[len(x):2 * len(x)], initial_eval_all[0:len(x)])
    # Make some parameter differ between the five instances
    demo_params[labels[1] + 'center'].set(value=0.42)
    initial_eval_0 = dict_of_models[labels[0]].eval(demo_params, x=x)
    initial_eval_1 = dict_of_models[labels[1]].eval(demo_params, x=x)
    initial_eval_all = out.model.eval(demo_params, x=x)
    assert np.array_equal(initial_eval_all[0:len(x)], initial_eval_0)
    assert np.array_equal(initial_eval_all[len(x):2 * len(x)], initial_eval_1)
    assert not np.array_equal(initial_eval_all[len(x):2 * len(x)], initial_eval_all[0:len(x)])


def test_two_unequal_indpendent_vars():
    print("=== Dict-example with unequal independent variables for the signals")
    np.random.seed(2025)
    x_short = np.arange(-1, 1.5, 0.3)
    x_long = np.arange(-2, 2, 0.1)
    assert len(x_long) != len(x_short)
    irregular_data = {
        'short': models.gaussian(x_short, 4, 0.4, 0.3) + np.random.randn(*x_short.shape),
        'long': models.gaussian(x_long, 4, 0.5, 0.4) + np.random.randn(*x_long.shape)}
    mm = {label: models.GaussianModel(prefix=label[0] + '_') for label in irregular_data.keys()}

    # When the two submodels use the same independent variable name 'x' but need
    # different arrays for their 'x', they need to be passed in a separate_kwargs-dict.
    x_dict = {'short': {'x': x_short}, 'long': {'x': x_long}}
    # mm.independent_vars = []  # Don't require any common 'x'
    # print(repr(mm))

    initial_mm = multi_make_params(mm)
    # initial_mm = multi_make_params(mm, s_center={'value': 2, 'max': 342})  # Common assignments
    # initial_mm = multi_make_params(mm, l_center={'expr': 's_center'})  # Impractical when many

    # Constrain their center-parameters to be equal using utility-function multi_constrain():
    initial_mm = multi_constrain(mm, initial_mm, {'center': True})
    # initial_mm = multi_constrain(mm, initial_mm, {'center': {'short', 'long'}})  # Another way
    # initial_mm = multi_constrain(mm, initial_mm, {'center': {'long': 'short'}})  # Another way
    # initial_mm = multi_constrain(list(mm.values()), initial_mm, {'center': [0, 0]})  # Another way
    # initial_mm.pretty_print()

    # For the guessing to respect the grouping constraint ('center') requires param_hints-argument:
    # Approach 1: Provide Params with already specified grouping expressions
    initial_mm = multi_guess(mm, irregular_data, separate_kwargs=x_dict, param_hints=initial_mm)
    # Approach 2: Specify the grouping expressions as hints here without using multi_constrain()
    # initial_mm = multi_guess(mm, irregular_data, separate_kwargs=x_dict,
    #                          param_hints={'l_center': {'expr': 's_center'}})
    # Approach 3 (bad): While multi_constrain() can be called after multi_guess(), all but one of
    # the guesses would currently be thrown away by multi_constrain() (not taking the mean of them).

    print("\nInitial guess:")
    initial_mm.pretty_print()

    print("\nFitted:")
    result_mm = multi_fit(mm, irregular_data, initial_mm, separate_kwargs=x_dict)
    # result_mm.params.pretty_print()
    print(result_mm.fit_report())

    figure, axes = plt.subplots(len(mm), 1, num=4, clear=True)
    plt.suptitle(f"Global fit of {len(mm)} unequally sized datasets with {result_mm.ndata} points "
                 f"in total.\nFitting {', '.join(result_mm.var_names)}.")
    for (label, model), ax in zip(mm.items(), axes):
        current_x = x_dict[label]['x']
        ax.set_title(f'"{label}" dataset with {len(current_x)} points')
        ax.plot(current_x, irregular_data[label], '.:k', label='Data')
        ax.plot(current_x, mm[label].eval(initial_mm, **x_dict[label]), '--y', label='Initial')
        ax.plot(current_x, model.eval(result_mm.params, **x_dict[label]), '-r', label='Fitted')
        del current_x
        ax.set_xlim(x_dict['long']['x'][[0, -1]] + [-0.1, 0.1])  # Use the long range for both
        ax.legend(loc='upper left')
        ax.set_xlabel('x')
        ax.set_ylabel('Signal')
    plt.tight_layout()

    assert result_mm.params['l_center'].value == result_mm.params['s_center'].value
    assert result_mm.params['s_fwhm'].value == pytest.approx(0.6183, rel=0.001)
    assert result_mm.params['l_fwhm'].value == pytest.approx(1.109, rel=0.001)
    assert result_mm.rsquared == pytest.approx(0.79, abs=0.01)

    # Demonstrate that you can re-evaluate the result with other independent variable values
    fitted_1D = result_mm.eval()
    other_1D = result_mm.eval(separate_kwargs={'short': {'x': x_short + 1},
                                               'long': {'x': x_long[1:] - 1}})
    assert len(other_1D) == len(fitted_1D) - 1  # x_long was shortened by one point
    assert not np.array_equal(fitted_1D, other_1D)

    # Demonstrate separate evaluation:
    fitted_dict = result_mm.multi_eval()
    assert np.array_equal(fitted_dict['short'], fitted_1D[:len(x_short)])
    assert np.array_equal(fitted_dict['long'], fitted_1D[len(x_short):])
    other_dict = result_mm.multi_eval(separate_kwargs={'short': {'x': x_short + 1},
                                                       'long': {'x': x_long[1:] - 1}})
    assert np.array_equal(other_dict['short'], other_1D[:len(x_short)])
    assert np.array_equal(other_dict['long'], other_1D[len(x_short):])
    fitted_short = mm['short'].eval(result_mm.params, x=x_short + 1)
    assert np.array_equal(other_dict['short'], fitted_short)


# ---- Four decaying step functions
@pytest.fixture
def four_decaying_signals():
    t = np.arange(10)  # Time axis
    # Start with noise for four time-dependent signals, with 10 points each.
    np.random.seed(2025)
    signal_0 = np.random.randn(len(t))
    signal_1 = np.random.randn(len(t))
    signal_2 = np.random.randn(len(t))
    signal_3 = np.random.randn(len(t))
    # Create some decaying peaks, for more relevat fits with ExponentialGaussianModel
    signal_0[2:8] += [9, 16, 14, 8, 5, 2]  # Short-lived (large gamma), blurry peak for mass 1
    signal_1[2:6] += [2, 15, 9, 5]  # Short-lived, sharp peak (small sigma) for mass 17
    signal_2[2:] += [2, 9, 8, 6, 5, 4, 3, 3]  # Long-lived, sharp start (small sigma) for mass 18
    signal_3[2:4] += [2, 4]  # Long-lived and blurrier start (large sigma) for mass 28
    signal_3[4:] += 6
    data = np.stack([signal_0, signal_1, signal_2, signal_3])
    return t, data


def test_xarray_decay_dict(four_decaying_signals):
    xr = pytest.importorskip('xarray', reason="Extra functionality with xarary.DataArray.coords")
    print("=== Example using repeat_model() and xarray.DataArray in dict mode")
    x, data = four_decaying_signals
    # 'x' means 'time' but models.ExponentialGaussianModel doesn't allow other independent_vars.
    example = xr.DataArray(data, dims=('selected_mass', 'time'),
                           coords={'selected_mass': [1, 17, 18, 28], 'x': ('time', x)})
    example.coords['selected_mass'].attrs['units'] = 'u'
    # To fit the time-dependence ('x'-dependence) four signals will be labelled by 'selected_mass'.
    # The independent variable 'x' will be taken automatically from example.coords['x'].

    # Define one kind of model to be repeated:
    base_model = models.ExponentialGaussianModel  # Class without keyword arguments
    # base_model = 'k*x + m'  # String for ExpressionModel
    # base_model = lambda x, k, m: k*x + m  # Function for Model

    # Generate the labels, prefixes and repeated models:
    repeated = repeat_model(base_model, example.coords['selected_mass'].values, numeric_prefix='m')
    # Convert the 2D-data to a matching dict
    dict_of_signals = dict(zip(repeated, example))
    # Alternatively, we could first produce a dict of signals with custom labels and use them:
    # dict_of_signals = dict(zip((f'm{m:.0f}' for m in example.selected_mass), example))
    # repeat_model(base_model, dict_of_signals)

    initial = multi_constrain(repeated, multi_make_params(repeated), {
        'center': True,  # Share 'center' by all
        'sigma': {'m17', 'm18'}  # Share 'sigma' between two labels in dict-mode
        # 'sigma': {'m18': 'm17', 'm28': 'm1'}  # Two groups of 'sigma' in dict-mode
    })
    initial = multi_guess(repeated, dict_of_signals, param_hints=initial)
    print("Initial guess:")
    initial.pretty_print()
    assert initial['m18_sigma'].expr == 'm17_sigma'
    assert initial['m18_center'].expr == 'm1_center'
    assert initial['m18_center'].value == initial['m17_center'].value
    assert initial['m18_center'].value == pytest.approx(4.375, abs=0.001)
    assert initial['m1_sigma'].value == pytest.approx(1.0, abs=0.01)
    assert initial['m18_sigma'].value == pytest.approx(1.5, abs=0.01)
    assert initial['m18_amplitude'].value == pytest.approx(44.58, abs=0.1)
    assert initial['m28_gamma'].max == pytest.approx(20, abs=1)

    result = multi_fit(repeated, dict_of_signals, initial)  # x will be found automatically
    print("Result evaluated as dict of arrays:")
    print(str(result.multi_eval()).replace('), ', '),\n '))
    print(result.fit_report())

    plt.figure(5, clear=True)
    for label, curve in result.multi_eval().items():
        m = dict_of_signals[label].selected_mass
        plotted = plt.plot(x, curve, '-',
                           label=f'{m.values} {m.units}: "{label}": {result.models[label]}')
        plt.plot(x, dict_of_signals[label], '.:', color=plotted[0].get_color())
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout()

    assert result.ndata == 40
    assert result.nvarys == 12
    assert result.model.underlying_independent_vars == ['x']
    assert result.model.underlying_models['m18'].prefix == 'm18_'
    assert result.models is result.model.underlying_models
    assert np.array_equal(result.multi_data['m17'].values, example.sel(selected_mass=17).values)
    assert not np.array_equal(result.multi_data['m17'].values, example.sel(selected_mass=1).values)
    assert result.multi_weights is None
    assert list(result.multi_eval().keys()) == ['m1', 'm17', 'm18', 'm28']
    assert np.array_equal(result.multi_eval()['m18'],
                          repeated['m18'].eval(result.params, x=example['x'].values))
    assert result.params['m18_sigma'].expr == 'm17_sigma'
    assert result.params['m17_sigma'].value == result.params['m18_sigma'].value  # Should be shared
    assert result.params['m17_sigma'].value != result.params['m1_sigma'].value  # Not shared
    assert result.params['m18_center'].value == result.params['m17_center'].value
    assert result.params['m18_center'].value == pytest.approx(2.605, abs=0.001)
    assert result.params['m17_center'].value == result.params['m1_center'].value  # Should be shared
    assert result.params['m1_sigma'].value == pytest.approx(0.981, abs=0.01)
    assert result.params['m18_sigma'].value == pytest.approx(0.395, abs=0.01)
    assert result.params['m18_amplitude'].value == pytest.approx(51.26, abs=0.1)
    assert result.params['m28_gamma'].value == pytest.approx(0.0056, abs=0.0015)  # 0.0048 or 0.0064
    assert result.rsquared == pytest.approx(0.973, abs=0.01)
    assert result.nfev > 2000   # Got 2262 and 26000 with different package versions

    # Not certain (due to noise), but the example signals are intended like this:
    assert result.params['m17_sigma'].value < (
        result.params['m1_sigma'].value + result.params['m28_sigma'].value) / 2
    assert (result.params['m18_gamma'].value + result.params['m28_gamma'].value) / 2 < (
        result.params['m1_gamma'].value + result.params['m17_gamma'].value) / 2


def test_xarray_decay_list(four_decaying_signals):
    xr = pytest.importorskip('xarray', reason="Extra functionality with xarary.DataArray.coords")
    print("=== Example using repeat_model() and xarray.DataArray in list mode")
    x, data = four_decaying_signals
    # 'x' means 'time' but models.ExponentialGaussianModel doesn't allow other independent_vars.
    example = xr.DataArray(data, dims=('selected_mass', 'time'),
                           coords={'selected_mass': [1, 17, 18, 28], 'x': ('time', x)})
    example.coords['selected_mass'].attrs['units'] = 'u'
    # To fit the time-dependence ('x'-dependence) four signals will be labelled by 'selected_mass'.
    # The independent variable 'x' will be taken automatically from example.coords['x'].

    # Generate the labels, prefixes and repeated models:
    repeated = repeat_model(models.ExponentialGaussianModel,
                            example.coords['selected_mass'].values,
                            numeric_prefix='m',
                            nan_policy='propagate')  # Argument to ExponentialGaussianModel()
    # To work in list mode (since example is a 2D-array) we can convert repeated from dict to list.
    model_list = list(repeated.values())

    initial = multi_constrain(model_list, multi_make_params(repeated), {
        'center': True,  # Share 'center' by all
        'sigma': {1, 2}  # Share 'sigma' between the middle two labels in list-mode
        # 'sigma': [0, 1, 1, 3]  # Share 'sigma' between the middle two labels in list-mode
        # 'sigma': [0, 1, 1, 0]  # Two groups of 'sigma' in list-mode
    })
    initial = multi_guess(model_list, example, param_hints=initial)
    print("Initial guess:")
    initial.pretty_print()

    result = multi_fit(model_list, example, initial)  # x will be found automatically
    print("Result evaluated as 2D-array:")
    print(result.multi_eval())
    print(result.fit_report())

    plt.figure(6, clear=True)
    for m, (index, curve) in zip(example.selected_mass, enumerate(result.multi_eval())):
        plotted = plt.plot(x, curve, '-',
                           label=f'{m.values} {m.units}: [{index}]: {result.models[index]}')
        plt.plot(x, example.sel(selected_mass=m), '.:', color=plotted[0].get_color())
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout()

    assert result.multi_eval().shape == example.shape
    assert result.params['m1_sigma'].value == pytest.approx(0.9807, abs=0.01)
    assert result.params['m18_sigma'].value == pytest.approx(0.395, abs=0.01)
    assert result.params['m18_amplitude'].value == pytest.approx(51.26, abs=0.1)
    assert result.params['m1_center'].value == pytest.approx(2.605, abs=0.01)
    assert result.params['m28_gamma'].value == pytest.approx(0.0056, abs=0.0015)  # 0.0048 or 0.0064
    assert result.rsquared == pytest.approx(0.973, abs=0.01)
    assert result.nfev > 2000   # Got 2262 and 26000 with different package versions
    assert result.params['m17_center'].value == result.params['m1_center'].value  # Should be shared
    assert result.params['m17_sigma'].value == result.params['m18_sigma'].value  # Should be shared
    assert result.params['m17_sigma'].value != result.params['m1_sigma'].value  # Not shared
    # Not certain (due to noise), but the example signals are intended like this:
    assert result.params['m17_sigma'].value < (
        result.params['m1_sigma'].value + result.params['m28_sigma'].value) / 2
    assert (result.params['m18_gamma'].value + result.params['m28_gamma'].value) / 2 < (
        result.params['m1_gamma'].value + result.params['m17_gamma'].value) / 2


def test_xarray_deacy_dict_indep_name_t(four_decaying_signals):
    xr = pytest.importorskip('xarray', reason="Extra functionality with xarary.DataArray.coords")
    print("=== Example using repeat_model() and xarray.DataArray with other indepdnent than x")
    t, data = four_decaying_signals
    example = xr.DataArray(data, dims=('selected_mass', 'time'),
                           coords={'selected_mass': [1, 17, 18, 28], 't': ('time', t)})
    example.coords['selected_mass'].attrs['units'] = 'u'
    # To fit the time-dependence ('t'-dependence) four signals will be labelled by 'selected_mass'.
    # The independent variable 't' will be taken automatically from example.coords['t'].

    # Use string expression approximating ExponentialGaussianModel but with 't' as independent.
    expgauss_t = """height * np.convolve(
                    (t >= center) * np.exp(-gamma * np.maximum(0, t - center)),  # Decaying step
                    np.exp(-(t - t.mean())**2 / (2 * sigma**2)),  # Centered Gaussian
                    'same')"""
    # Generate the labels, prefixes and repeated models
    repeated = repeat_model(expgauss_t, example.coords['selected_mass'].values, numeric_prefix='m',
                            independent_vars=['t'])  # Extra keyword argument to ExpressionModel()
    assert repeated['m1'].independent_vars == ['t']

    # With a Python function, independent_vars is automatically set to the first argument
    def expgauss_t(t, height, center, sigma, gamma):
        return height * np.convolve(
            (t >= center) * np.exp(-gamma * np.maximum(0, t - center)),  # Decaying step
            np.exp(-(t - t.mean())**2 / (2 * sigma**2)),  # Centered Gaussian
            'same')
    # Generate the labels, prefixes and repeated models
    repeated = repeat_model(expgauss_t, example.coords['selected_mass'].values, numeric_prefix='m')
    assert repeated['m1'].independent_vars == ['t']
    # To work in list mode (since example is a 2D-array) we can convert repeated from dict to list.
    model_list = list(repeated.values())

    # Some initial values must be set since the model does not implement guess().
    # Probably because np.maximum(0, t - center) gives jumps at integer center-values, the result
    # is sensitive to the initials. These give a good fit:
    initial = multi_make_params(
        repeated,
        height=1,
        center={'value': 1.5, 'min': t.min() + 1, 'max': t.max() - 1},
        sigma={'value': 0.5, 'min': 0.01, 'max': 5},  # Sigma should be positive
        gamma={'value': 0.3, 'min': 0, 'max': 5})  # Gamma should be non-negative
    initial = multi_constrain(model_list, initial, {
        'center': True,  # Share 'center' by all
        'sigma': [0, 1, 1, 3]  # Share 'sigma' between the middle two labels in list-mode
        # 'sigma': {1, 2}  # Share 'sigma' between the middle two labels in list-mode
    })
    # A plain ExpressionModel or Model by function does not implement guess()
    no_guesses = multi_guess(model_list, example, param_hints=initial)
    assert no_guesses == initial
    print("Initial guess:")
    initial.pretty_print()

    result = multi_fit(model_list, example, initial)  # t will be found automatically
    print(result.fit_report())
    plt.figure(7, clear=True)
    for m, (index, curve) in zip(example.selected_mass, enumerate(result.multi_eval())):
        plotted = plt.plot(t, curve, '-',
                           label=f'{m.values} {m.units}: [{index}]: {result.models[index]}')
        plt.plot(t, example.sel(selected_mass=m), '.:', color=plotted[0].get_color())
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout()

    assert result.rsquared == pytest.approx(0.918, abs=0.01)
    assert result.params['m1_sigma'].value == pytest.approx(1.34, abs=0.02)
    assert result.params['m18_sigma'].value == pytest.approx(0.786, abs=0.02)
    assert result.params['m17_sigma'].value == result.params['m18_sigma'].value  # Should be shared
    assert result.params['m18_height'].value == pytest.approx(7.24, abs=0.1)
    assert result.params['m28_gamma'].value == pytest.approx(0.0126, abs=0.0005)
    assert result.params['m1_center'].value == pytest.approx(2.67, abs=0.02)
    assert result.nfev == pytest.approx(531, abs=30)  # Got 537 and 524 in different versions


# ---- Manually managing models and parameter-sharing

def test_manually_managed_sharing(four_decaying_signals):
    print("=== Example of manually managing models and parameter-sharing")
    # With one signal array (handled by one model) per row
    x, data = four_decaying_signals
    x = x - 0.75 * x.mean()  # To make the quadratic model able to get a peak near the signal peak

    # NOTE: The few-parameter models used here are not really able to fit the example data,
    # the purpose is just to show that model types can be mixed and instances duplicated.
    model_0 = models.LinearModel()
    model_1 = models.LinearModel(prefix='other_')
    model_2 = models.ExpressionModel('q * x**2 + 2 * intercept')  # sharing 'intercept' with model_0
    model_3 = model_0  # Reusing the exact same model and parameters (so it will have to compromise)
    four_models = [model_0, model_1, model_2, model_3]

    # Since ExpressionModel doesn't implement guess(), parameter 'q' of model_2 needs an initial
    # value. One approach could be to persistently set model_2.set_param_hint('q', value=7)
    # before guessing. Here we instead pass q as keyword argument and let guesses override:
    initial = multi_make_params(four_models, q=7) + multi_guess(four_models, data, x=x)
    # Equivalent alternative:
    # initial = multi_guess(four_models, data, x=x, param_hints={'q': 7})
    print("Initial guess:")
    initial.pretty_print()
    assert initial['intercept'].value == pytest.approx(4.522, rel=0.01)
    assert initial['slope'].value == pytest.approx(0.258, rel=0.01)
    assert initial['q'].value == 7

    np.random.seed(2025)
    result = multi_fit(four_models, data, initial, x=x,
                       weights=1 / np.random.randint(3, 5, data.shape))
    print("Initial guess evaluated as 1D-array:")
    print(result.model.eval(initial, x=x))
    print("Initial guess evaluated as 2D-array:")
    print(result.multi_eval(initial, x=x))
    print(result.fit_report())
    print("\nFitted:")
    result.params.pretty_print()
    # Some ways of returning the evaluated result
    print("Result evaluated as 1D-array:")
    print(result.eval())
    print("Result evaluated as 2D-array:")
    print(result.multi_eval())

    plt.figure(8, clear=True)
    for label, curve in enumerate(result.multi_eval()):
        # The markers are customized to see that the first and last model give overlapping curves.
        plotted = plt.plot(x, curve, '-' + '+.-.'[label],
                           label=f'Index {label}: {four_models[label]}')
        plt.plot(x, data[label], '.:', color=plotted[0].get_color())
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout()

    assert result.params['intercept'].value == pytest.approx(3.671, rel=0.002)
    assert result.params['other_intercept'].value == pytest.approx(3.321, rel=0.002)
    assert result.params['other_slope'].value == pytest.approx(-0.1187, rel=0.002)
    assert result.params['q'].value == pytest.approx(-0.2026, rel=0.002)
    assert result.params['slope'].value == pytest.approx(0.2617, rel=0.002)
    assert result.nfev == 13
    assert ("Model(Global fit of 4 models of types ExpressionModel, LinearModel)"
            in str(result.model))
    # lmfit 1.2.2 gave the same fit but results.rsquared = 0.898, which was visually unreasonable
    # and probably means that the weights were considered differently then.
    assert (result.rsquared == pytest.approx(0.898, abs=0.002)  # Surprisingly high in lmfit 1.2.2
            or result.rsquared == pytest.approx(0.028, abs=0.002))  # Reasonable for the (bad) fit.
