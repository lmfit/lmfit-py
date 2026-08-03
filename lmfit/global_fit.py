"""
Tools to facilitate "global fitting" of separate signals by models that may share parameters.

As a quick introduction, the following three examples demonstrate slightly different workflows.
Extended versions can be found in tests/test_global_fit.py

Example using lists and only basic functionality
------------------------------------------------
from global_fit import multi_make_params, multi_fit
from lmfit import models
import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-0.8, 2.2, 81)
signals = [models.gaussian(x, 3 + i, i / 3, 0.2) + np.random.normal(size=x.size) for i in range(5)]
five_models = [models.GaussianModel(prefix=f'p{i + 1}_') for i in range(5)]
params = multi_make_params(five_models)

# Customize all initial values and constraints
for i, m in enumerate(five_models):
    params[m.prefix + 'amplitude'].set(value=1, min=0.0, max=100)
    params[m.prefix + 'center'].set(value=0.2 * i, min=-2.0, max=2.0)  # i-dependence for ordering
    params[m.prefix + 'sigma'].set(value=0.3, min=0.01, max=3.0)
    if i != 0:
        # Constrain sigma to be shared, setting an `expr` to reuse the first sigma.
        params[m.prefix + 'sigma'].set(expr=five_models[0].prefix + 'sigma')

result = multi_fit(five_models, signals, params, x=x)
print(result.fit_report())
plt.figure(clear=True)
for label, model in enumerate(five_models):
    plotted = plt.plot(x, model.eval(result.params, x=x), label=model.prefix)
    plt.plot(x, signals[label], '.:', color=plotted[0].get_color())
plt.legend()

Example with unequal independent variables
------------------------------------------
from global_fit import multi_make_params, multi_constrain, multi_guess, multi_fit
from lmfit import models
import numpy as np
from matplotlib import pyplot as plt

x_short = np.arange(-1, 1.5, 0.3)
x_long = np.arange(-2, 2, 0.1)
irregular_data = {'short': models.gaussian(x_short, 4, 0.4, 0.3) + np.random.randn(*x_short.shape),
                  'long': models.gaussian(x_long, 4, 0.5, 0.4) + np.random.randn(*x_long.shape)}
# A dict is needed to pass different 'x'-arrays to the two models
x_dict = {'short': {'x': x_short}, 'long': {'x': x_long}}

# Create labels, models and parameters
models = {label: models.GaussianModel(prefix=label[0] + '_') for label in irregular_data.keys()}
initial = multi_make_params(models)

# Optionally constrain the center-parameters to be equal
initial = multi_constrain(models, initial, {'center': True})
# Optionally guess (with averaging of the separate guesses for the shared 'center')
initial = multi_guess(models, irregular_data, separate_kwargs=x_dict, param_hints=initial)
# More fine-grained parameter sharing can be configured by other values to multi_constrain()
# or by providing param_hints with customized 'expr' for dependent parameters.

result = multi_fit(models, irregular_data, initial, separate_kwargs=x_dict)
print(result.fit_report())

figure, axes = plt.subplots(len(models), 1, clear=True)
for (label, model), ax in zip(models.items(), axes):
    current_x = x_dict[label]['x']
    ax.plot(current_x, irregular_data[label], '.k', label='Data')
    ax.plot(current_x, model.eval(initial, **x_dict[label]), '--y', label='Initial')
    ax.plot(current_x, model.eval(result.params, **x_dict[label]), '-r', label='Fitted')
    ax.set_xlim(x_dict['long']['x'][[0, -1]] + [-0.1, 0.1])  # Use the long range for both panels
    ax.set_xlabel('x')
    ax.set_ylabel(f'"{label}" signal')
    ax.legend(loc='upper left')

Example with repeat_model() and xarray.DataArray
------------------------------------------------
import xarray as xr
import numpy as np
from global_fit import repeat_model, multi_make_params, multi_constrain, multi_guess, multi_fit
from lmfit import models
from matplotlib import pyplot as plt

x = np.arange(10)  # Time axis
# To fit the time-dependence ('x'-dependence) four signals will be labelled by 'selected_mass'.
example = xr.DataArray(
    np.random.randn(4, len(x)),
    dims=('selected_mass', 'time'),
    # 'x' means 'time' here, but models.ExponentialGaussianModel doesn't allow other names.
    coords={'selected_mass': [1, 17, 18, 28], 'x': ('time', x)})
example.coords['selected_mass'].attrs['units'] = 'u'
# The independent variable 'x' will be taken automatically from example.coords['x'].
# Create some decaying peaks, for more relevat fits with ExponentialGaussianModel
example[0, 2:8] += [9, 16, 14, 8, 5, 2]  # Short-lived, blurry peak for mass 1
example[1, 2:6] += [2, 15, 9, 5]  # Short-lived (large gamma), sharp peak (small sigma) for mass 17
example[2, 2:] += [2, 9, 8, 6, 5, 4, 3, 3]  # Long-lived, sharp start for mass 18
example[3, 2:4] += [2, 4]; example[3, 4:] += 6  # Long-lived and blurrier start for mass 28

# Generate labels from coordinate values and generate models by repetition:
repeated = repeat_model(models.ExponentialGaussianModel,
                        example.coords['selected_mass'].values, numeric_prefix='m')
dict_of_signals = dict(zip(repeated, example)) # Convert the 2D-data to a matching dict

# Configure some parameters to be shared across the signals
initial = multi_constrain(repeated, multi_make_params(repeated), {
    'center': True,  # Share 'center' by all
    'sigma': {'m17', 'm18'}  # Share 'sigma' between two of the signal labels
})
# Guess initial values (and average the separate guesses for the shared parameters)
initial = multi_guess(repeated, dict_of_signals, param_hints=initial)
result = multi_fit(repeated, dict_of_signals, initial)  # x will be found automatically
print(result.fit_report())

plt.figure(clear=True)
for label, curve in result.multi_eval().items():
    m = dict_of_signals[label].selected_mass
    plotted = plt.plot(x, curve, '-', label=f'Ion mass {m.values} {m.units}: "{label}"')
    plt.plot(x, dict_of_signals[label], '.:', color=plotted[0].get_color())
plt.legend(loc='upper right')
plt.xlabel('Time (arb. u.)')
plt.ylabel('Signal (arb. u.)')
"""
import re
from typing import Any
import warnings

import asteval
import numpy as np
from numpy.typing import ArrayLike

from lmfit import Model, Parameter, Parameters
from lmfit.models import ExpressionModel


def multi_fit(models: dict[str, Model] | list[Model],
              data: dict[str, ArrayLike] | list[ArrayLike],
              params: Parameters,
              weights: dict[str, ArrayLike] | list[ArrayLike] | None = None,
              separate_kwargs: dict[str, dict[str, Any]] = {},
              **kwargs: dict[str, Any]):
    """Run a global fit of several datasets with parameters possibly shared between models.

    Data, models, separate_kwargs and optional weights are labelled in a consistent way within
    their containers. The containers can either all be dict-like with (preferably string) keys
    or all be list-like with automatic numeric indices.
    Example:
        models = {'this': one_model, 'that': another_model}
        data = {'this': some_data, 'that': the_other_data}
    Example:
        models = [one_model, another_model]
        data = [some_data, the_other_data]

    The point of a global fit, as opposed to separately fitting each model to
    its dataset, is the possibility of sharing some parameters between the models.
    Any models that use the same parameter name will use its shared value,
    but sharing can be introduced for differently named (or prefixed) parameters
    by setting the expression (`expr`) of some `Parameter`-instances to refer to
    the name of another Parameter.

    Parameters
    ----------
    models : dict[str, Model] | list[Model]
        The labelled models, where ``models[label]`` should fit ``data[label]``.
        The independent variables that a model needs may be provided in `kwargs` or
        `separate_kwargs` (or ``data[label].coords`` if ``data[label]`` is an `xarray.DataArray`).
        Each model will be evaluated like
        ``models[label].eval(**{**params, **kwargs, **separate_kwargs[label]})``
        so that if an argument is defined in both `kwargs` and ``separate_kwargs[label]``,
        the latter value is used.
    data : dict[str, ArrayLike] | list[ArrayLike]
        Labelled arrays of data to be fit, one array per model.
    params : Parameters
        The parameters, which will be passed to all models.
    weights : dict[str, ArrayLike] | list[ArrayLike] | None, optional
        Weights, for any dataset where the weights are not all ones.
        Note that if weights are given for any dataset, the weights are implicityly
        set to 1 for every point in other datasets -- which does matter for how
        for how residuals are compared between different datasets.
        The default is None.
    separate_kwargs : dict[str, dict[str, Any]], optional
        To work around name conflicts, ``separate_kwargs[label]`` holds any keyword arguments that
        should be passed to a specific model but not to all and not to `Model.fit()`.
        This can be used to provide underlying models with different arrays for their independent
        variables (even if named 'x' in all of them). The default is {}.
    **kwargs : dict[str,Any]
        Any keyword arguments that should be passed to `Model.fit()` (e.g. fit algorithm options)
        and to the `Model.eval()` of all underlying models (e.g. independent or problem variables).
        The function within a model will still not receive any unexpected argument that would raise
        a `TypeError`, because `Model.make_funcargs()` applies filtering based on unprefixed names
        extracted from the signature of the function (listed in `Model._func_allargs`). However,
        if a model function accepts arbitrary keyword arguments (`Model._func_haskeywords` is True),
        then that function will also get arguments meant for other models within the `multi_fit()`.
        If that is a problem, use `separate_kwargs` to limit which models get arguments.

    Returns
    -------
    result_1D : ModelResult
        Currently this is the 1D ModelResult of fitting the concatenated (flattened) data
        with a few extra attributes:
            `.models`, `.multi_data` and `.multi_weights` hold the dicts or lists of labelled
            data, models and weights that were used.
            `.multi_eval()` is similar to result_1D.eval() but returns a dict or list
            with the separate labelled arrays for each underlying model.
        TODO: It could be useful to define a MultiModelResult class holding this additional data
        and perhaps some methods to iterate over the labels. There should be a method taking a
        label and returning the subset of parameters relating to that underlying model, or even
        constructing and returning a `ModelResult`-instance as if that model had been fitted alone.

    See Also
    --------
    repeat_model()
    multi_make_params()
    multi_constrain()
    multi_guess()
    """
    just_return_model = kwargs.pop('_just_return_model', False)
    if len(data) < 1:
        raise ValueError("At least one labelled data array is required.")
    data, numeric_indices = _dictify(data)
    labels = tuple(data.keys())
    any_xarray = any(hasattr(d, 'coords') for d in data.values())

    models, numeric = _dictify(models)
    numeric_indices &= numeric
    if set(models) != set(labels):
        raise ValueError("models and data must use the same set of labels "
                         "(string keys or numeric indices).")

    if weights is not None:
        weights, _ = _dictify(weights)
        if all(w is None for w in weights.values()):
            weights = None  # Simplify when the weights-dict didn't contain any actual weights
    if weights is not None:
        if set(weights) - set(labels):
            raise ValueError(f"There are weights for {set(weights) - set(labels)} "
                             "but no corresponding data.")

    separate_kwargs, _ = _dictify(separate_kwargs)
    if set(separate_kwargs) - set(labels):
        raise ValueError(f"There are separate_kwargs for {set(separate_kwargs) - set(labels)} "
                         "but no corresponding data.")

    # Produce flattened 1D-arrays for data and weights
    data_1D = _concatenate(data, labels)
    if weights is not None and len(weights) >= 1:
        # Weights are specified (or possibly set to None) for some dataset.
        # For padding, use the dtype of the first specified weights (to allow float32 or complex).
        weight_dtype = np.float64  # Default
        for w in weights.values():
            if w is not None:
                weight_dtype = w.dtype
                break
        # Need to pad with ones for the datasets where weights are not specified.
        for label in labels:
            if label not in weights or weights[label] is None:
                weights[label] = np.ones(data[label].shape, weight_dtype)

        weights_1D = _concatenate(weights, labels)
        if weights_1D.shape != data_1D.shape:
            for label in labels:
                if weights[labels].shape != data[labels].shape:
                    raise ValueError(f'For "{label}" the data has shape {data[labels].shape} '
                                     f'but the weights has shape {weights[labels].shape}.')
            raise ValueError(f"Concatenated weights to shape {weights_1D.shape} "
                             f"which differs from the data's {data_1D.shape}.")
        if np.all(weights_1D == 1):
            print("INFO`: All weights are 1, so it would seem more efficient to set weights=None.")
            # TODO: Should we enforce this optimization? Or skip the check?
    else:
        weights_1D = None

    # Produce merged lists of the parameter names and independent variables, avoiding duplication.
    param_names = []
    independent_vars = []
    param_root_names = []
    for m in models.values():
        for n in m.param_names:
            if n not in param_names:
                param_names.append(n)
        for n in m.independent_vars:
            if n not in independent_vars:
                independent_vars.append(n)
        for n in m._param_root_names or m.param_names:
            # It seems relevant to know also the actual parameters to the model function
            # (skipping auxiliary parameters computed by expression, such as GaussianModel's fwhm
            # computed from sigma, which are in param_names after make_params() has been called)).
            # m._param_root_names can be None for some models, then m.param_names is used.
            if m.prefix + n not in param_root_names:
                param_root_names.append(m.prefix + n)
    # Model.eval() calls Model.make_funcargs() which only selects those arguments
    # with names listed by model.param_names, so we need the global parameter name
    # list for the global model.

    def multi_eval(*positional, **args):
        """Return the concatenated evaluations of all underlying models.

        Will be called with arguments containing the parameters and the common
        kwargs but not necessarily all independent variables.
        """
        if positional:
            print("WARNING: multi_eval() uses keyword arguments. Got unexpected positional "
                  f"arguments: {positional}")
        if 'separate_kwargs' in args:
            # To allow calling result.eval() with values for the independent variables,
            # separate_kwargs from the args can override those originally passed to multi_fit().
            current_separate_kwargs = args.pop('separate_kwargs')
        else:  # Normally
            current_separate_kwargs = separate_kwargs
        # To also be able to get the underlying eval()-arays separately, in result.multi_eval().
        return_separately = args.pop('_return_separately', False)

        # Reimplementing _concatenate()'s label-ordered case here, using list instead
        # of tuple just to allow multiple lines of code in the loop that produces it.
        underlying_arrays = []
        for label in labels:
            model = models[label]
            # Note, although the docstring shows
            # `models[label].eval(**{**params, **kwargs, **separate_kwargs[label]})`
            # the merging of Parameters and keyword arguments has already happened
            # in Model.eval() which calls Model.make_funcargs(), so from here on
            # we consider only plain keyword arguments (no Parameters-instance).
            separate = current_separate_kwargs.get(label, {})
            if separate:
                model_args = {**args, **separate}  # separate can override args
                copied = True
            else:
                # As long as no change is needed, the all-model args can be used by reference.
                model_args = args
                copied = False

            if any_xarray:
                for name in model.independent_vars:
                    if name not in model_args and name in getattr(data[label], 'coords', {}):
                        # Found independent variable as coordinate of xarray.DataArray
                        if not copied:
                            # model_args needs to differ from the all-model args.
                            model_args = model_args.copy()
                            copied = True
                        model_args[name] = data[label].coords[name].values

            # Evaluate one underlying model and append its output
            underlying_arrays.append(np.asanyarray(model.eval(**model_args)).ravel())

        if return_separately:
            # Return the underlying eval()-arays separately, e.g. in result.multi_eval().
            if numeric_indices:
                # This list of arrays will become a 2D array due to lmfit's coerce_arraylike().
                return underlying_arrays
            else:
                return dict(zip(labels, underlying_arrays))
        else:
            # Return 1D-array, e.g. for use while fitting
            return np.concatenate(underlying_arrays)

    # Wrap it in a single-use `Model`-instance
    underlying_model_types = set()
    for m in models.values():
        underlying_model_types.add(type(m).__name__)
    # Ensure a stable, alphatbetical order of the shown model class names
    underlying_model_types = ', '.join(sorted(underlying_model_types))
    model_name = f"Global fit of {len(labels)} models of types {underlying_model_types}"
    # To prevent Model._parse_params() from complaining about multi_eval() not having a signature
    # that explicitly lists the parameter names and independent variables, we pass func=None to
    # the constructor then set the func afterwards. An empty list of independent_vars is used,
    # since the base lmfit.Model would not be able to find them (e.g. in separate_kwargs).
    multi_model = Model(None, independent_vars=[], param_names=param_names, name=model_name)
    multi_model.func = multi_eval
    multi_model._param_names = param_names  # Will be returned by multi_model.param_names
    multi_model._param_root_names = param_root_names
    # We could produce _func_allargs (holding non-prefixed arguments from any underlying model)
    # multi_model._func_allargs = reduce(set.union, (set(m._func_allargs) for m in models.values()))
    # but it seems more efficient and less confusing to leave multi_model._func_allargs empty
    # and just make Model._func_allargs() pass in all provided parameters and keyword arguments.
    multi_model._func_haskeywords = True

    # Keep attributes with info about the underlying models, even if multi_eval() works by closure.
    multi_model.underlying_models = models
    multi_model.underlying_independent_vars = independent_vars

    # For simple cases, we can implement make_params() and guess() by calling the corresponding
    # multi_...-function with the dict of models and no optional arguments.
    # Users are however recommended to directly use multi_make_params() andmulti_guess().
    multi_model.make_params = lambda v=False, **values: multi_make_params(models, **values)
    multi_model.guess = lambda data, x=None, **kws: multi_guess(models, _dictify(data), x, **kws)

    # HINT: multi_model.param_hints is probably not used (and it is left empty). Instead,
    # multi_make_params() calls make_params() on each underlying model (which uses its param_hints)
    # and multi_guess() can take a param_hints argument.

    if just_return_model:
        # Special case to allow tests, or perhaps to reuse the model for repeated fits.
        return multi_model

    with warnings.catch_warnings():
        # Ignore that Model.fit() warns when the multi_model is given keyword arguments with
        # parameters not listed in multi_model.independent_vars (but needed by underlying models).
        warnings.filterwarnings('ignore', 'The keyword argument .+ does not match any arguments '
                                          'of the model function.')
        # Run the usual Model.fit()
        result_1D = multi_model.fit(data_1D, params, weights_1D, **kwargs)

    # TODO Consider defining a MultiModelResult-class with some method to to get produce a
    # 1D ModelResult with that info for each label and only its subset of result_1D.params & data,
    # as if a fit had been run by that underlying model.
    # For now, just insert some more attributes in the 1D ModelResult):
    result_1D.models = models
    result_1D.multi_data = data
    result_1D.multi_weights = weights
    result_1D.multi_eval = (lambda params=None, **kwargs:
                            result_1D.eval(params=params, _return_separately=True, **kwargs))

    return result_1D


def _dictify(dict_or_iterable):
    """Return a dict, even if dict_or_iterable was other kind of iterable (e.g. list)."""
    if hasattr(dict_or_iterable, 'items'):  # Already dict-like.
        # Create a shallow copy to not allow modification by reference.
        # To not also require .copy() method on the dict-like object, use a dict comprehension here.
        # return {label: content for label, content in dict_or_iterable.items()}
        return dict(dict_or_iterable.items()), False
    else:
        # Use numeric indices in the dict
        try:
            return dict(enumerate(dict_or_iterable)), True
        except TypeError as e:
            raise TypeError("Expecting dict (with string labels) or list (with integer indices) "
                            "for data, models, weights and model-specific keyword arguments. "
                            f"Got type {type(dict_or_iterable).__name__}.") from e


def _concatenate(dict_of_arrays, ordered_by_labels=None):
    """Produce an 1D array by flattening and concatenating contents of dict_of_arrays."""
    # The calls to np.asanyarray() are needed to convert array-like objects
    # that don't implement .ravel(), e.g. xarray.DataArray(). When the DataArray()
    # wraps a numpy array, it doe not cause any copying of the underlying array.
    if ordered_by_labels is None:
        return np.concatenate(tuple(np.asanyarray(a).ravel() for a in dict_of_arrays.values()))
    else:
        # Use a predefined order (and skip any unexpected label)
        return np.concatenate(tuple(np.asanyarray(dict_of_arrays[label]).ravel()
                                    for label in ordered_by_labels))


def multi_make_params(models: dict[str, Model] | list[Model],
                      separate_assignments: dict[str, dict[str, float | dict[str, Any]]] = {},
                      **common_assignments: dict[str, float | dict[str, Any]]):
    """Merge the `Parameters` obtained by calling `Model.make_params()` on all the models.

    `Model.make_params()` is called on each underlying model. ``Model.param_hints[parameter_name]``
    may hold a dict to configure its default 'value', 'min', 'max', 'expr' (expression) and 'vary'.
    You may override these by providing user-defined hint-dicts via keyword arguments for each
    parameter, and/or separate_assignments holding a `param_hints` dict per model label.

    At the top level (where parameters are typically prefixed), expressions that refer between the
    underlying models can be set manually either via param hints in the assignments-arguments to
    this function, manually on the returned Parameters or conveniently for typical cases by passing
    the returned `Parameters`-object to `multi_constrain()`.

    Parameters
    ----------
    models : dict[str,Model]
        The labelled models, where ``models[label]`` should fit ``data[label]``.
    separate_assignments : dict[str, dict[str, Any]], optional
        ``separate_assignments[label][parameter_name]`` may be provided to specify an initial
        value or a dict of `Parameter`-attributes (like `Model.param_hints`), that will be passed
        only to ``models[label]``. The default is {}.
    **common_assignments : dict[str, dict[str, Any]], optional
        A keyword argument of ``parameter_name`` may be provided to specify an initial value or a
        dict of `Parameter`-attributes (like `Model.param_hints`). The prefix of each model
        will be inserted before the ``parameter_name``, unless that explicit parameter has
        been specified too. Assignments can be overridden for some model(s) by specifying
        ``separate_assignments[label][parameter_name]``. The default is {}.

    Returns
    -------
    params : Parameters
    """
    models, _ = _dictify(models)
    params = Parameters()
    for label, model in models.items():
        assignments = {**common_assignments, **separate_assignments.get(label, {})}
        # Inject the copies of common_assignments with the prefix expected by this model
        for name, hint in common_assignments.items():
            if name in model.param_names:
                continue  # No need
            prefixed_name = model.prefix + name
            if prefixed_name in model.param_names and prefixed_name not in assignments:
                assignments[prefixed_name] = hint

        new_params = model.make_params(**assignments)
        duplicate_names = params.keys() & new_params.keys()
        if duplicate_names:
            for name in duplicate_names.copy():
                if (params[name] == new_params[name]
                        and params[name].min == new_params[name].min
                        and params[name].max == new_params[name].max):
                    # No need to warn if the value, expr and bounds are identical
                    duplicate_names.remove(name)
            # TODO: Should we use the warning() framework to control whether conflicts
            # get printed or raise an exception? Or use a machinery common with multi_guess()
            # so that averaging of the conflicting values would be possible here too.
            print(f'Parameters {duplicate_names} replaced by those from "{label}".')
        params += new_params
    return params


def multi_constrain(models: dict[str, Model] | list[Model],
                    params: Parameters,
                    grouping: dict[str, bool | set | dict[str, str] | list[str]]):
    """Constrain some prefixed parameter(s) to the value of a differently prefixed parameter.

    If the related parameters are named arbitrarily instead of with prefixes before a common name,
    you can not use this function. Set the `expr` of the relevant parameters directly instead.

    If you want to use `multi_guess()` too, pass the result of multi_constrain() as the param_hints-
    argument there to retain the grouping expressions and combine (e.g. average) the individual
    guesses for grouped parameters. If you instead apply grouping after guessing, the guesses
    of all but one of the shared parameter instances would be ignored.

    Parameters
    ----------
    models: dict[str, Model] | list[Model]
        The labelled models, for mapping a label to models[label].prefix.
    params : Parameters
        The set of parameters.
    grouping : dict[str, bool | set | dict[str, str] | list[str]]
        Specify if and how parameters should be shared between prefixed models.
        If defined, ``grouping[base_name]`` controls the sharing of values between parameters
        named ``prefix + base_name``.
        * If ``grouping[base_name]`` is True, all prefixed versions of that parameter are
          fitted using a common value (with the first of the prefixes).
        * If ``grouping[base_name]`` is a set, models whose labels (not necessarily prefixes) are in
          that set will share a parameter while other models will have independent parameters.
        * If ``grouping[base_name]`` is a dict, ``grouping[base_name][lackey_label] = boss_label``
          means that ``models[lackey_label].prefix + base_name`` will be sharing the value of
          ``models[boss_label].prefix + base_name``. Labels not mentioned or referring to themselves
          will remain independent. This can be used regardless of whether string or numeric labels.
        * If ``grouping[base_name]`` is a list it is required to be of the same length as models
          (which also must be a list). Then ``grouping[base_name][index] = boss_index`` means
          that ``models[index].prefix + base_name`` will be sharing the value of
          ``models[boss_index].prefix + base_name``.
          Parameters where ``source_index == index`` remain independent.

    Returns
    -------
    params : Parameters
        A copy of the given Parameters-instance, possibly with some Parameter.expr modified.
    """
    params = params.copy()  # Don't modify it by reference
    original_models = models
    models, _ = _dictify(models)
    labels = list(models.keys())
    for base_name, specification in grouping.items():
        # HINT: The boss--lackey naming is chosen by the recommendation (35 to) 46 minutes into
        # https://player.fm/series/series-2577577/ep-111-mnemonic-serenading-and-mechanism-naming.
        if specification is True:  # All grouped
            boss = labels[0]
            specification = {lackey: boss for lackey in labels[1:]}
        elif isinstance(specification, set):  # One group for a set of labels
            unexpected = specification - set(labels)
            if unexpected:
                raise ValueError(f"Grouping of '{base_name}' is specified for labels {unexpected} "
                                 "with no corresponding model.")
            boss = min(specification)
            specification = specification.copy()
            specification.remove(boss)
            specification = {lackey: boss for lackey in specification}
        elif isinstance(specification, list):  # Source/controlling index specified for each model
            if not isinstance(original_models, list) or len(models) != len(specification):
                raise ValueError("List of controlling indices is only allowed when models is a "
                                 f"list of the same length. Got {specification} for '{base_name}'.")
            specification = dict(enumerate(specification))
        else:
            assert isinstance(specification, dict)
        dependent_count = 0
        for lackey, boss in specification.items():
            if boss != lackey:  # Share another parameter's value
                params[models[lackey].prefix + base_name].set(expr=models[boss].prefix + base_name)
                dependent_count += 1
            else:  # Independent parameter
                params[models[lackey].prefix + base_name].set(expr=None)
        if dependent_count >= len(labels):
            raise ValueError("Grouping is specified in a cyclic way, without any prefixed "
                             f"version of '{base_name}' being freely fitted.")
    return params


def multi_guess(models: dict[str, Model] | list[Model],
                data: dict[str, ArrayLike] | list[ArrayLike],
                separate_kwargs: dict[str, dict[str, Any]] = {},
                combine_values: str = 'mean',
                combine_bounds: str = 'loosen_finite',
                include_nonfinite: bool = False,
                print_unimplemented: bool = True,
                param_hints: dict[str, dict[str, Any]] = {},
                **kws):
    """Guess starting values for the parameters of all underlying models, and combine them.

    This starts by calling `multi_make_params()` to create all the parameters, which can be
    influenced via the `param_hints`-argument. Make sure that any parameter sharing/grouping
    (e.g. by `multi_constrain()`) is configured first and provided in the `param_hints`-argument,
    so that the individual guesses for grouped parameters can be combined (e.g. averaged) here.
    See the `combine_valuess` and `combine_bounds` for customizing how.

    It is optional for models to implement guess() and this method returns an empty Parameters-
    object in case no guesses were provided. The keyword argument print_unimplemented can be set
    to True to print a notice about models that do not implement `guess()`.

    Parameters
    ----------
    models : dict[str, Model] | list[Model]
        The labelled models, where ``models[label]`` should fit ``data[label]``.
        The independent variables that a model needs may be provided in `kwargs` or
        `separate_kwargs` (or ``data[label].coords`` if ``data[label]`` is an `xarray.DataArray`).
        See `multi_fit()` for details.
    data : dict[str, ArrayLike] | list[ArrayLike]
        Labelled arrays of data to inform the guess (and later be fit), one array per model.
    separate_kwargs : dict[str, dict[str, Any]], optional
        To work around name conflicts, `separate_kwargs[label]` holds any keyword arguments that
        should be passed to a specific model but not to all and not to `Model.fit()`.
        This can be used to provide underlying models with different arrays for their independent
        variables (even if named 'x' in all of them). The default is {}.
    combine_values : str, optional
        Determines how to combine guesses for a parameter shared by by seveal underlying models:
        'mean', 'median', 'min', 'max', 'first' or 'last'. The default value is 'mean'.
        'first' means a guessed parameter is never updated by later model, while
        'last' means that the last model overrides (if it provides a finite value).
        Non-finite values are mostly ignored, i.e. they only remain in final guess
        if no model made a finite guess or if the 'first'-option was used.
    combine_bounds : str, optional
        Determines how to combine bounds for a parameter shared by by seveal underlying models:
        'mean', 'median', 'tighten', 'loosen', 'loosen_finite', 'first' or 'last'.
        Both 'loosen' and 'loosen_finite' mean that a wider numeric bound overrides a narrower,
        but 'loosen_finite' will let a finite bound override a (default) infinite bound.
        The default value is 'loosen_finite'.
    include_nonfinite : bool, optional
        For parameters where no guess is made, parameter hints may still
        provide values and bounds via `Model.make_params()`.
        Set include_nonfinite = True to allow returning even non-finite hints.
        The default value is False, which means that any hint whose value
        is -inf or +inf or NaN is not returned in the guessed parameters.
        This allows the pattern
            `initial = model.make_params(foo=2, bar=5) + model.guess(data, x=x)`
        for specifying defaults and overriding with guesses where available.
    print_unimplemented : bool, optional
        True: Print a notice for each model not implementing guess()
        False: Don't print anything.
        The default value is True.
    param_hints : dict[str, dict[str, Any]] | Parameters
        Allows customizing parameter attributes before the guesses of models are combined.
        The main use case is to to set 'expr' to cause grouping of parameters (and for instance
        averaging of the guesses from individual models, see `combine_values`).
        If you set finite 'min' or 'max' bounds they override any bounds set by a model's guess
        (irrespective of the `combine_bounds`-argument). Hints for 'value' and 'vary' will however
        be overridden by guesses from models.
        You can also define new parameters which no underlying model provides a guess for.
        * If `param_hints` is a dict, its contents will be passed as keyword arguments to
          `multi_make_params()`. `param_hints[parameter_name]` should then be an inner dict like in
          `Model.param_hints`, with keys like 'expr', 'min' and 'max' (and 'name' to create a new).
        * If `param_hints` is a `Parameters`-object, it is used as-is without any call to
          `multi_make_params()`. Min and max bounds of +-inf will in this case not override
          finite bounds provided by the model.
        The default value is {}.

    **kwargs : dict[str,Any]
        Any keyword arguments that should be passed to `Model.fit()` (e.g. fit algorithm options)
        and to the `Model.eval()` of all underlying models (e.g. independent or problem variables).
        The function within a model will still not receive any unexpected argument that would raise
        a `TypeError`, because `Model.make_funcargs()` applies filtering based on unprefixed names
        extracted from the signature of the function (listed in `Model._func_allargs`). However,
        if a model function accepts arbitrary keyword arguments (`Model._func_haskeywords` is True),
        then that function will also get arguments meant for other models within the `multi_fit()`.
        If that is a problem, use `separate_kwargs` to limit which models get arguments.

    Returns
    -------
    parameters : Parameters
        Initialized using `multi_make_params()` and/or `param_hints` then updated with guesses and
        bounds from the underlying models. In case of grouping, the value, min and max for the
        boss-parameter are chosen by considering also the individual guesses for lackey-parameters.

    Example
    -------
    ``initial = multi_make_params(models)``
    ``initial = multi_constrain(models, initial, {'center': True})``  # Share the 'center'-parameter
    ``initial = multi_guess(models, data, separate_kwargs=x_dict, param_hints=initial)``

    Example
    -------
        # Explicit expressions, without first using multi_make_params() or multi_constrain().
        assert list(m.prefix for m in models.values()) == ['a_', 'b_', 'c_']
        # Make b_center and c_center share a_center's value
        param_hints = {'b_center': {'expr': 'a_center'}, 'c_center': {'expr': 'a_center'}}
        initial = multi_guess(models, data, separate_kwargs=x_dict, param_hints=param_hints)
    """
    models, _ = _dictify(models)
    if combine_values not in {'mean', 'median', 'min', 'max', 'first', 'last'}:
        raise ValueError(f'Invalid value for combine_values: "{combine_values}"')
    if combine_bounds not in {'mean', 'median', 'tighten', 'loosen', 'loosen_finite',
                              'first', 'last'}:
        raise ValueError(f'Invalid value for combine_bounds: "{combine_bounds}"')

    if isinstance(param_hints, Parameters):
        # Special case, allow passing a set of Parameters, after possible customization.
        parameters = param_hints.copy()
        # Convert to dict-type hint at least for the 'expr', as expected below.
        param_hints = {}
        for n, p in parameters.items():
            hints = {}
            if p.expr:
                hints['expr'] = p.expr
            # For min & max, which can't be left blank when providing Parameters-instance,
            # assume that only finite bounds are actually intended to override the model defaults.
            if np.isfinite(p.min):
                hints['min'] = p.min
            if np.isfinite(p.max):
                hints['max'] = p.max
            param_hints[n] = hints
    else:
        # Start with the default parameters, including param_hints of each underlying model.
        # Model.make_params() requires hinted parameters to have either 'value' or 'expr'
        # but those shouldn't be required here as we will be getting values from Model.guess().
        # Exclude any such parameters from the multi_make_params()-call.
        filtered_hints = {n: hint for n, hint in param_hints.items()
                          if not isinstance(hint, dict) or 'value' in hint or 'expr' in hint}
        parameters = multi_make_params(models, **filtered_hints)
    parameter_hinted_but_not_guessed = set(parameters)
    parameter_aliases = {}
    values = {}
    mins = {}
    maxs = {}
    for label, model in models.items():
        if isinstance(label, int):
            quoted_label = f'#{label}'
        else:
            quoted_label = f'"{label}"'
        try:
            model_args = {**kws, **separate_kwargs.get(label, {})}
            model_data = data[label]
            if hasattr(model_data, 'coords'):
                for name in model.independent_vars:
                    if name not in model_args and name in model_data.coords:
                        # Found independent variable as coordinate of xarray.DataArray
                        model_args[name] = model_data.coords[name].values
                model_data = model_data.values  # Get the plain numpy.ndarray
            new_parameters = model.guess(model_data, **model_args)
        except NameError as e:
            suspicions = ''
            for param_name, hint in model.param_hints.items():
                expr = hint.get('expr', '')
                if expr and expr not in model.param_names:
                    # This only ignores the most trivial safe expressions, but better than no check
                    if not suspicions:
                        suspicions = f'\nConsider the following expressions: {param_name}="{expr}"'
                    else:
                        suspicions += f', {model._prefix + param_name}="{expr}"'
            raise NameError(f"NameError in model {quoted_label} {repr(model)}. This may happen if "
                            "a parameter hint with an expression referring to a parameter in "
                            "another model has incorrectly been assigned at the level of an "
                            "underlying model. Use the param_hints argument or multi_constrain() "
                            f"instead.{suspicions}") from e
        except NotImplementedError:
            if print_unimplemented:
                print(f"Notice: No guess() for model {quoted_label} {model}.")
                continue
        except TypeError as e:
            if ("guess() missing 1 required positional argument: 'x'" in str(e)
                    and model.guess.__func__ == Model.guess):
                # This happens if guess() is not implemented, and the independent variable
                # has another name than the 'x' assumed by base Model.guess()
                if print_unimplemented:
                    print(f"Notice: Could not guess() for model {quoted_label} {model}: {e}")
                    continue
            else:  # Seems to be a real error
                raise

        if len(parameters) == 0:
            parameters = new_parameters
            parameter_hinted_but_not_guessed = set()
            continue
        if combine_bounds == 'last' and combine_values == 'last':  # New bounds replace old bounds
            parameters += new_parameters
            parameter_hinted_but_not_guessed -= set(new_parameters)
            continue
        try:
            for n, p in new_parameters.items():
                if combine_values not in ('first', 'last'):
                    hint = param_hints.get(n, None)
                    if hint and hint.get('expr'):
                        # This parameter is not directly fitted, but computed from other parameters.
                        parameter_aliases[n] = hint['expr']
                        if hint['expr'] in parameters:
                            # Sharing another parameter's value (due to grouping). Let the lackey's
                            # guess contribute to the combined guess for the boss-parameter.
                            n = hint['expr']
                        else:
                            # Arbitrary expressions are ignored, as it is is not practical to
                            # try to invert them to influence the guess(es) of used parameter(s).
                            continue
                        # Non-fitted expressions defined already by the model (e.g. GaussianModel's
                        # fwhm computed from sigma with a coefficient in front) have p.expr
                        # rather than a user-provided hint['expr']. But they normally get no guess
                        # or initial value (and those wouldn't influence the fit anyway).

                    # If the user-provided param_hints (which may be Parameters) define min and max
                    # they override the model's guess before combining (not using combine_bounds).
                    if hint and hint.get('min'):
                        p.min = hint['min']
                    if hint and hint.get('max'):
                        p.max = hint['max']

                if n not in parameters or n in parameter_hinted_but_not_guessed:
                    parameters[n] = p  # No other model has made a guess for this parameter
                    parameter_hinted_but_not_guessed.remove(n)
                    continue
                old = parameters[n]

                if combine_values == 'first':
                    pass
                elif combine_values == 'min':
                    if np.isfinite(p.value):
                        if np.isfinite(old.value):
                            old.value = min(old.value, p.value)
                        else:
                            old.value = p.value
                elif combine_values == 'max':
                    if np.isfinite(p.value):
                        if np.isfinite(old.value):
                            old.value = max(old.value, p.value)
                        else:
                            old.value = p.value
                elif combine_values == 'last':
                    if np.isfinite(p.value):
                        if combine_bounds != 'tighten':
                            if p.value < old.min:
                                old.min = p.value  # Adjust bound too
                            if p.value > old.max:
                                old.max = p.value  # Adjust bound too
                        old.value = p.value
                else:  # 'mean' or 'median'
                    if n not in values:
                        values[n] = []
                        if np.isfinite(old.value):
                            values[n].append(old.value)
                    if np.isfinite(p.value):
                        values[n].append(p.value)

                if combine_bounds == 'first':
                    pass
                elif combine_bounds == 'exact' and (old.min != p.min or old.max != p.max):
                    raise ValueError(f'Bounds for {p} in model {quoted_label} differ from {old}.')
                elif combine_bounds == 'tighten':
                    old.min = max(old.min, p.min)
                    old.max = min(old.max, p.max)
                    # NOTE: old.value could potentially be outside its bounds now
                elif combine_bounds == 'loosen' or combine_bounds == 'loosen_finite':
                    tentative_min = min(old.min, p.min)
                    tentative_max = max(old.max, p.max)
                    if combine_bounds == 'loosen_finite' and not np.isfinite(tentative_min):
                        tentative_min = old.min if np.isfinite(old.min) else p.min
                    if combine_bounds == 'loosen_finite' and not np.isfinite(tentative_max):
                        tentative_max = old.max if np.isfinite(old.max) else p.max
                    old.min = tentative_min
                    old.max = tentative_max
                elif combine_bounds == 'last':
                    old.min = p.min
                    old.max = p.max
                else:  # 'mean' or 'median'
                    if n not in mins:  # Initializing list when the second guess is made
                        mins[n] = [old.min, p.min]
                        maxs[n] = [old.max, p.max]
                    else:
                        mins[n].append(p.min)
                        maxs[n].append(p.max)
        except NameError as e:
            suspicions = ''
            for param_name, hint in model.param_hints.items():
                expr = hint.get('expr', '')
                if expr and expr not in model.param_names:
                    # This only ignores the most trivial safe expressions, but better than no check
                    if not suspicions:
                        suspicions = f'\nConsider the following expressions: {param_name}="{expr}"'
                    else:
                        suspicions += f', {model._prefix + param_name}="{expr}"'
            raise NameError(f"NameError when using guess of model {quoted_label} {repr(model)}. "
                            "This may happen if a parameter hint with an expression referring to a "
                            "parameter in another model has incorrectly been assigned at the level "
                            "of an underlying model. Use the param_hints argument or "
                            f" multi_constrain() instead.{suspicions}") from e

    for n, entries in values.items():
        if len(entries) > 0:
            if combine_values == 'median':
                parameters[n].value = float(np.median(entries))
            else:  # assume 'mean'
                parameters[n].value = float(np.mean(entries))

    if combine_bounds == 'median':
        averaging_function = np.median
    else:  # assume 'mean'
        averaging_function = np.mean
    for n, entries in mins.items():
        avg = float(averaging_function(entries))
        if np.isnan(avg):
            avg = -np.inf
        parameters[n].min = avg
        avg = float(averaging_function(maxs[n]))
        if np.isnan(avg):
            avg = np.inf
        parameters[n].max = avg

    # HINT: Within model.make_params() and thus within model.guess() the Parameters for one
    # underlying model can not have an expr referring to parameters defined in another model.
    # Since multi_fit() needs such expressions to achieve grouping (shared values), such expressions
    # must be set in the top level Parameters-object used by multi_fit(). That can be done
    # after/without multi_guess() using multi_constrain(), or here via the param_hints argument
    # which is (re)applied now after having collected the parameters:
    for name, hint in param_hints.items():
        if isinstance(hint, dict) and hint.get('expr'):
            expr = hint['expr']
            if name in parameters:
                parameters[name].set(expr=expr)
            elif n in parameter_aliases:
                # The parameter creation was suppressed to let all grouped models
                # contribute their guesses to a combined average or extreme value.
                if parameter_aliases[name] != expr:
                    print(f"Warning: {name} = '{hint['expr']}' used instead of underlying model's "
                          f"'{parameter_aliases[name]}'")
                parameters.add(Parameter(name=name, expr=expr))
    if not include_nonfinite:
        for name in parameter_hinted_but_not_guessed:
            if not np.isfinite(parameters[name].value):
                parameters.pop(name)
    return parameters


def repeat_model(base_model: Any, labels: list[str | int | float] | dict[str, Any] | range,
                 automatic_formatting=True, numeric_prefix='n', labels_like_prefixes=False,
                 assign_names=False, **model_constructor_kwargs):
    """Expand a simple model to allow global fit of multiple signals with different prefixes.

    The idea is to take a base_model representing the "curve shape" and then repeat it with
    different prefixes for each signal.

    Parameters
    ----------
    base_model
        Specifies the simple model that should be repeated.
        If class: An instance of the class is constructed, with a prefix as keyword argument.
            Additional keyword arguments to `repeat_model()` get passed on to the constructor.
        If string: The expression for an ExpressionModel (prefixing is done by string replacements).
                By default, the independent variable is 'x' but you can pass `independent_vars`
                as a keyword argument to `repeat_model()` to change that.
        If function: The function for a Model.
    labels : list[str | int | float] | dict[str, Any] | range
        The model labels or a a list (or array) of prefixes to use for the repeated model instances.
        The length of this argument determines how many repetitions will be created.
        If a dict, then its keys are used as prefixes and the values are ignored.
        If a list, it is allowed to have multiple elements with the same prefix.
        To just use successive integers, a range can be used.
        String elements are always allowed but numbers require automatic_formatting=True.
        Entries with empty strings are allowed, but any parameters with the same prefixed names
        will become shared.
    automatic_formatting : bool
        Apply formatting to convert numbers to strings, abbreviate long strings and ensure
        that each prefix ends with an underscore. The default is True.
    numeric_prefix : str
        When automatic_formatting is used for numeric labels, this string is put before the number
        since variable names must start with a character. The default is 'n'.
    labels_like_prefixes: bool
        Use the prefixes in the returned models-dict without removing underscores from the end.
        The default is False.
    assign_names : bool, optional
        Set each repeated model's `_name`-attribute to its prefix (without any trailing underscore).
        The default is False.
    **model_constructor_kwargs
        Any additional keyword arguments get passed on to the constructor of every model
        which for instance allows setting `independent_vars`.

    Returns
    -------
    models : dict[str, Model]
        The labels and single-signal models for use with `multi_fit()` etc.
    """
    labels, numeric = _dictify(labels)
    if numeric:
        prefixes = list(labels.values())
    else:
        prefixes = list(labels.keys())
    models = {}
    used_prefixes = set()
    for prefix in prefixes:
        # Prepare the prefix string
        if not isinstance(prefix, str):
            if not automatic_formatting:
                raise ValueError("Non-string labels require automatic_formatting=True to be "
                                 "converted into prefixes")
            if np.issubdtype(type(prefix), np.floating):
                prefix = f'{prefix:G}'
                prefix = prefix.replace('E+0', 'Ep').replace('E+', 'Ep')  # Exponential notation
                prefix = prefix.replace('E-0', 'Em').replace('E-', 'Em')  # Exponential notation
                prefix = prefix.replace('-', 'm').replace('.', 'd')  # Minus and decimal separator
            else:  # Handles integers and perhaps some unexpected types
                prefix = str(prefix).replace('-', 'm')
                if not prefix:
                    raise ValueError('Got a non-string label for which str() was empty.')
            prefix = numeric_prefix + prefix
            if not prefix.endswith('_'):
                prefix = prefix + '_'  # Make the prefix end with an underscore
        elif automatic_formatting:
            prefix = re.sub(r'[^A-Za-z_0-9]+', '_', prefix.strip())
            if not re.match('[A-Za-z_]', prefix):
                prefix = '_' + prefix  # Start with underscore to avoid leading digit
            if len(prefix) > 8:  # Try to abbreviate long string label
                for length in range(len(prefix)):
                    if length < 3:
                        # Try initials of length 1, 2 or 3
                        candidate = '_'.join(s[:1 + length] for s in prefix.split('_'))
                    else:
                        candidate = prefix[:length]
                    if not candidate.endswith('_'):
                        candidate = candidate + '_'
                    if candidate not in used_prefixes:  # Found an abbreviation
                        prefix = candidate
                        break
            if not prefix.endswith('_'):
                prefix = prefix + '_'  # Make the prefix end with an underscore
        used_prefixes.add(prefix)

        # Create a Model instance with the prefix
        if isinstance(base_model, type):  # Class
            model_class = base_model
            model = model_class(prefix=prefix, **model_constructor_kwargs)
        elif callable(base_model):  # Function
            model = Model(base_model, prefix=prefix, **model_constructor_kwargs)
        elif isinstance(base_model, str):  # String expression
            # The default independent variable name is 'x'.
            independent_vars = model_constructor_kwargs.get('independent_vars', ['x'])
            if isinstance(independent_vars, str):
                # Autocorrect string with single name to a list with that string
                independent_vars = [independent_vars]
                model_constructor_kwargs['independent_vars'] = independent_vars
            if prefix:
                # ExpressionModel doesn't support prefixing but we can replace names in the string.
                replacements = {name: prefix + name for name in names_in_expression(base_model)}
                for name in independent_vars:
                    replacements.pop(name, None)
                model = ExpressionModel(replace_names_in_expression(base_model, replacements),
                                        **model_constructor_kwargs)
            else:
                model = ExpressionModel(base_model, **model_constructor_kwargs)
            try:
                model.base_expr = base_model
            except Exception:
                pass
        elif isinstance(base_model, Model):
            if prefix:
                raise ValueError('Model-instances can currently not be wrapped in new prefixes.')
                # TODO: If a PrefixedWrapper would be created an already constructed instances of
                # a Model subclass would become usable too, by making it appear with an extra
                # prefix and delegating the evaluation to the original instance.
            else:
                # Can use it unprefixed (the repetition is rather trivial and hardly useful then)
                model = base_model
        else:
            raise ValueError(f'Unexpected type of base_model: {type(base_model)}')
        # TODO: Could consider emitting a warning in case prefixing would unintentionally give
        # a name identical to non-prefixed parameter, e.g. if a prefix of 'p' would turn a
        # 'q'-parameter into 'pq' which already might exist with another meaning.

        if labels_like_prefixes:
            label = prefix
        else:
            label = prefix.rstrip('_')
        if assign_names:
            model.name = label
        models[label] = model

    # if numeric:
    #     models = list(models.values())
    return models


def names_in_expression(expression, also_builtins=False):
    """Return a list of symbol (variable) names used in an expression.

    Parameters
    ----------
    expression : str
        The Python-like expression, to be parsed by asteval. Mathematical
        functions are to be written without any "numpy." prefix.
    also_builtins : bool, optional
        Skip symbol names of known functions and constants (everything in the
        asteval.Interpreter's symtable without evaluating the expression).
        The default is False.

    Returns
    -------
    list
        A list with the found names as strings, without repetition.L

    See Also
    --------
    replace_names_in_expression()
    """
    av = asteval.Interpreter()
    astcode = av.parse(expression.strip())
    sym_names = asteval.get_ast_names(astcode)  # find all symbol names in expression
    if not also_builtins:
        # Drop symbols used as function e.g. 'exp(...)', if not automatically done
        sym_names = [n for n in sym_names if n not in av.symtable]
    return sym_names


def replace_names_in_expression(expression, replacements):
    """Replace words in a string.

    Parameters
    ----------
    expression : str
        The original string, expected but not required to be Python-like program code.
    replacements : dict
        Dict mapping of original string name to new string name.

    Returns
    -------
    str
        The (possibly) modified expression string.
    """
    # From Python 3.9, import ast; ast.unparse(astcode) should be able to produce a string
    # from the syntax tree. Then perhaps ast.walk() or ast.NodeTransformer()-subclass would be
    # usable to iterate over nodes and if a name and expression.id == original: expression.id = new
    # For now, simple string replacements are done:
    return re.sub(r'\b[A-Za-z]\w*\b', lambda match: replacements.get(match[0], match[0]),
                  expression, flags=re.ASCII)
