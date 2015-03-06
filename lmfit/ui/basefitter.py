import warnings
import numpy as np

from ..model import Model
from ..models import ExponentialModel  # arbitrary default
from ..asteval import Interpreter
from ..astutils import NameFinder
from ..parameter import check_ast_errors


_COMMON_DOC = """
    This an interactive container for fitting models to particular data.

    It maintains the attributes `current_params` and `current_result`. When
    its fit() method is called, the best fit becomes the new `current_params`.
    The most basic usage is iteratively fitting data, taking advantage of
    this stateful memory that keep the parameters between each fit.
"""

_COMMON_EXAMPLES_DOC = """

    Examples
    --------
    >>> fitter = Fitter(data, model=SomeModel, x=x)

    >>> fitter.model
    # This property can be changed, to try different models on the same
    # data with the same independent vars.
    # (This is especially handy in the notebook.)

    >>> fitter.current_params
    # This copy of the model's Parameters is updated after each fit.

    >>> fitter.fit()
    # Perform a fit using fitter.current_params as a guess.
    # Optionally, pass a params argument or individual keyword arguments
    # to override current_params.

    >>> fitter.current_result
    # This is the result of the latest fit. It contain the usual
    # copies of the Parameters, in the attributes params and init_params.

    >>> fitter.data = new_data
    # If this property is updated, the `current_params` are retained an used
    # as an initial guess if fit() is called again.
    """


class BaseFitter(object):
    __doc__ = _COMMON_DOC + """

    Parameters
    ----------
    data : array-like
    model : lmfit.Model
        optional initial Model to use, maybe be set or changed later
    """ + _COMMON_EXAMPLES_DOC
    def __init__(self, data, model=None, **kwargs):
        self._data = data
        self.kwargs = kwargs

        # GUI-based subclasses need a default value for the menu of models,
        # and so an arbitrary default is applied here, for uniformity
        # among the subclasses.
        if model is None:
            model = ExponentialModel
        self.model = model

    def _on_model_value_change(self, name, value):
        self.model = value

    def _on_fit_button_click(self, b):
        self.fit()

    def _on_guess_button_click(self, b):
        self.guess()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if callable(value):
            model = value()
        else:
            model = value
        self._model = model
        self.current_result = None
        self._current_params = model.make_params()

        # Use these to evaluate any Parameters that use expressions.
        self.asteval = Interpreter()
        self.namefinder = NameFinder()

        self._finalize_model(value)

        self.guess()

    def _finalize_model(self, value):
        # subclasses optionally override to update display here
        pass

    @property
    def current_params(self):
        """Each time fit() is called, these will be updated to reflect
        the latest best params. They will be used as the initial guess
        for the next fit, unless overridden by arguments to fit()."""
        return self._current_params

    @current_params.setter
    def current_params(self, new_params):
        # Copy contents, but retain original params objects.
        for name, par in new_params.items():
            self._current_params[name].value = par.value
            self._current_params[name].expr = par.expr
            self._current_params[name].vary = par.vary
            self._current_params[name].min = par.min
            self._current_params[name].max = par.max

        # Compute values for expression-based Parameters.
        self.__assign_deps(self._current_params)
        for _, par in self._current_params.items():
            if par.value is None:
                self.__update_paramval(self._current_params, par.name)

        self._finalize_params()

    def _finalize_params(self):
        # subclasses can override this to pass params to display
        pass

    def guess(self):
        count_indep_vars = len(self.model.independent_vars)
        guessing_successful = True
        try:
            if count_indep_vars == 0:
                guess = self.model.guess(self._data)
            elif count_indep_vars == 1:
                key = self.model.independent_vars[0]
                val = self.kwargs[key]
                d = {key: val}
                guess = self.model.guess(self._data, **d)
        except NotImplementedError:
            guessing_successful = False
        self.current_params = guess
        return guessing_successful

    def __assign_deps(self, params):
        # N.B. This does not use self.current_params but rather
        # new Parameters that are being built by self.guess().
        for name, par in params.items():
            if par.expr is not None:
                par.ast = self.asteval.parse(par.expr)
                check_ast_errors(self.asteval.error)
                par.deps = []
                self.namefinder.names = []
                self.namefinder.generic_visit(par.ast)
                for symname in self.namefinder.names:
                    if (symname in self.current_params and
                        symname not in par.deps):
                        par.deps.append(symname)
                self.asteval.symtable[name] = par.value
                if par.name is None:
                    par.name = name

    def __update_paramval(self, params, name):
        # N.B. This does not use self.current_params but rather
        # new Parameters that are being built by self.guess().
        par = params[name]
        if getattr(par, 'expr', None) is not None:
            if getattr(par, 'ast', None) is None:
                par.ast = self.asteval.parse(par.expr)
            if par.deps is not None:
                for dep in par.deps:
                    self.__update_paramval(params, dep)
            par.value = self.asteval.run(par.ast)
            out = check_ast_errors(self.asteval.error)
            if out is not None:
                self.asteval.raise_exception(None)
        self.asteval.symtable[name] = par.value

    def fit(self, *args, **kwargs):
        "Use current_params unless overridden by arguments passed here."
        guess = dict(self.current_params)
        guess.update(self.kwargs)  # from __init__, e.g. x=x
        guess.update(kwargs)
        self.current_result = self.model.fit(self._data, *args, **guess)
        self.current_params = self.current_result.params


class MPLFitter(BaseFitter):
    # This is a small elaboration on BaseModel; it adds a plot()
    # method that depends on matplotlib. It adds several plot-
    # styling arguments to the signature.
    __doc__ = _COMMON_DOC + """

    Parameters
    ----------
    data : array-like
    model : lmfit.Model
        optional initial Model to use, maybe be set or changed later

    Additional Parameters
    ---------------------
    axes_style : dictionary representing style keyword arguments to be
        passed through to `Axes.set(...)`
    data_style : dictionary representing style keyword arguments to be passed
        through to the matplotlib `plot()` command the plots the data points
    init_style : dictionary representing style keyword arguments to be passed
        through to the matplotlib `plot()` command the plots the initial fit
        line
    best_style : dictionary representing style keyword arguments to be passed
        through to the matplotlib `plot()` command the plots the best fit
        line
    **kwargs : independent variables or extra arguments, passed like `x=x`
        """ + _COMMON_EXAMPLES_DOC
    def __init__(self, data, model=None, axes_style={},
                data_style={}, init_style={}, best_style={}, **kwargs):
        self.axes_style = axes_style
        self.data_style = data_style
        self.init_style = init_style
        self.best_style = best_style
        super(MPLFitter, self).__init__(data, model, **kwargs)

    def plot(self, axes_style={}, data_style={}, init_style={}, best_style={},
             ax=None):
        """Plot data, initial guess fit, and best fit.

    Optional style arguments pass keyword dictionaries through to their
    respective components of the matplotlib plot.

    Precedence is:
    1. arguments passed to this function, plot()
    2. arguments passed to the Fitter when it was first declared
    3. hard-coded defaults

    Parameters
    ---------------------
    axes_style : dictionary representing style keyword arguments to be
        passed through to `Axes.set(...)`
    data_style : dictionary representing style keyword arguments to be passed
        through to the matplotlib `plot()` command the plots the data points
    init_style : dictionary representing style keyword arguments to be passed
        through to the matplotlib `plot()` command the plots the initial fit
        line
    best_style : dictionary representing style keyword arguments to be passed
        through to the matplotlib `plot()` command the plots the best fit
        line
    ax : matplotlib.Axes
            optional `Axes` object. Axes will be generated if not provided.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required to use this Fitter. "
                              "Use BaseFitter or a subclass thereof "
                              "that does not depend on matplotlib.")

        # Configure style
        _axes_style= dict()  # none, but this is here for possible future use
        _axes_style.update(self.axes_style)
        _axes_style.update(axes_style)
        _data_style= dict(color='blue', marker='o', linestyle='none')
        _data_style.update(**_normalize_kwargs(self.data_style, 'line2d'))
        _data_style.update(**_normalize_kwargs(data_style, 'line2d'))
        _init_style = dict(color='gray')
        _init_style.update(**_normalize_kwargs(self.init_style, 'line2d'))
        _init_style.update(**_normalize_kwargs(init_style, 'line2d'))
        _best_style= dict(color='red')
        _best_style.update(**_normalize_kwargs(self.best_style, 'line2d'))
        _best_style.update(**_normalize_kwargs(best_style, 'line2d'))

        if ax is None:
            fig, ax = plt.subplots()
        count_indep_vars = len(self.model.independent_vars)
        if count_indep_vars == 0:
            ax.plot(self._data, **_data_style)
        elif count_indep_vars == 1:
            indep_var = self.kwargs[self.model.independent_vars[0]]
            ax.plot(indep_var, self._data, **_data_style)
        else:
            raise NotImplementedError("Cannot plot models with more than one "
                                      "indepedent variable.")
        result = self.current_result  # alias for brevity
        if not result:
            ax.set(**_axes_style)
            return  # short-circuit the rest of the plotting
        if count_indep_vars == 0:
            ax.plot(result.init_fit, **_init_style)
            ax.plot(result.best_fit, **_best_style)
        elif count_indep_vars == 1:
            ax.plot(indep_var, result.init_fit, **_init_style)
            ax.plot(indep_var, result.best_fit, **_best_style)
        ax.set(**_axes_style)


def _normalize_kwargs(kwargs, kind='patch'):
    """Convert matplotlib keywords from short to long form."""
    # Source:
    # github.com/tritemio/FRETBursts/blob/fit_experim/fretbursts/burst_plot.py
    if kind == 'line2d':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          mec='markeredgecolor', mew='markeredgewidth',
                          mfc='markerfacecolor', ms='markersize',)
    elif kind == 'patch':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          ec='edgecolor', fc='facecolor',)
    for short_name in long_names:
        if short_name in kwargs:
            kwargs[long_names[short_name]] = kwargs.pop(short_name)
    return kwargs
