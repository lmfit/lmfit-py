import warnings
import numpy as np

from ..model import Model

from .basefitter import MPLFitter, _COMMON_DOC, _COMMON_EXAMPLES_DOC

# Note: If IPython is not available of the version is < 2,
# this module will not be imported, and a different Fitter.

import IPython
from IPython.display import display, clear_output
# Widgets were only experimental in IPython 2.x, but this does work there.
# Handle the change in naming from 2.x to 3.x.
IPY2 = IPython.release.version_info[0] == 2
IPY3 = IPython.release.version_info[0] == 3
if IPY2:
    from IPython.html.widgets import DropdownWidget as Dropdown
    from IPython.html.widgets import ButtonWidget as Button
    from IPython.html.widgets import ContainerWidget
    from IPython.html.widgets import FloatTextWidget as FloatText
    from IPython.html.widgets import CheckboxWidget as Checkbox
    class HBox(ContainerWidget):
        def __init__(self, *args, **kwargs):
           self.add_class('hbox')
           super(self, ContainerWidget).__init__(*args, **kwargs)
elif IPY3:
    # as of IPython 3.x:
    from IPython.html.widgets import Dropdown
    from IPython.html.widgets import Button
    from IPython.html.widgets import HBox
    from IPython.html.widgets import FloatText
    from IPython.html.widgets import Checkbox
else:
    # as of IPython 4.x+:
    from ipywidgets import Dropdown
    from ipywidgets import Button
    from ipywidgets import HBox
    from ipywidgets import FloatText
    from ipywidgets import Checkbox


class ParameterWidgetGroup(object):
    """Construct several widgets that together represent a Parameter.

    This will only be used if IPython is available."""
    def __init__(self, par):
        self.par = par

        # Define widgets.
        self.value_text = FloatText(description=par.name,
                                    min=self.par.min, max=self.par.max)
        self.value_text.width = 100
        self.min_text = FloatText(description='min', max=self.par.max)
        self.min_text.width = 100
        self.max_text = FloatText(description='max', min=self.par.min)
        self.max_text.width = 100
        self.min_checkbox = Checkbox(description='min')
        self.max_checkbox = Checkbox(description='max')
        self.vary_checkbox = Checkbox(description='vary')

        # Set widget values and visibility.
        if par.value is not None:
            self.value_text.value = self.par.value
        min_unset = self.par.min is None or self.par.min == -np.inf
        max_unset = self.par.max is None or self.par.max == np.inf
        self.min_checkbox.value = not min_unset
        self.min_text.visible = not min_unset
        self.min_text.value = self.par.min
        self.max_checkbox.value = not max_unset
        self.max_text.visible = not max_unset
        self.max_text.value = self.par.max
        self.vary_checkbox.value = self.par.vary

        # Configure widgets to sync with par attributes.
        self.value_text.on_trait_change(self._on_value_change, 'value')
        self.min_text.on_trait_change(self._on_min_value_change, 'value')
        self.max_text.on_trait_change(self._on_max_value_change, 'value')
        self.min_checkbox.on_trait_change(self._on_min_checkbox_change,
                                          'value')
        self.max_checkbox.on_trait_change(self._on_max_checkbox_change,
                                          'value')
        self.vary_checkbox.on_trait_change(self._on_vary_change, 'value')

    def _on_value_change(self, name, value):
        self.par.value = value

    def _on_min_checkbox_change(self, name, value):
        self.min_text.visible = value
        if value:
            # -np.inf does not play well with a numerical text field,
            # so set min to -1 if activated (and back to -inf if deactivated).
            self.min_text.value = -1
            self.par.min = self.min_text.value
            self.value_text.min = self.min_text.value
        else:
            self.par.min = None

    def _on_max_checkbox_change(self, name, value):
        self.max_text.visible = value
        if value:
            # np.inf does not play well with a numerical text field,
            # so set max to 1 if activated (and back to inf if deactivated).
            self.max_text.value = 1
            self.par.max = self.max_text.value
            self.value_text.max = self.max_text.value
        else:
            self.par.max = None

    def _on_min_value_change(self, name, value):
        self.par.min = value
        self.value_text.min = value
        self.max_text.min = value

    def _on_max_value_change(self, name, value):
        self.par.max = value
        self.value_text.max = value
        self.min_text.max = value

    def _on_vary_change(self, name, value):
        self.par.vary = value
        # self.value_text.disabled = not value

    def close(self):
        # one convenience method to close (i.e., hide and disconnect) all
        # widgets in this group
        self.value_text.close()
        self.min_text.close()
        self.max_text.close()
        self.vary_checkbox.close()
        self.min_checkbox.close()
        self.max_checkbox.close()

    def _repr_html_(self):
        box = HBox()
        box.children = [self.value_text, self.vary_checkbox,
                        self.min_checkbox, self.min_text,
                        self.max_checkbox, self.max_text]
        display(box)

    # Make it easy to set the widget attributes directly.
    @property
    def value(self):
        return self.value_text.value

    @value.setter
    def value(self, value):
        self.value_text.value = value

    @property
    def vary(self):
        return self.vary_checkbox.value

    @vary.setter
    def vary(self, value):
        self.vary_checkbox.value = value

    @property
    def min(self):
        return self.min_text.value

    @min.setter
    def min(self, value):
        self.min_text.value = value

    @property
    def max(self):
        return self.max_text.value

    @max.setter
    def max(self, value):
        self.max_text.value = value

    @property
    def name(self):
       return self.par.name


class NotebookFitter(MPLFitter):
    __doc__ = _COMMON_DOC + """
    If IPython is available, it uses the IPython notebook's rich display
    to fit data interactively in a web-based GUI. The Parameters are
    represented in a web-based form that is kept in sync with `current_params`.
    All subclasses to Model, including user-defined ones, are shown in a
    drop-down menu.

    Clicking the "Fit" button updates a plot, as above, and updates the
    Parameters in the form to reflect the best fit.

    Parameters
    ----------
    data : array-like
    model : lmfit.Model
        optional initial Model to use, maybe be set or changed later
    all_models : list
        optional list of Models to populate drop-down menu, by default
        all built-in and user-defined subclasses of Model are used

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
    def __init__(self, data, model=None, all_models=None, axes_style={},
                data_style={}, init_style={}, best_style={}, **kwargs):
        # Dropdown menu of all subclasses of Model, incl. user-defined.
        self.models_menu = Dropdown()
        # Dropbox API is very different between IPy 2.x and 3.x.
        if IPY2:
            if all_models is None:
                all_models = dict([(m.__name__, m) for m in Model.__subclasses__()])
            self.models_menu.values = all_models
        else:
            if all_models is None:
                all_models = [(m.__name__, m) for m in Model.__subclasses__()]
            self.models_menu.options = all_models
        self.models_menu.on_trait_change(self._on_model_value_change,
                                             'value')
        # Button to trigger fitting.
        self.fit_button = Button(description='Fit')
        self.fit_button.on_click(self._on_fit_button_click)

        # Button to trigger guessing.
        self.guess_button = Button(description='Auto-Guess')
        self.guess_button.on_click(self._on_guess_button_click)

        # Parameter widgets are not built here. They are (re-)built when
        # the model is (re-)set.
        super(NotebookFitter, self).__init__(data, model, axes_style,
                                             data_style, init_style,
                                             best_style, **kwargs)

    def _repr_html_(self):
        display(self.models_menu)
        button_box = HBox()
        button_box.children = [self.fit_button, self.guess_button]
        display(button_box)
        for pw in self.param_widgets:
            display(pw)
        self.plot()

    def guess(self):
        guessing_successful = super(NotebookFitter, self).guess()
        self.guess_button.disabled = not guessing_successful

    def _finalize_model(self, value):
        first_run = not hasattr(self, 'param_widgets')
        if not first_run:
            # Remove all Parameter widgets, and replace them with widgets
            # for the new model.
            for pw in self.param_widgets:
                pw.close()
        self.models_menu.value = value
        self.param_widgets = [ParameterWidgetGroup(p)
                              for _, p in self._current_params.items()]
        if not first_run:
            for pw in self.param_widgets:
                display(pw)

    def _finalize_params(self):
        for pw in self.param_widgets:
            pw.value = self._current_params[pw.name].value
            pw.min = self._current_params[pw.name].min
            pw.max = self._current_params[pw.name].max
            pw.vary = self._current_params[pw.name].vary

    def plot(self):
        clear_output(wait=True)
        super(NotebookFitter, self).plot()

    def fit(self):
        super(NotebookFitter, self).fit()
        self.plot()
