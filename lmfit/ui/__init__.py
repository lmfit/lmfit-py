# These variables are used at the end of the module to decide
# which BaseFitter subclass the Fitter will point to.
try:
    import matplotlib
except ImportError:
    has_matplotlib = False
else:
    has_matplotlib = True
try:
    import IPython
except ImportError:
    has_ipython = False
else:
    has_ipython = IPython.get_ipython() is not None
    if IPython.release.version_info[0] < 2:
        warnings.warn("IPython versions before 2.0 are not supported. Fitter will operate in "
                      "basic mode, as it would in a plain python interpreter.")
        has_ipython = False

if has_ipython:
    from .ipy_fitter import NotebookFitter
    from ..fitter import BaseFitter, MPLFitter
    Fitter = NotebookFitter
elif has_matplotlib:
    from ..fitter import BaseFitter, MPLFitter
    Fitter = MPLFitter
else:
    # no dependencies beyond core lmfit dependencies
    from ..fitter import BaseFitter
    Fitter = BaseFitter
