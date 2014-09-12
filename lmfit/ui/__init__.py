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
    has_ipython = True
except ImportError:
    has_ipython = False
else:
    if IPython.release.version_info[0] < 2:
        warnings.warn("IPython versions before 2.0 are not supported. Fitter will operate in "
                      "basic mode, as it would in a plain python interpreter.")
        has_ipython = False


from .basefitter import BaseFitter
Fitter = BaseFitter
if has_matplotlib:
    from .basefitter import MPLFitter
    BaseFitter = BaseFitter
    Fitter = MPLFitter

if has_ipython:
    from .ipy_fitter import NotebookFitter
    BaseFitter = BaseFitter
    MPLFitter = MPLFitter
    Fitter = NotebookFitter

