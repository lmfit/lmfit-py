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
    try:
        if IPython.release.version_info[0] < 2:
            warnings.warn("IPython versions before 2.0 are not supported. "
                          "Fitter will operate in basic mode, as it would "
                          "in a plain python interpreter.")
            has_ipython = False
        elif IPython.get_ipython() is None:
            has_ipython = False  # It's installed, but we are not in an IPython session.
        else:
            has_ipython = True
    except Exception as e:
        warnings.warn("An error occurred while trying to detect IPython. "
                      "Fitter will operate in basic mode. The error is: "
                      "{0}".format(e))


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

