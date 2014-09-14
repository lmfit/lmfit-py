# These variables are used at the end of the module to decide
# which BaseFitter subclass the Fitter will point to.

has_ipython, has_matplotlib = False, False

try:
    import matplotlib
except ImportError:
    pass
else:
    has_matplotlib = True

try:
    import IPython
except ImportError:
    pass
else:
    try:
        if IPython.release.version_info[0] < 2:
            warnings.warn("IPython versions before 2.0 are not supported. "
                          "Fitter will operate in basic mode, as it would "
                          "in a plain python interpreter.")
        else:
            # has_ipython = iPython installed and we are in an IPython session.
            has_ipython = IPython.get_ipython() is not None
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

