# These variables are used at the end of the module to decide
# which BaseFitter subclass the Fitter will point to.
import warnings

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
    _ipy_msg1 = "lmfit.Fitter will use basic mode, not IPython: need IPython2."
    _ipy_msg2 = "lmfit.Fitter will use basic mode, not IPython: could not get IPython version"
    try:
        if IPython.release.version_info[0] < 2:
            warnings.warn(_ipy_msg1)
        else:
            # has_ipython = iPython installed and we are in an IPython session.
            has_ipython = IPython.get_ipython() is not None
    except Exception as e:
        warnings.warn(_ipy_msg2)

from .basefitter import BaseFitter
Fitter = BaseFitter
if has_matplotlib:
    from .basefitter import MPLFitter
    Fitter = MPLFitter

if has_ipython:
    from .ipy_fitter import NotebookFitter
    Fitter = NotebookFitter
