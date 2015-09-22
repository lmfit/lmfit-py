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
    warnings.warn("lmfit.Fitter will use basic mode, not IPython: need matplotlib")
else:
    _ipy_msg1 = "lmfit.Fitter will use basic mode, not IPython: need IPython2."
    _ipy_msg2 = "lmfit.Fitter will use basic mode, not IPython: could not get IPython version"
    _ipy_msg3 = "lmfit.Fitter will use basic mode, not IPython: need ipywidgets."
    try:
        major_version = IPython.release.version_info[0]
        if major_version < 2:
            warnings.warn(_ipy_msg1)
        elif major_version > 3:
            # After IPython 3, widgets were moved to a separate package.
            # There is a shim to allow the old import, but the package has to be
            # installed for that to work.
            try:
                import ipywidgets
            except ImportError:
                warnings.warn(_ipy_msg3)
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
