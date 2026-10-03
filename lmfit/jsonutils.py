"""JSON utilities."""

from base64 import b64decode, b64encode
import importlib
from io import StringIO
import sys
import warnings

import dill
import numpy as np
import uncertainties

HAS_DILL = True

try:
    from pandas import DataFrame, Series, read_json
except ImportError:
    DataFrame = Series = type(NotImplemented)
    read_json = None


pyvers = f'{sys.version_info.major}.{sys.version_info.minor}'


def find_importer(obj):
    """Find importer of an object.

    This is the module that defines the object (or else the first module
    found to hold it), replaced by its topmost parent package that holds
    the same object. Public package paths are more stable than private
    modules, which libraries may reorganize.

    """
    oname = obj.__name__
    modname = getattr(obj, '__module__', None)
    if (not isinstance(modname, str) or modname.startswith('__main__')
            or getattr(sys.modules.get(modname), oname, None) is not obj):
        for modname, module in sys.modules.copy().items():
            if modname.startswith('__main__'):
                continue
            t = getattr(module, oname, None)
            if t is obj:
                break
        else:
            return None
    parts = modname.split('.')
    for i in range(1, len(parts)):
        parent = '.'.join(parts[:i])
        if getattr(sys.modules.get(parent), oname, None) is obj:
            return parent
    return modname


def import_from(modulepath, objectname):
    """Import `objectname` from the module `modulepath`.

    The module itself is imported, so a submodule that is not imported yet
    is found too, instead of failing like an attribute lookup on its parent
    package would.

    """
    return getattr(importlib.import_module(modulepath), objectname)


def encode4js(obj):
    """Prepare an object for JSON encoding.

    It has special handling for many Python types, including:
    - pandas DataFrames and Series
    - NumPy ndarrays
    - complex numbers

    """
    if isinstance(obj, DataFrame):
        return dict(__class__='PDataFrame', value=obj.to_json())
    if isinstance(obj, Series):
        return dict(__class__='PSeries', value=obj.to_json())
    if isinstance(obj, uncertainties.core.AffineScalarFunc):
        return dict(__class__='UFloat', val=obj.nominal_value, err=obj.std_dev)
    if isinstance(obj, np.ndarray):
        if 'complex' in obj.dtype.name:
            val = [(obj.real).tolist(), (obj.imag).tolist()]
        elif obj.dtype.name == 'object':
            val = [encode4js(item) for item in obj]
        else:
            val = obj.flatten().tolist()
        return dict(__class__='NDArray', __shape__=obj.shape,
                    __dtype__=obj.dtype.name, value=val)
    if isinstance(obj, (float, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (int, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, str):
        try:
            return str(obj)
        except UnicodeError:
            return obj
    if isinstance(obj, complex):
        return dict(__class__='Complex', value=(obj.real, obj.imag))
    if isinstance(obj, (tuple, list)):
        ctype = 'List'
        if isinstance(obj, tuple):
            ctype = 'Tuple'
        val = [encode4js(item) for item in obj]
        return dict(__class__=ctype, value=val)
    if isinstance(obj, dict):
        out = dict(__class__='Dict')
        for key, val in obj.items():
            out[encode4js(key)] = encode4js(val)
        return out
    if callable(obj):
        value = str(b64encode(dill.dumps(obj)), 'utf-8')
        return dict(__class__='Callable', __name__=obj.__name__,
                    pyversion=pyvers, value=value,
                    importer=find_importer(obj))
    return obj


def decode4js(obj):
    """Return decoded Python object from encoded object."""
    if not isinstance(obj, dict):
        return obj
    out = obj
    classname = obj.pop('__class__', None)
    if classname is None and isinstance(obj, dict):
        classname = 'dict'
    if classname is None:
        return obj
    if classname == 'Complex':
        out = obj['value'][0] + 1j*obj['value'][1]
    elif classname in ('List', 'Tuple'):
        out = []
        for item in obj['value']:
            out.append(decode4js(item))
        if classname == 'Tuple':
            out = tuple(out)
    elif classname == 'NDArray':
        if obj['__dtype__'].startswith('complex'):
            re = np.fromiter(obj['value'][0], dtype='double')
            im = np.fromiter(obj['value'][1], dtype='double')
            out = re + 1j*im
        elif obj['__dtype__'].startswith('object'):
            val = [decode4js(v) for v in obj['value']]
            out = np.array(val, dtype=obj['__dtype__'])
        else:
            out = np.fromiter(obj['value'], dtype=obj['__dtype__'])
        out.shape = obj['__shape__']
    elif classname == 'PDataFrame' and read_json is not None:
        out = read_json(StringIO(obj['value']))
    elif classname == 'PSeries' and read_json is not None:
        out = read_json(StringIO(obj['value']), typ='series')
    elif classname == 'UFloat':
        out = uncertainties.ufloat(obj['val'], obj['err'])
    elif classname == 'Callable':
        out = obj['__name__']
        importer = obj['importer']
        unpacked = False
        # if the callable moved to another module within its package, it
        # may still be found in a parent package: try them, nearest first
        while importer and not unpacked:
            try:
                out = import_from(importer, out)
                unpacked = True
            except (ImportError, AttributeError):
                importer = importer.rpartition('.')[0]
        if not unpacked:
            spyvers = obj.get('pyversion', '?')
            if not pyvers == spyvers:
                msg = f"Could not unpack dill-encoded callable '{out}', saved with Python version {spyvers}"
                warnings.warn(msg)

            try:
                out = dill.loads(b64decode(obj['value']))
            except RuntimeError:
                msg = f"Could not unpack dill-encoded callable '{out}`, saved with Python version {spyvers}"
                warnings.warn(msg)

    elif classname in ('Dict', 'dict'):
        out = {}
        for key, val in obj.items():
            out[key] = decode4js(val)
    return out
