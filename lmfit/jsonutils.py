#!/usr/bin/env python
"""JSON utilities for larch objects."""
from base64 import b64decode, b64encode
import json
import sys

import numpy as np
import six

try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

try:
    from pandas import DataFrame, Series, read_json
except ImportError:
    DataFrame = Series = type(NotImplemented)
    read_json = None


def bindecode(val):
    """b64decode wrapper, Python 2 and 3 version."""
    return b64decode(six.b(val))


if six.PY3:
    def binencode(val):
        """b64encode wrapper, Python 3 version."""
        return str(b64encode(val), 'utf-8')  # b64encode results is /always/ UTF-8
else:
    def binencode(val):
        """b64encode wrapper, Python 2 version."""
        return str(b64encode(val))


def encode4js(obj):
    """Prepare an object for json encoding.

    It has special handling for many Python types, including:
    - pandas dataframes and series
    - numpy ndarrays
    - complex numbers

    """
    if isinstance(obj, DataFrame):
        return dict(__class__='PDataFrame', value=json.loads(obj.to_json()))
    elif isinstance(obj, DataFrame):
        return dict(__class__='PSeries', value=encode4js(obj.to_dict()))
    elif isinstance(obj, np.ndarray):
        if 'complex' in obj.dtype.name:
            val = [(obj.real).tolist(), (obj.imag).tolist()]
        elif obj.dtype.name == 'object':
            val = [encode4js(item) for item in obj['value']]
        else:
            val = obj.flatten().tolist()
        return dict(__class__='NDArray', __shape__=obj.shape,
                    __dtype__=obj.dtype.name, value=val)
    elif isinstance(obj, (np.float, np.int)):
        return float(obj)
    elif isinstance(obj, six.string_types):
        try:
            return str(obj)
        except UnicodeError:
            return obj
    elif isinstance(obj, np.complex):
        return dict(__class__='Complex', value=(obj.real, obj.imag))
    elif isinstance(obj, (tuple, list)):
        ctype = 'List'
        if isinstance(obj, tuple):
            ctype = 'Tuple'
        val = [encode4js(item) for item in obj]
        return dict(__class__=ctype, value=val)
    elif isinstance(obj, dict):
        out = dict(__class__='Dict')
        for key, val in obj.items():
            out[encode4js(key)] = encode4js(val)
        return out
    elif callable(obj):
        val = None
        pyvers = "%d.%d" % (sys.version_info.major,
                            sys.version_info.minor)
        if HAS_DILL:
            val = binencode(dill.dumps(obj))
        return dict(__class__='Callable', __name__=obj.__name__,
                    pyversion=pyvers, value=val)
    return obj


def decode4js(obj):
    """Return decoded Python object from encoded object."""
    if not isinstance(obj, dict):
        return obj
    out = obj
    classname = obj.pop('__class__', None)
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
        out = read_json(json.dumps(obj['value']))
    elif classname == 'PSeries':
        out = Series(obj['value'])
    elif classname == 'Callable':
        out = val = obj['__name__']
        pyvers = "%d.%d" % (sys.version_info.major,
                            sys.version_info.minor)
        if pyvers == obj['pyversion'] and HAS_DILL:
            out = dill.loads(bindecode(obj['value']))

    elif classname in ('Dict', 'dict'):
        out = {}
        for key, val in obj.items():
            out[key] = decode4js(val)
    return out
