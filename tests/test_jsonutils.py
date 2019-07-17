"""Tests for the JSON utilities."""
# coding: utf-8
# TODO: remove the line above when dropping Python 2.7
from types import BuiltinFunctionType, FunctionType

import numpy as np
import pytest
from scipy.optimize import basinhopping
import six

import lmfit
from lmfit.jsonutils import decode4js, encode4js, find_importer, import_from
from lmfit.printfuncs import alphanumeric_sort


@pytest.mark.parametrize('obj', [alphanumeric_sort, np.array, basinhopping])
def test_import_from(obj):
    """Check return value of find_importer function."""
    importer = find_importer(obj)
    assert isinstance(import_from(importer, obj.__name__),
                      (BuiltinFunctionType, FunctionType))


objects = [('test_string', six.string_types),
           (np.array([7.0]), np.ndarray),
           (np.array([1.0+2.0j]), np.ndarray),
           (123.456, np.float),
           (10, np.float),
           (u'café', six.string_types),
           (10.0-5.0j, np.complex),
           (['a', 'b', 'c'], list),
           (('a', 'b', 'c'), tuple),
           ({'a': 1.0, 'b': 2.0, 'c': 3.0}, dict),
           (lmfit.lineshapes.gaussian, FunctionType)]


@pytest.mark.parametrize('obj, obj_type', objects)
def test_encode_decode(obj, obj_type):
    """Test encoding/decoding of the various object types to/from JSON."""
    encoded = encode4js(obj)
    decoded = decode4js(encoded)

    assert decoded == obj
    assert isinstance(decoded, obj_type)


def test_encode_decode_pandas():
    """Test encoding/decoding of various pandas objects to/from JSON."""
    pytest.importorskip('pandas')
    import pandas as pd

    obj_df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                          columns=['a', 'b', 'c'])
    encoded_df = encode4js(obj_df)
    decoded_df = decode4js(encoded_df)
    assert np.all(pd.DataFrame.eq(obj_df, decoded_df))
    assert isinstance(decoded_df, pd.DataFrame)

    obj_ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    encoded_ser = encode4js(obj_ser)
    decoded_ser = decode4js(encoded_ser)
    assert np.all(pd.Series.eq(obj_ser, decoded_ser))
    assert isinstance(decoded_ser, pd.Series)
