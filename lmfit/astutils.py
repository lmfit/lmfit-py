"""
utility functions for asteval

   Matthew Newville <newville@cars.uchicago.edu>,
   The University of Chicago
"""
from __future__ import division, print_function
import ast
from sys import exc_info

# inherit these from python's __builtins__
FROM_PY = ('ArithmeticError', 'AssertionError', 'AttributeError',
           'BaseException', 'BufferError', 'BytesWarning',
           'DeprecationWarning', 'EOFError', 'EnvironmentError',
           'Exception', 'False', 'FloatingPointError', 'GeneratorExit',
           'IOError', 'ImportError', 'ImportWarning', 'IndentationError',
           'IndexError', 'KeyError', 'KeyboardInterrupt', 'LookupError',
           'MemoryError', 'NameError', 'None', 'NotImplemented',
           'NotImplementedError', 'OSError', 'OverflowError',
           'ReferenceError', 'RuntimeError', 'RuntimeWarning',
           'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError',
           'SystemExit', 'True', 'TypeError', 'UnboundLocalError',
           'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError',
           'UnicodeTranslateError', 'UnicodeWarning', 'ValueError',
           'Warning', 'ZeroDivisionError', 'abs', 'all', 'any', 'bin',
           'bool', 'bytearray', 'bytes', 'chr', 'complex', 'delattr',
           'dict', 'dir', 'divmod', 'enumerate', 'filter', 'float',
           'format', 'frozenset', 'getattr', 'hasattr', 'hash', 'hex',
           'id', 'int', 'isinstance', 'len', 'list', 'map', 'max', 'min',
           'oct', 'open', 'ord', 'pow', 'property', 'range', 'repr',
           'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'str',
           'sum', 'tuple', 'type', 'zip')

# inherit these from python's math
FROM_MATH = ('acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
             'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'e', 'exp',
             'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum',
             'hypot', 'isinf', 'isnan', 'ldexp', 'log', 'log10', 'log1p',
             'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan',
             'tanh', 'trunc')

# inherit these from numpy, if available
FROM_NUMPY = ('Inf', 'NAN', 'abs', 'absolute', 'add', 'alen', 'all',
              'allclose', 'alltrue', 'alterdot', 'amax', 'amin', 'angle',
              'any', 'append', 'apply_along_axis', 'apply_over_axes',
              'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan',
              'arctan2', 'arctanh', 'argmax', 'argmin', 'argsort',
              'argwhere', 'around', 'array', 'array2string', 'array_equal',
              'array_equiv', 'array_repr', 'array_split', 'array_str',
              'asanyarray', 'asarray', 'asarray_chkfinite',
              'ascontiguousarray', 'asfarray', 'asfortranarray',
              'asmatrix', 'asscalar', 'atleast_1d', 'atleast_2d',
              'atleast_3d', 'average', 'bartlett', 'base_repr', 'bench',
              'binary_repr', 'bincount', 'bitwise_and', 'bitwise_not',
              'bitwise_or', 'bitwise_xor', 'blackman', 'bmat', 'bool',
              'bool8', 'broadcast', 'broadcast_arrays', 'byte',
              'byte_bounds', 'bytes_', 'c_', 'can_cast', 'cast', 'cdouble',
              'ceil', 'cfloat', 'char', 'character', 'chararray', 'choose',
              'clip', 'clongdouble', 'clongfloat', 'column_stack',
              'common_type', 'compare_chararrays', 'compat', 'complex',
              'complex128', 'complex64', 'complex_', 'complexfloating',
              'compress', 'concatenate', 'conj', 'conjugate', 'convolve',
              'copy', 'copysign', 'core', 'corrcoef', 'correlate', 'cos',
              'cosh', 'cov', 'cross', 'csingle', 'ctypeslib', 'cumprod',
              'cumproduct', 'cumsum', 'datetime_data', 'deg2rad',
              'degrees', 'delete', 'diag', 'diag_indices',
              'diag_indices_from', 'diagflat', 'diagonal', 'diff',
              'digitize', 'disp', 'divide', 'dot', 'double', 'dsplit',
              'dstack', 'dtype', 'e', 'ediff1d', 'emath', 'empty',
              'empty_like', 'equal', 'exp', 'exp2', 'expand_dims', 'expm1',
              'extract', 'eye', 'fabs', 'fft', 'fill_diagonal',
              'find_common_type', 'finfo', 'fix', 'flatiter',
              'flatnonzero', 'flexible', 'fliplr', 'flipud', 'float',
              'float32', 'float64', 'float_', 'floating', 'floor',
              'floor_divide', 'fmax', 'fmin', 'fmod', 'format_parser',
              'frexp', 'frombuffer', 'fromfile', 'fromfunction',
              'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'fv',
              'generic', 'genfromtxt', 'getbufsize', 'geterr',
              'geterrcall', 'geterrobj', 'gradient', 'greater',
              'greater_equal', 'hamming', 'hanning', 'histogram',
              'histogram2d', 'histogramdd', 'hsplit', 'hstack', 'hypot',
              'i0', 'identity', 'iinfo', 'imag', 'in1d', 'index_exp',
              'indices', 'inexact', 'inf', 'info', 'infty', 'inner',
              'insert', 'int', 'int0', 'int16', 'int32', 'int64', 'int8',
              'int_', 'int_asbuffer', 'intc', 'integer', 'interp',
              'intersect1d', 'intp', 'invert', 'ipmt', 'irr', 'iscomplex',
              'iscomplexobj', 'isfinite', 'isfortran', 'isinf', 'isnan',
              'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar',
              'issctype', 'issubclass_', 'issubdtype', 'issubsctype',
              'iterable', 'ix_', 'kaiser', 'kron', 'ldexp', 'left_shift',
              'less', 'less_equal', 'lexsort', 'lib', 'linalg', 'linspace',
              'little_endian', 'load', 'loads', 'loadtxt', 'log', 'log10',
              'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and',
              'logical_not', 'logical_or', 'logical_xor', 'logspace',
              'long', 'longcomplex', 'longdouble', 'longfloat', 'longlong',
              'lookfor', 'ma', 'mafromtxt', 'mask_indices', 'mat', 'math',
              'matrix', 'matrixlib', 'max', 'maximum', 'maximum_sctype',
              'may_share_memory', 'mean', 'median', 'memmap', 'meshgrid',
              'mgrid', 'min', 'minimum', 'mintypecode', 'mirr', 'mod',
              'modf', 'msort', 'multiply', 'nan', 'nan_to_num',
              'nanargmax', 'nanargmin', 'nanmax', 'nanmin', 'nansum',
              'nbytes', 'ndarray', 'ndenumerate', 'ndfromtxt', 'ndim',
              'ndindex', 'negative', 'newaxis', 'nextafter', 'nonzero',
              'not_equal', 'nper', 'npv', 'number', 'obj2sctype', 'object',
              'object0', 'object_', 'ogrid', 'ones', 'ones_like', 'outer',
              'packbits', 'percentile', 'pi', 'piecewise', 'pkgload',
              'place', 'pmt', 'poly', 'poly1d', 'polyadd', 'polyder',
              'polydiv', 'polyfit', 'polyint', 'polymul', 'polynomial',
              'polysub', 'polyval', 'power', 'ppmt', 'prod', 'product',
              'ptp', 'put', 'putmask', 'pv', 'r_', 'rad2deg', 'radians',
              'random', 'rank', 'rate', 'ravel', 'real', 'real_if_close',
              'rec', 'recarray', 'recfromcsv', 'recfromtxt', 'reciprocal',
              'record', 'remainder', 'repeat', 'require', 'reshape',
              'resize', 'restoredot', 'right_shift', 'rint', 'roll',
              'rollaxis', 'roots', 'rot90', 'round', 'round_', 'row_stack',
              's_', 'safe_eval', 'save', 'savetxt', 'savez', 'sctype2char',
              'sctypeDict', 'sctypeNA', 'sctypes', 'searchsorted',
              'select', 'setbufsize', 'setdiff1d', 'seterr', 'setxor1d',
              'shape', 'short', 'sign', 'signbit', 'signedinteger', 'sin',
              'sinc', 'single', 'singlecomplex', 'sinh', 'size',
              'sometrue', 'sort', 'sort_complex', 'source', 'spacing',
              'split', 'sqrt', 'square', 'squeeze', 'std', 'str', 'str_',
              'subtract', 'sum', 'swapaxes', 'take', 'tan', 'tanh',
              'tensordot', 'test', 'testing', 'tile', 'trace', 'transpose',
              'trapz', 'tri', 'tril', 'tril_indices', 'tril_indices_from',
              'trim_zeros', 'triu', 'triu_indices', 'triu_indices_from',
              'true_divide', 'trunc', 'typeDict', 'typeNA', 'typecodes',
              'typename', 'ubyte', 'ufunc', 'uint', 'uint0', 'uint16',
              'uint32', 'uint64', 'uint8', 'uintc', 'uintp', 'ulonglong',
              'union1d', 'unique', 'unravel_index', 'unsignedinteger',
              'unwrap', 'ushort', 'vander', 'var', 'vdot', 'vectorize',
              'version', 'void', 'void0', 'vsplit', 'vstack', 'where',
              'who', 'zeros', 'zeros_like')

NUMPY_RENAMES = {'ln':'log', 'asin':'arcsin', 'acos':'arccos',
                 'atan':'arctan', 'atan2':'arctan2', 'atanh':'arctanh',
                 'acosh':'arccosh', 'asinh':'arcsinh'}

class ExceptionHolder(object):
    "basic exception handler"
    def __init__(self, node, exc=None, msg='', expr=None, lineno=None):
        self.node   = node
        self.expr   = expr
        self.msg    = msg
        self.exc    = exc
        self.lineno = lineno
        self.exc_info = exc_info()
        if self.exc is None and self.exc_info[0] is not None:
            self.exc = self.exc_info[0]
        if self.msg is '' and self.exc_info[1] is not None:
            self.msg = self.exc_info[1]

    def get_error(self):
        "retrieve error data"
        col_offset = -1
        if self.node is not None:
            try:
                col_offset = self.node.col_offset
            except AttributeError:
                pass
        try:
            exc_name = self.exc.__name__
        except AttributeError:
            exc_name = str(self.exc)
        if exc_name in (None, 'None'):
            exc_name = 'UnknownError'

        out = ["   %s" % self.expr]
        if col_offset > 0:
            out.append("    %s^^^" % ((col_offset)*' '))
        out.append(str(self.msg))
        return (exc_name, '\n'.join(out))

class NameFinder(ast.NodeVisitor):
    """find all symbol names used by a parsed node"""
    def __init__(self):
        self.names = []
        ast.NodeVisitor.__init__(self)

    def generic_visit(self, node):
        nodename = node.__class__.__name__.lower()
        if nodename == 'name':
            if (node.ctx.__class__ == ast.Load and
                node.id not in self.names):
                self.names.append(node.id)
        ast.NodeVisitor.generic_visit(self, node)
