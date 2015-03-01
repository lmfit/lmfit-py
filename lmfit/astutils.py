"""
utility functions for asteval

   Matthew Newville <newville@cars.uchicago.edu>,
   The University of Chicago
"""
from __future__ import division, print_function
import re
import ast
from sys import exc_info

RESERVED_WORDS = ('and', 'as', 'assert', 'break', 'class', 'continue',
                  'def', 'del', 'elif', 'else', 'except', 'exec',
                  'finally', 'for', 'from', 'global', 'if', 'import',
                  'in', 'is', 'lambda', 'not', 'or', 'pass', 'print',
                  'raise', 'return', 'try', 'while', 'with', 'True',
                  'False', 'None', 'eval', 'execfile', '__import__',
                  '__package__')

NAME_MATCH = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*$").match

UNSAFE_ATTRS = ('__subclasses__', '__bases__', '__globals__', '__code__',
                '__closure__', '__func__', '__self__', '__module__',
                '__dict__', '__class__', '__call__', '__get__',
                '__getattribute__', '__subclasshook__', '__new__',
                '__init__', 'func_globals', 'func_code', 'func_closure',
                'im_class', 'im_func', 'im_self', 'gi_code', 'gi_frame',
                '__asteval__')

# inherit these from python's __builtins__
FROM_PY = ('ArithmeticError', 'AssertionError', 'AttributeError',
           'BaseException', 'BufferError', 'BytesWarning',
           'DeprecationWarning', 'EOFError', 'EnvironmentError',
           'Exception', 'False', 'FloatingPointError', 'GeneratorExit',
           'IOError', 'ImportError', 'ImportWarning', 'IndentationError',
           'IndexError', 'KeyError', 'KeyboardInterrupt', 'LookupError',
           'MemoryError', 'NameError', 'None',
           'NotImplementedError', 'OSError', 'OverflowError',
           'ReferenceError', 'RuntimeError', 'RuntimeWarning',
           'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError',
           'SystemExit', 'True', 'TypeError', 'UnboundLocalError',
           'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError',
           'UnicodeTranslateError', 'UnicodeWarning', 'ValueError',
           'Warning', 'ZeroDivisionError', 'abs', 'all', 'any', 'bin',
           'bool', 'bytearray', 'bytes', 'chr', 'complex', 'dict', 'dir',
           'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset',
           'hash', 'hex', 'id', 'int', 'isinstance', 'len', 'list', 'map',
           'max', 'min', 'oct', 'ord', 'pow', 'range', 'repr',
           'reversed', 'round', 'set', 'slice', 'sorted', 'str', 'sum',
           'tuple', 'type', 'zip')

# inherit these from python's math
FROM_MATH = ('acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
             'ceil', 'copysign', 'cos', 'cosh', 'degrees', 'e', 'exp',
             'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum',
             'hypot', 'isinf', 'isnan', 'ldexp', 'log', 'log10', 'log1p',
             'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan',
             'tanh', 'trunc')

FROM_NUMPY = ('Inf', 'NAN', 'abs', 'add', 'alen', 'all', 'amax', 'amin',
              'angle', 'any', 'append', 'arange', 'arccos', 'arccosh',
              'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh',
              'argmax', 'argmin', 'argsort', 'argwhere', 'around', 'array',
              'array2string', 'asanyarray', 'asarray', 'asarray_chkfinite',
              'ascontiguousarray', 'asfarray', 'asfortranarray',
              'asmatrix', 'asscalar', 'atleast_1d', 'atleast_2d',
              'atleast_3d', 'average', 'bartlett', 'base_repr',
              'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor',
              'blackman', 'bool', 'broadcast', 'broadcast_arrays', 'byte',
              'c_', 'cdouble', 'ceil', 'cfloat', 'chararray', 'choose',
              'clip', 'clongdouble', 'clongfloat', 'column_stack',
              'common_type', 'complex', 'complex128', 'complex64',
              'complex_', 'complexfloating', 'compress', 'concatenate',
              'conjugate', 'convolve', 'copy', 'copysign', 'corrcoef',
              'correlate', 'cos', 'cosh', 'cov', 'cross', 'csingle',
              'cumprod', 'cumsum', 'datetime_data', 'deg2rad', 'degrees',
              'delete', 'diag', 'diag_indices', 'diag_indices_from',
              'diagflat', 'diagonal', 'diff', 'digitize', 'divide', 'dot',
              'double', 'dsplit', 'dstack', 'dtype', 'e', 'ediff1d',
              'empty', 'empty_like', 'equal', 'exp', 'exp2', 'expand_dims',
              'expm1', 'extract', 'eye', 'fabs', 'fill_diagonal', 'finfo',
              'fix', 'flatiter', 'flatnonzero', 'fliplr', 'flipud',
              'float', 'float32', 'float64', 'float_', 'floating', 'floor',
              'floor_divide', 'fmax', 'fmin', 'fmod', 'format_parser',
              'frexp', 'frombuffer', 'fromfile', 'fromfunction',
              'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'fv',
              'genfromtxt', 'getbufsize', 'geterr', 'gradient', 'greater',
              'greater_equal', 'hamming', 'hanning', 'histogram',
              'histogram2d', 'histogramdd', 'hsplit', 'hstack', 'hypot',
              'i0', 'identity', 'iinfo', 'imag', 'in1d', 'index_exp',
              'indices', 'inexact', 'inf', 'info', 'infty', 'inner',
              'insert', 'int', 'int0', 'int16', 'int32', 'int64', 'int8',
              'int_', 'int_asbuffer', 'intc', 'integer', 'interp',
              'intersect1d', 'intp', 'invert', 'ipmt', 'irr', 'iscomplex',
              'iscomplexobj', 'isfinite', 'isfortran', 'isinf', 'isnan',
              'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar',
              'issctype', 'iterable', 'ix_', 'kaiser', 'kron', 'ldexp',
              'left_shift', 'less', 'less_equal', 'linspace',
              'little_endian', 'load', 'loads', 'loadtxt', 'log', 'log10',
              'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and',
              'logical_not', 'logical_or', 'logical_xor', 'logspace',
              'long', 'longcomplex', 'longdouble', 'longfloat', 'longlong',
              'mafromtxt', 'mask_indices', 'mat', 'matrix', 'max',
              'maximum', 'maximum_sctype', 'may_share_memory', 'mean',
              'median', 'memmap', 'meshgrid', 'mgrid', 'min', 'minimum',
              'mintypecode', 'mirr', 'mod', 'modf', 'msort', 'multiply',
              'nan', 'nan_to_num', 'nanargmax', 'nanargmin', 'nanmax',
              'nanmin', 'nansum', 'ndarray', 'ndenumerate', 'ndfromtxt',
              'ndim', 'ndindex', 'negative', 'newaxis', 'nextafter',
              'nonzero', 'not_equal', 'nper', 'npv', 'number',
              'obj2sctype', 'ogrid', 'ones', 'ones_like', 'outer',
              'packbits', 'percentile', 'pi', 'piecewise', 'place', 'pmt',
              'poly', 'poly1d', 'polyadd', 'polyder', 'polydiv', 'polyfit',
              'polyint', 'polymul', 'polysub', 'polyval', 'power', 'ppmt',
              'prod', 'product', 'ptp', 'put', 'putmask', 'pv', 'r_',
              'rad2deg', 'radians', 'rank', 'rate', 'ravel', 'real',
              'real_if_close', 'reciprocal', 'record', 'remainder',
              'repeat', 'reshape', 'resize', 'restoredot', 'right_shift',
              'rint', 'roll', 'rollaxis', 'roots', 'rot90', 'round',
              'round_', 'row_stack', 's_', 'sctype2char', 'searchsorted',
              'select', 'setbufsize', 'setdiff1d', 'seterr', 'setxor1d',
              'shape', 'short', 'sign', 'signbit', 'signedinteger', 'sin',
              'sinc', 'single', 'singlecomplex', 'sinh', 'size',
              'sometrue', 'sort', 'sort_complex', 'spacing', 'split',
              'sqrt', 'square', 'squeeze', 'std', 'str', 'str_',
              'subtract', 'sum', 'swapaxes', 'take', 'tan', 'tanh',
              'tensordot', 'tile', 'trace', 'transpose', 'trapz', 'tri',
              'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros',
              'triu', 'triu_indices', 'triu_indices_from', 'true_divide',
              'trunc', 'ubyte', 'uint', 'uint0', 'uint16', 'uint32',
              'uint64', 'uint8', 'uintc', 'uintp', 'ulonglong', 'union1d',
              'unique', 'unravel_index', 'unsignedinteger', 'unwrap',
              'ushort', 'vander', 'var', 'vdot', 'vectorize', 'vsplit',
              'vstack', 'where', 'who', 'zeros', 'zeros_like')

NUMPY_RENAMES = {'ln': 'log', 'asin': 'arcsin', 'acos': 'arccos',
                 'atan': 'arctan', 'atan2': 'arctan2', 'atanh':
                 'arctanh', 'acosh': 'arccosh', 'asinh': 'arcsinh'}

def _open(filename, mode='r', buffering=0):
    """read only version of open()"""
    umode = 'r'
    if mode == 'rb':
        umode = 'rb'
    return open(filename, umode, buffering)

LOCALFUNCS = {'open': _open}

OPERATORS = {ast.Is: lambda a, b: a is b,
             ast.IsNot: lambda a, b: a is not b,
             ast.In: lambda a, b: a in b,
             ast.NotIn: lambda a, b: a not in b,
             ast.Add: lambda a, b: a + b,
             ast.BitAnd: lambda a, b: a & b,
             ast.BitOr: lambda a, b: a | b,
             ast.BitXor: lambda a, b: a ^ b,
             ast.Div: lambda a, b: a / b,
             ast.FloorDiv: lambda a, b: a // b,
             ast.LShift: lambda a, b: a << b,
             ast.RShift: lambda a, b: a >> b,
             ast.Mult: lambda a, b: a * b,
             ast.Pow: lambda a, b: a ** b,
             ast.Sub: lambda a, b: a - b,
             ast.Mod: lambda a, b: a % b,
             ast.And: lambda a, b: a and b,
             ast.Or: lambda a, b: a or b,
             ast.Eq: lambda a, b: a == b,
             ast.Gt: lambda a, b: a > b,
             ast.GtE: lambda a, b: a >= b,
             ast.Lt: lambda a, b: a < b,
             ast.LtE: lambda a, b: a <= b,
             ast.NotEq: lambda a, b: a != b,
             ast.Invert: lambda a: ~a,
             ast.Not: lambda a: not a,
             ast.UAdd: lambda a: +a,
             ast.USub: lambda a: -a}


def valid_symbol_name(name):
    """determines whether the input symbol name is a valid name

    This checks for reserved words, and that the name matches the
    regular expression ``[a-zA-Z_][a-zA-Z0-9_]``
    """
    if name in RESERVED_WORDS:
        return False
    return NAME_MATCH(name) is not None


def op2func(op):
    "return function for operator nodes"
    return OPERATORS[op.__class__]


class Empty:
    """empty class"""
    def __init__(self):
        pass

    def __nonzero__(self):
        return False

ReturnedNone = Empty()


class ExceptionHolder(object):
    "basic exception handler"
    def __init__(self, node, exc=None, msg='', expr=None, lineno=None):
        self.node = node
        self.expr = expr
        self.msg = msg
        self.exc = exc
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
        if node.__class__.__name__ == 'Name':
            if node.ctx.__class__ == ast.Load and node.id not in self.names:
                self.names.append(node.id)
        ast.NodeVisitor.generic_visit(self, node)

def get_ast_names(astnode):
    "returns symbol Names from an AST node"
    finder = NameFinder()
    finder.generic_visit(astnode)
    return finder.names
