"""
Safe(ish) evaluator of python expressions, using ast module.
Emphasis is on mathematical expressions, and so uses numpy
functions if available.

Symbols are held in the Interpreter symtable -- a simple
dictionary supporting a simple, flat namespace.

Expressions can be compiled into ast node and then evaluated
later, using the current values in the symtable.

   Copyright (c) 2011 Matthew Newville, The University of Chicago
   <newville@cars.uchicago.edu>

  Permission to use and redistribute the source code or binary forms of this
  software and its documentation, with or without modification is hereby
  granted provided that the above notice of copyright, these terms of use,
  and the disclaimer of warranty below appear in the source code and
  documentation, and that none of the names of The University of Chicago or
  the authors appear in advertising or endorsement of works derived from this
  software without specific prior written permission from all parties.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THIS SOFTWARE.
"""

from __future__ import division, print_function
import os
import sys
import ast
import math

from .astutils import (FROM_PY, FROM_MATH, FROM_NUMPY,
                      NUMPY_RENAMES, op2func, ExceptionHolder)
try:
    import numpy
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

__version__ = '0.1'

class Procedure(object):
    """Procedure: user-defined function for asteval

    This stores the compiled ast nodes as from the
    'functiondef' ast node for later evaluation.
    """
    def __init__(self, name, interp, doc=None, lineno=0,
                 body=None, args=None, kwargs=None,
                 vararg=None, varkws=None):
        self.name = name
        self.interpreter = interp
        self.raise_exc = self.interpreter.raise_exception
        self.__doc__ = doc
        self.body = body
        self.argnames = args
        self.kwargs = kwargs
        self.vararg = vararg
        self.varkws = varkws
        self.lineno = lineno

    def __repr__(self):
        sig = ""
        if len(self.argnames) > 0:
            sig = "%s%s" % (sig, ', '.join(self.argnames))
        if self.vararg is not None:
            sig = "%s, *%s" % (sig, self.vararg)
        if len(self.kwargs) > 0:
            if len(sig) > 0:
                sig = "%s, " % sig
            _kw = ["%s=%s" % (k, v) for k, v in self.kwargs]
            sig = "%s%s" % (sig, ', '.join(_kw))

        if self.varkws is not None:
            sig = "%s, **%s" % (sig, self.varkws)
        sig = "<Procedure %s(%s)>" % (self.name, sig)
        if self.__doc__ is not None:
            sig = "%s\n  %s" % (sig, self.__doc__)
        return sig

    def __call__(self, *args, **kwargs):
        symtable  = self.interpreter.symtable
        symlocals = {}
        args = list(args)
        n_args = len(args)
        n_names = len(self.argnames)

        if n_args != n_names:
            msg = None
            if n_args < n_names:
                msg = 'not enough arguments for Procedure %s' % self.name
                msg = '%s (expected %i, got %i)'% (msg, n_names, n_args)
                self.raise_exc(None, msg=msg, expr='<>',
                               lineno=self.lineno)

            msg = "too many arguments for Procedure %s" % self.name

        for argname in self.argnames:
            symlocals[argname] = args.pop(0)

        if len(args) > 0 and self.kwargs is not None:
            msg = "got multiple values for keyword argument '%s' Procedure %s"
            for t_a, t_kw in zip(args, self.kwargs):
                if t_kw[0] in kwargs:
                    msg = msg % (t_kw[0], self.name)
                    self.raise_exc(None, msg=msg, expr='<>',
                                   lineno=self.lineno)
                else:
                    kwargs[t_a] = t_kw[1]

        try:
            if self.vararg is not None:
                symlocals[self.vararg] = tuple(args)

            for key, val in self.kwargs:
                if key in kwargs:
                    val = kwargs.pop(key)
                symlocals[key] = val

            if self.varkws is not None:
                symlocals[self.varkws] = kwargs

            elif len(kwargs) > 0:
                msg = 'extra keyword arguments for Procedure %s (%s)'
                msg = msg % (self.name, ','.join(list(kwargs.keys())))
                self.raise_exc(None, msg=msg, expr='<>',
                               lineno=self.lineno)

        except (ValueError, LookupError, TypeError,
                NameError, AttributeError):
            msg = 'incorrect arguments for Procedure %s' % self.name
            self.raise_exc(None, msg=msg, expr='<>',
                           lineno=self.lineno)

        save_symtable = self.interpreter.symtable.copy()
        self.interpreter.symtable.update(symlocals)
        self.interpreter.retval = None
        retval = None

        # evaluate script of function
        for node in self.body:
            self.interpreter.interp(node, expr='<>', lineno=self.lineno)
            if len(self.interpreter.error) > 0:
                break
            if self.interpreter.retval is not None:
                retval = self.interpreter.retval
                break

        self.interpreter.symtable = save_symtable
        symlocals = None
        return retval


class NameFinder(ast.NodeVisitor):
    """find all symbol names used by a parsed node"""
    def __init__(self):
        self.names = []

    def generic_visit(self, node):
        nodename = node.__class__.__name__.lower()
        if nodename == 'name':
            if (node.ctx.__class__ == ast.Load and
                node.id not in self.names):
                self.names.append(node.id)
        ast.NodeVisitor.generic_visit(self, node)


class Interpreter:
    """mathematical expression compiler and interpreter.

  This module compiles expressions and statements to AST representation,
  using python's ast module, and then executes the AST representation
  using a dictionary of named object (variable, functions).

  This then gives a restricted version of Python, being a procedural
  language (though working on Python objects) with a simplified, flat
  namespace (this is overcome in related implementaions). The program
  syntax here is expected to be valid Python.

  The following Python syntax elements are not supported:
      Import, Exec, Lambda, Class, Global, Generators,
      Yield, Decorators,  Else and Finally for Try-Except

  Many parts of Python syntax are supported, including:
     advanced slicing:    a[::-1], array[-3:, :, ::2]
     if-expressions:      out = one_thing if TEST else other
     list comprehension
     for-loops, while-loops
     if-then-elif-else conditionals
     try-except (but not the 'else' or 'finally' variants ...)
     function definitions with def

  """

    supported_nodes = ('assert', 'assign', 'attribute', 'augassign',
                       'binop', 'boolop', 'break', 'call', 'compare',
                       'continue', 'delete', 'dict', 'ellipsis',
                       'excepthandler', 'expr', 'expression', 'extslice',
                       'for', 'functiondef', 'if', 'ifexp', 'index',
                       'interrupt', 'list', 'listcomp', 'module', 'name',
                       'num', 'pass', 'print', 'raise', 'repr', 'return',
                       'slice', 'str', 'subscript', 'tryexcept', 'tuple',
                       'unaryop', 'while')

    def __init__(self, symtable=None, writer=None):
        self.writer = writer or sys.stdout

        if symtable is None:
            symtable = {}
        self.symtable   = symtable
        self._interrupt = None
        self.error      = []
        self.expr       = None
        self.retval     = None
        self.lineno    = 0

        for sym in FROM_PY:
            if sym in __builtins__:
                symtable[sym] = __builtins__[sym]
        for sym in FROM_MATH:
            if hasattr(math, sym):
                symtable[sym] = getattr(math, sym)

        if HAS_NUMPY:
            for sym in FROM_NUMPY:
                if hasattr(numpy, sym):
                    symtable[sym] = getattr(numpy, sym)
            for name, sym in NUMPY_RENAMES.items():
                if hasattr(numpy, sym):
                    symtable[name] = getattr(numpy, sym)

        self.node_handlers = {}
        for tnode in self.supported_nodes:
            self.node_handlers[tnode] = getattr(self, "on_%s" % tnode)

    def unimplemented(self, node):
        "unimplemented nodes"
        self.raise_exception(node,
                             "'%s' not supported" % (node.__class__.__name__))

    def raise_exception(self, node, msg='', expr=None):
        "add an exception"
        if self.error is None:
            self.error = []
        if expr  is None:
            expr  = self.expr
        if len(self.error) > 0 and not isinstance(node, ast.Module):
            msg = '%s' % msg

        etype, evalue, tback = sys.exc_info()
        err = ExceptionHolder(node, msg=msg, expr= expr,
                              py_exc=(etype, evalue))
        self._interrupt = ast.Break()
        self.error.append(err)

    # main entry point for Ast node evaluation
    #  compile:  text of statement -> ast
    #  interp :  ast -> result
    #  eval   :  string statement -> result = interp(compile(statement))
    def compile(self, text, lineno=-4):
        """compile statement/expression to Ast representation    """
        self.expr  = text
        try:
            return ast.parse(text)
        except:
            self.raise_exception(None, msg='Syntax Error', expr=text)

    def interp(self, node, expr=None, lineno=None):
        """executes compiled Ast representation for an expression"""
        # Note: keep the 'node is None' test: internal code here may run
        #    interp(None) and expect a None in return.
        if node is None:
            return None
        if isinstance(node, str):
            node = self.compile(node)
        if lineno is not None:
            self.lineno = lineno

        if expr   is not None:
            self.expr   = expr

        # get handler for this node:
        #   on_xxx with handle nodes of type 'xxx', etc
        try:
            handler = self.node_handlers[node.__class__.__name__.lower()]
        except KeyError:
            return self.unimplemented(node)

        # run the handler:  this will likely generate
        # recursive calls into this interp method.
        try:
            ret = handler(node)
            if isinstance(ret, enumerate):
                ret = list(ret)
            return ret

        except:
            self.raise_exception(node, msg='Runtime Error', expr=expr)

    def __call__(self, expr, **kw):
        return self.eval(expr, **kw)

    def eval(self, expr, lineno=0, show_errors=True):
        """evaluates a single statement"""
        self.lineno = lineno
        self.error = []

        node = self.compile(expr, lineno=lineno)
        out = None
        if len(self.error) > 0:
            self.raise_exception(node, msg='Syntax Error', expr=expr)

        else:
            out = self.interp(node, expr=expr, lineno=lineno)

        if show_errors and len(self.error) > 0:
            for err in self.error:
                msg = err.get_error()
                print(*msg, file=self.writer)
        return out

    def dump(self, node, **kw):
        "simple ast dumper"
        return ast.dump(node, **kw)

    # handlers for ast components
    def on_expr(self, node):
        "expression"
        return self.interp(node.value)  # ('value',)

    def on_index(self, node):
        "index"
        return self.interp(node.value)  # ('value',)

    def on_return(self, node): # ('value',)
        "return statement"
        self.retval = self.interp(node.value)
        return

    def on_repr(self, node):
        "repr "
        return repr(self.interp(node.value))  # ('value',)

    def on_module(self, node):    # ():('body',)
        "module def"
        out = None
        for tnode in node.body:
            out = self.interp(tnode)
        return out

    def on_expression(self, node):
        "basic expression"
        return self.on_module(node) # ():('body',)

    def on_pass(self, node):
        "pass statement"
        return None  # ()

    def on_ellipsis(self, node):
        "ellipses"
        return Ellipsis

    # for break and continue: set the instance variable _interrupt
    def on_interrupt(self, node):    # ()
        "interrupt handler"
        self._interrupt = node
        return node

    def on_break(self, node):
        "break"
        return self.on_interrupt(node)

    def on_continue(self, node):
        "continue"
        return self.on_interrupt(node)

    def on_assert(self, node):    # ('test', 'msg')
        "assert statement"
        if not self.interp(node.test):
            raise AssertionError(self.interp(node.msg()))
        return True

    def on_list(self, node):    # ('elt', 'ctx')
        "list"
        return [self.interp(e) for e in node.elts]

    def on_tuple(self, node):    # ('elts', 'ctx')
        "tuple"
        return tuple(self.on_list(node))

    def on_dict(self, node):    # ('keys', 'values')
        "dictionary"
        nodevals = list(zip(node.keys, node.values))
        interp = self.interp
        return dict([(interp(k), interp(v)) for k, v in nodevals])

    def on_num(self, node):
        'return number'
        return node.n  # ('n',)

    def on_str(self, node):
        'return string'
        return node.s  # ('s',)

    def on_name(self, node):    # ('id', 'ctx')
        """ Name node """
        ctx = node.ctx.__class__
        if ctx == ast.Del:
            val = node.id # can't delete here??
        elif ctx == ast.Param:  # for Function Def
            val = str(node.id)
        else:
            try:
                val = self.symtable[node.id]
            except KeyError:
                errmsg = "cannot find %s" % node.id
                self.raise_exception(node, errmsg)
        return val

    def node_assign(self, node, val):
        """here we assign a value (not the node.value object) to a node
        this is used by on_assign, but also by for, list comprehension, etc.
        """
        if len(self.error) > 0:
            return
        if node.__class__ == ast.Name:
            sym = self.symtable[node.id] = val
        elif node.__class__ == ast.Attribute:
            if node.ctx.__class__  == ast.Load:
                errmsg = "cannot assign to attribute %s" % node.attr
                self.raise_exception(node, errmsg)

            setattr(self.interp(node.value), node.attr, val)

        elif node.__class__ == ast.Subscript:
            sym    = self.interp(node.value)
            xslice = self.interp(node.slice)
            if isinstance(node.slice, ast.Index):
                sym.__setitem__(xslice, val)
            elif isinstance(node.slice, ast.Slice):
                sym.__setslice__(xslice.start, xslice.stop, val)
            elif isinstance(node.slice, ast.ExtSlice):
                sym[(xslice)] = val
        elif node.__class__ in (ast.Tuple, ast.List):
            if len(val) == len(node.elts):
                for telem, tval in zip(node.elts, val):
                    self.node_assign(telem, tval)
            else:
                raise ValueError('too many values to unpack')

    def on_attribute(self, node):    # ('value', 'attr', 'ctx')
        "extract attribute"
        ctx = node.ctx.__class__
        if ctx == ast.Load:
            sym = self.interp(node.value)
            if hasattr(sym, node.attr):
                return getattr(sym, node.attr)
            else:
                obj = self.interp(node.value)
                fmt = "%s does not have attribute '%s'"
                msg = fmt % (obj, node.attr)
                self.raise_exception(node, msg=msg)

        elif ctx == ast.Del:
            return delattr(sym, node.attr)
        elif ctx == ast.Store:
            msg = "attribute for storage: shouldn't be here!"
            self.raise_exception(node, msg=msg)

    def on_assign(self, node):    # ('targets', 'value')
        "simple assignment"
        val = self.interp(node.value)
        if len(self.error) > 0:
            return
        for tnode in node.targets:
            self.node_assign(tnode, val)
        return # return val

    def on_augassign(self, node):    # ('target', 'op', 'value')
        "augmented assign"
        return self.on_assign(ast.Assign(targets=[node.target],
                                         value=ast.BinOp(left = node.target,
                                                         op = node.op,
                                                         right= node.value)))

    def on_slice(self, node):    # ():('lower', 'upper', 'step')
        "simple slice"
        return slice(self.interp(node.lower), self.interp(node.upper),
                     self.interp(node.step))

    def on_extslice(self, node):    # ():('dims',)
        "extended slice"
        return tuple([self.interp(tnode) for tnode in node.dims])

    def on_subscript(self, node):    # ('value', 'slice', 'ctx')
        "subscript handling -- one of the tricky parts"
        val    = self.interp(node.value)
        nslice = self.interp(node.slice)
        ctx = node.ctx.__class__
        if ctx in ( ast.Load, ast.Store):
            if isinstance(node.slice, (ast.Index, ast.Slice, ast.Ellipsis)):
                return val.__getitem__(nslice)
            elif isinstance(node.slice, ast.ExtSlice):
                return val[(nslice)]
        else:
            msg = "subscript with unknown context"
            self.raise_exception(node, msg=msg)

    def on_delete(self, node):    # ('targets',)
        "delete statement"
        for tnode in node.targets:
            if tnode.ctx.__class__ != ast.Del:
                break
            children = []
            while tnode.__class__ == ast.Attribute:
                children.append(tnode.attr)
                tnode = tnode.value

            if tnode.__class__ == ast.Name:
                children.append(tnode.id)
                children.reverse()
                self.symtable.pop('.'.join(children))
            else:
                msg = "could not delete symbol"
                self.raise_exception(node, msg=msg)

    def on_unaryop(self, node):    # ('op', 'operand')
        "unary operator"
        return op2func(node.op)(self.interp(node.operand))

    def on_binop(self, node):    # ('left', 'op', 'right')
        "binary operator"
        return op2func(node.op)(self.interp(node.left),
                                self.interp(node.right))

    def on_boolop(self, node):    # ('op', 'values')
        "boolean operator"
        val = self.interp(node.values[0])
        is_and = ast.Or != node.op.__class__
        if (is_and and val) or (not is_and and not val):
            for n in node.values:
                val = op2func(node.op)(val, self.interp(n))
                if (is_and and not val) or (not is_and and val):
                    break
        return val

    def on_compare(self, node):    # ('left', 'ops', 'comparators')
        "comparison operators"
        lval = self.interp(node.left)
        out  = True
        for op, rnode in zip(node.ops, node.comparators):
            rval = self.interp(rnode)
            out  = out and op2func(op)(lval, rval)
            lval = rval
            if not out:
                break
        return out

    def on_print(self, node):    # ('dest', 'values', 'nl')
        """ note: implements Python2 style print statement, not
        print() function.  May need improvement...."""
        dest = self.interp(node.dest) or self.writer
        end = ''
        if node.nl:
            end = '\n'
        out = [self.interp(tnode) for tnode in node.values]
        if out and len(self.error)==0:
            print(*out, file=dest, end=end)

    def on_if(self, node):    # ('test', 'body', 'orelse')
        "regular if-then-else statement"
        block = node.orelse
        if self.interp(node.test):
            block = node.body
        for tnode in block:
            self.interp(tnode)

    def on_ifexp(self, node):    # ('test', 'body', 'orelse')
        "if expressions"
        expr = node.orelse
        if self.interp(node.test):
            expr = node.body
        return self.interp(expr)

    def on_while(self, node):    # ('test', 'body', 'orelse')
        "while blocks"
        while self.interp(node.test):
            self._interrupt = None
            for tnode in node.body:
                self.interp(tnode)
                if self._interrupt is not None:
                    break
            if isinstance(self._interrupt, ast.Break):
                break
        else:
            for tnode in node.orelse:
                self.interp(tnode)
        self._interrupt = None

    def on_for(self, node):    # ('target', 'iter', 'body', 'orelse')
        "for blocks"
        for val in self.interp(node.iter):
            self.node_assign(node.target, val)
            if len(self.error) > 0:
                return
            self._interrupt = None
            for tnode in node.body:
                self.interp(tnode)
                if len(self.error) > 0:
                    return
                if self._interrupt is not None:
                    break
            if isinstance(self._interrupt, ast.Break):
                break
        else:
            for tnode in node.orelse:
                self.interp(tnode)
        self._interrupt = None

    def on_listcomp(self, node):    # ('elt', 'generators')
        "list comprehension"
        out = []
        for tnode in node.generators:
            if tnode.__class__ == ast.comprehension:
                for val in self.interp(tnode.iter):
                    self.node_assign(tnode.target, val)
                    if len(self.error) > 0:
                        return
                    add = True
                    for cond in tnode.ifs:
                        add = add and self.interp(cond)
                    if add:
                        out.append(self.interp(node.elt))
        return out

    def on_excepthandler(self, node): # ('type', 'name', 'body')
        "exception handler..."
        return (self.interp(node.type), node.name, node.body)

    def on_tryexcept(self, node):    # ('body', 'handlers', 'orelse')
        "try/except blocks"
        for tnode in node.body:
            self.interp(tnode)
            if self.error:
                e_type, e_value = self.error[-1].py_exc
                for hnd in node.handlers:
                    htype = None
                    if hnd.type is not None:
                        htype = __builtins__.get(hnd.type.id, None)
                    if htype is None or isinstance(e_type(), htype):
                        self.error = []
                        if hnd.name is not None:
                            self.node_assign(hnd.name, e_value)
                        for tline in hnd.body:
                            self.interp(tline)
                        break

    def on_raise(self, node):    # ('type', 'inst', 'tback')
        "raise statement"
        msg = "%s: %s" % (self.interp(node.type).__name__,
                          self.interp(node.inst))
        self.raise_exception(node.type, msg=msg)

    def on_call(self, node):
        "function/procedure execution"
        # ('func', 'args', 'keywords', 'starargs', 'kwargs')
        func = self.interp(node.func)
        if not hasattr(func, '__call__'):
            msg = "'%s' is not callable!!" % (func)
            self.raise_exception(node, msg=msg)

        args = [self.interp(targ) for targ in node.args]
        if node.starargs is not None:
            args = args + self.interp(node.starargs)

        keywords = {}
        for key in node.keywords:
            if not isinstance(key, ast.keyword):
                msg = "keyword error in function call '%s'" % (func)
                self.raise_exception(node, msg=msg)

            keywords[key.arg] = self.interp(key.value)
        if node.kwargs is not None:
            keywords.update(self.interp(node.kwargs))
        return func(*args, **keywords)

    def on_functiondef(self, node):
        "define a function"
        # ('name', 'args', 'body', 'decorator_list')
        if node.decorator_list != []:
            print("Warning: decorated procedures not supported!")

        kwargs = []
        while node.args.defaults:
            defval = self.interp(node.args.defaults.pop())
            key    = self.interp(node.args.args.pop())
            kwargs.append((key, defval))
        kwargs.reverse()
        args = [tnode.id for tnode in node.args.args]
        doc = None
        if (isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Str)):
            docnode = node.body.pop(0)
            doc = docnode.value.s

        self.symtable[node.name] = Procedure(node.name, self, doc=doc,
                                             body = node.body,
                                             lineno = self.lineno,
                                             args = args,
                                             kwargs = kwargs,
                                             vararg = node.args.vararg,
                                             varkws = node.args.kwarg)

