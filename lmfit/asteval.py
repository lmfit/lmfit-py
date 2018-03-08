"""
Safe(ish) evaluator of python expressions, using ast module.
The emphasis here is on mathematical expressions, and so
numpy functions are imported if available and used.

Symbols are held in the Interpreter symtable -- a simple
dictionary supporting a simple, flat namespace.

Expressions can be compiled into ast node and then evaluated
later, using the current values in the symboltable.
"""

from __future__ import division, print_function

import ast
import six
from sys import exc_info, stdout, stderr, version_info

from .astutils import (UNSAFE_ATTRS, HAS_NUMPY, make_symbol_table, ndarray,
                       op2func, ExceptionHolder, ReturnedNone,
                       valid_symbol_name)

builtins = __builtins__
if not isinstance(builtins, dict):
    builtins = builtins.__dict__

ALL_NODES = ['arg', 'assert', 'assign', 'attribute', 'augassign', 'binop',
             'boolop', 'break', 'call', 'compare', 'continue', 'delete',
             'dict', 'ellipsis', 'excepthandler', 'expr', 'extslice',
             'for', 'functiondef', 'if', 'ifexp', 'index', 'interrupt',
             'list', 'listcomp', 'module', 'name', 'nameconstant', 'num',
             'pass', 'print', 'raise', 'repr', 'return', 'slice', 'str',
             'subscript', 'try', 'tuple', 'unaryop', 'while']

# noinspection PyIncorrectDocstring
class Interpreter(object):
    """Mathematical expression compiler and interpreter.

    This module compiles expressions and statements to AST representation,
    using python's ast module, and then executes the AST representation
    using a dictionary of named object (variable, functions).

    The result is a restricted, simplified version of Python meant for
    numerical caclulations that is somewhat safer than 'eval' because
    some operations (such as 'import' and 'eval') are simply not allowed.
    The resulting language uses a flat namespace that works on Python
    objects, but does not allow new classes to be defined.

    Many parts of Python syntax are supported, including:
        for loops, while loops, if-then-elif-else conditionals
        try-except (including 'finally')
        function definitions with def
        advanced slicing:    a[::-1], array[-3:, :, ::2]
        if-expressions:      out = one_thing if TEST else other
        list comprehension   out = [sqrt(i) for i in values]

    The following Python syntax elements are not supported:
        Import, Exec, Lambda, Class, Global, Generators,
        Yield, Decorators

    In addition, while many builtin functions are supported, several
    builtin functions are missing ('eval', 'exec', and 'getattr' for
    example) that can be considered unsafe.

    If numpy is installed, many numpy functions are also imported.
    """
    def __init__(self, symtable=None, usersyms=None, writer=None,
                 err_writer=None, use_numpy=True, minimal=False,
                 no_if=False, no_for=False, no_while=False,
                 no_try=False, no_functiondef=False, no_ifexp=False,
                 no_listcomp=False, no_augassign=False,
                 no_assert=False, no_delete=False, no_raise=False,
                 no_print=False, max_time=30):

        """create an asteval Interpreter: a restricted, simplified interpreter
        of mathematical expressions using Python syntax.

        Parameters
        ----------
        symtable : dict or `None`
            dictionary to use as symbol table (if `None`, one will be created).
        usersyms : dict or `None`
            dictionary of user-defined symbols to add to symbol table.
        writer : file-like or `None`
            callable file-like object where standard output will be sent.
        err_writer : file-like or `None`
            callable file-like object where standard error will be sent.
        use_numpy : bool
            whether to use functions from numpy.
        minimal : bool
            create a minimal interpreter: disable all options (see Note 1).
        no_if : bool
            whether to support `if` blocks
        no_for : bool
            whether to support `for` blocks.
        no_while : bool
            whether to support `while` blocks.
        no_try : bool
            whether to support `try` blocks.
        no_functiondef : bool
            whether to support user-defined functions.
        no_ifexp : bool
            whether to support if expressions.
        no_listcomp : bool
            whether to support list comprehension.
        no_augassign : bool
            whether to support augemented assigments (`a += 1`, etc).
        no_assert : bool
            whether to support `assert`.
        no_delete : bool
            whether to support `del`.
        no_raise : bool
            whether to support `raise`.
        no_print : bool
            whether to support `print`.
        max_time : float
            deprecated.  no longer used (see Note 2)

        Notes
        -----
        1. setting `minimal=True` is equivalent to setting all
           `no_***` options to `True`.
        2. max_time is no longer supported.
        """
        self.writer = writer or stdout
        self.err_writer = err_writer or stderr

        if symtable is None:
            if usersyms is None:
                usersyms = {}
            symtable = make_symbol_table(use_numpy=use_numpy, **usersyms)

        symtable['print'] = self._printer

        self.symtable = symtable
        self._interrupt = None
        self.error = []
        self.error_msg = None
        self.expr = None
        self.retval = None
        self.lineno = 0
        self.use_numpy = HAS_NUMPY and use_numpy

        nodes = ALL_NODES[:]

        if minimal or no_if:           nodes.remove('if')
        if minimal or no_for:          nodes.remove('for')
        if minimal or no_while:        nodes.remove('while')
        if minimal or no_try:          nodes.remove('try')
        if minimal or no_functiondef:  nodes.remove('functiondef')
        if minimal or no_ifexp:        nodes.remove('ifexp')
        if minimal or no_assert:       nodes.remove('assert')
        if minimal or no_delete:       nodes.remove('delete')
        if minimal or no_raise:        nodes.remove('raise')
        if minimal or no_print:        nodes.remove('print')
        if minimal or no_listcomp:     nodes.remove('listcomp')
        if minimal or no_augassign:    nodes.remove('augassign')

        self.node_handlers = {}
        for node in nodes:
            self.node_handlers[node] = getattr(self, "on_%s" % node)

        # to rationalize try/except try/finally for Python2.6 through Python3.3
        if 'try' in self.node_handlers:
            self.node_handlers['tryexcept'] = self.node_handlers['try']
            self.node_handlers['tryfinally'] = self.node_handlers['try']

        self.no_deepcopy = [key for key, val in symtable.items()
                            if (callable(val)
                                or 'numpy.lib.index_tricks' in repr(val))]

    def remove_nodehandler(self, node):
        """remove support for a node
        returns current node handler, so that it
        might be re-added with add_nodehandler()
        """
        if node in self.node_handlers:
            return self.node_handlers.pop(node)

    def set_nodehandler(self, node, handler):
        """set node handler"""
        self.node_handlers[node] = handler

    def user_defined_symbols(self):
        """Return a set of symbols that have been added to symtable after
        construction.

        I.e., the symbols from self.symtable that are not in
        self.no_deepcopy.

        Returns
        -------
        unique_symbols : set
            symbols in symtable that are not in self.no_deepcopy

        """
        sym_in_current = set(self.symtable.keys())
        sym_from_construction = set(self.no_deepcopy)
        unique_symbols = sym_in_current.difference(sym_from_construction)
        return unique_symbols

    def unimplemented(self, node):
        """Unimplemented nodes."""
        self.raise_exception(node, exc=NotImplementedError,
                             msg="'%s' not supported" %
                             (node.__class__.__name__))

    def raise_exception(self, node, exc=None, msg='', expr=None,
                        lineno=None):
        """Add an exception."""
        if self.error is None:
            self.error = []
        if expr is None:
            expr = self.expr
        if len(self.error) > 0 and not isinstance(node, ast.Module):
            msg = '%s' % msg
        err = ExceptionHolder(node, exc=exc, msg=msg, expr=expr, lineno=lineno)
        self._interrupt = ast.Break()
        self.error.append(err)
        if self.error_msg is None:
            self.error_msg = "%s in expr='%s'" % (msg, self.expr)
        elif len(msg) > 0:
            self.error_msg = "%s\n %s" % (self.error_msg, msg)
        if exc is None:
            try:
                exc = self.error[0].exc
            except:
                exc = RuntimeError
        raise exc(self.error_msg)

    # main entry point for Ast node evaluation
    #  parse:  text of statements -> ast
    #  run:    ast -> result
    #  eval:   string statement -> result = run(parse(statement))
    def parse(self, text):
        """Parse statement/expression to Ast representation."""
        self.expr = text
        try:
            return ast.parse(text)
        except SyntaxError:
            self.raise_exception(None, msg='Syntax Error', expr=text)
        except:
            self.raise_exception(None, msg='Runtime Error', expr=text)

    def run(self, node, expr=None, lineno=None, with_raise=True):
        """Execute parsed Ast representation for an expression."""
        # Note: keep the 'node is None' test: internal code here may run
        #    run(None) and expect a None in return.
        if len(self.error) > 0:
            return
        if node is None:
            return
        if isinstance(node, str):
            node = self.parse(node)
        if lineno is not None:
            self.lineno = lineno
        if expr is not None:
            self.expr = expr

        # get handler for this node:
        #   on_xxx with handle nodes of type 'xxx', etc
        try:
            handler = self.node_handlers[node.__class__.__name__.lower()]
        except KeyError:
            return self.unimplemented(node)

        # run the handler:  this will likely generate
        # recursive calls into this run method.
        try:
            ret = handler(node)
            if isinstance(ret, enumerate):
                ret = list(ret)
            return ret
        except:
            if with_raise:
                self.raise_exception(node, expr=expr)

    def __call__(self, expr, **kw):
        """Call class instance as function."""
        return self.eval(expr, **kw)

    def eval(self, expr, lineno=0, show_errors=True):
        """Evaluate a single statement."""
        self.lineno = lineno
        self.error = []
        try:
            node = self.parse(expr)
        except:
            errmsg = exc_info()[1]
            if len(self.error) > 0:
                errmsg = "\n".join(self.error[0].get_error())
            if not show_errors:
                try:
                    exc = self.error[0].exc
                except:
                    exc = RuntimeError
                raise exc(errmsg)
            print(errmsg, file=self.err_writer)
            return
        try:
            return self.run(node, expr=expr, lineno=lineno)
        except:
            errmsg = exc_info()[1]
            if len(self.error) > 0:
                errmsg = "\n".join(self.error[0].get_error())
            if not show_errors:
                try:
                    exc = self.error[0].exc
                except:
                    exc = RuntimeError
                raise exc(errmsg)
            print(errmsg, file=self.err_writer)
            return

    @staticmethod
    def dump(node, **kw):
        """Simple ast dumper."""
        return ast.dump(node, **kw)

    # handlers for ast components
    def on_expr(self, node):
        """Expression."""
        return self.run(node.value)  # ('value',)

    def on_index(self, node):
        """Index."""
        return self.run(node.value)  # ('value',)

    def on_return(self, node):  # ('value',)
        """Return statement: look for None, return special sentinal."""
        self.retval = self.run(node.value)
        if self.retval is None:
            self.retval = ReturnedNone
        return

    def on_repr(self, node):
        """Repr."""
        return repr(self.run(node.value))  # ('value',)

    def on_module(self, node):    # ():('body',)
        """Module def."""
        out = None
        for tnode in node.body:
            out = self.run(tnode)
        return out

    def on_expression(self, node):
        "basic expression"
        return self.on_module(node) # ():('body',)

    def on_pass(self, node):
        """Pass statement."""
        return None  # ()

    def on_ellipsis(self, node):
        """Ellipses."""
        return Ellipsis

    # for break and continue: set the instance variable _interrupt
    def on_interrupt(self, node):    # ()
        """Interrupt handler."""
        self._interrupt = node
        return node

    def on_break(self, node):
        """Break."""
        return self.on_interrupt(node)

    def on_continue(self, node):
        """Continue."""
        return self.on_interrupt(node)

    def on_assert(self, node):    # ('test', 'msg')
        """Assert statement."""
        if not self.run(node.test):
            self.raise_exception(node, exc=AssertionError, msg=node.msg)
        return True

    def on_list(self, node):    # ('elt', 'ctx')
        """List."""
        return [self.run(e) for e in node.elts]

    def on_tuple(self, node):    # ('elts', 'ctx')
        """Tuple."""
        return tuple(self.on_list(node))

    def on_dict(self, node):    # ('keys', 'values')
        """Dictionary."""
        return dict([(self.run(k), self.run(v)) for k, v in
                     zip(node.keys, node.values)])

    def on_num(self, node):   # ('n',)
        """Return number."""
        return node.n

    def on_str(self, node):   # ('s',)
        """Return string."""
        return node.s

    def on_nameconstant(self, node):   # ('value',)
        """named constant"""
        return node.value

    def on_name(self, node):    # ('id', 'ctx')
        """Name node."""
        ctx = node.ctx.__class__
        if ctx in (ast.Param, ast.Del):
            return str(node.id)
        else:
            if node.id in self.symtable:
                return self.symtable[node.id]
            else:
                msg = "name '%s' is not defined" % node.id
                self.raise_exception(node, exc=NameError, msg=msg)

    def on_nameconstant(self, node):
        """ True, False, None in python >= 3.4 """
        return node.value

    def node_assign(self, node, val):
        """Assign a value (not the node.value object) to a node.

        This is used by on_assign, but also by for, list comprehension,
        etc.

        """
        if node.__class__ == ast.Name:
            if not valid_symbol_name(node.id):
                errmsg = "invalid symbol name (reserved word?) %s" % node.id
                self.raise_exception(node, exc=NameError, msg=errmsg)
            self.symtable[node.id] = val
            if node.id in self.no_deepcopy:
                self.no_deepcopy.remove(node.id)

        elif node.__class__ == ast.Attribute:
            if node.ctx.__class__ == ast.Load:
                msg = "cannot assign to attribute %s" % node.attr
                self.raise_exception(node, exc=AttributeError, msg=msg)

            setattr(self.run(node.value), node.attr, val)

        elif node.__class__ == ast.Subscript:
            sym = self.run(node.value)
            xslice = self.run(node.slice)
            if isinstance(node.slice, ast.Index):
                sym[xslice] = val
            elif isinstance(node.slice, ast.Slice):
                sym[slice(xslice.start, xslice.stop)] = val
            elif isinstance(node.slice, ast.ExtSlice):
                sym[xslice] = val
        elif node.__class__ in (ast.Tuple, ast.List):
            if len(val) == len(node.elts):
                for telem, tval in zip(node.elts, val):
                    self.node_assign(telem, tval)
            else:
                raise ValueError('too many values to unpack')

    def on_attribute(self, node):    # ('value', 'attr', 'ctx')
        """Extract attribute."""
        ctx = node.ctx.__class__
        if ctx == ast.Store:
            msg = "attribute for storage: shouldn't be here!"
            self.raise_exception(node, exc=RuntimeError, msg=msg)

        sym = self.run(node.value)
        if ctx == ast.Del:
            return delattr(sym, node.attr)

        # ctx is ast.Load
        errfmt = "'%s' object has not attribute '%s'"

        if (node.attr in UNSAFE_ATTRS or
            (isinstance(sym, six.string_types) and 'format' in node.attr) or
            (isinstance(sym, Procedure) and node.attr not in dir(sym))):
            self.raise_exception(node, exc=AttributeError,
                                 msg=errfmt % (sym, node.attr))

        try:
            return getattr(sym, node.attr)
        except AttributeError:
            self.raise_exception(node, exc=AttributeError,
                                 msg=errfmt % (sym, node.attr))


    def on_assign(self, node):    # ('targets', 'value')
        """Simple assignment."""
        val = self.run(node.value)
        for tnode in node.targets:
            self.node_assign(tnode, val)
        return

    def on_augassign(self, node):    # ('target', 'op', 'value')
        """Augmented assign."""
        return self.on_assign(ast.Assign(targets=[node.target],
                                         value=ast.BinOp(left=node.target,
                                                         op=node.op,
                                                         right=node.value)))

    def on_slice(self, node):    # ():('lower', 'upper', 'step')
        """Simple slice."""
        return slice(self.run(node.lower),
                     self.run(node.upper),
                     self.run(node.step))

    def on_extslice(self, node):    # ():('dims',)
        """Extended slice."""
        return tuple([self.run(tnode) for tnode in node.dims])

    def on_subscript(self, node):    # ('value', 'slice', 'ctx')
        """Subscript handling -- one of the tricky parts."""
        val = self.run(node.value)
        nslice = self.run(node.slice)
        ctx = node.ctx.__class__
        if ctx in (ast.Load, ast.Store):
            if isinstance(node.slice, (ast.Index, ast.Slice, ast.Ellipsis)):
                return val.__getitem__(nslice)
            elif isinstance(node.slice, ast.ExtSlice):
                return val[nslice]
        else:
            msg = "subscript with unknown context"
            self.raise_exception(node, msg=msg)

    def on_delete(self, node):    # ('targets',)
        """Delete statement."""
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
        """Unary operator."""
        return op2func(node.op)(self.run(node.operand))

    def on_binop(self, node):    # ('left', 'op', 'right')
        """Binary operator."""
        return op2func(node.op)(self.run(node.left),
                                self.run(node.right))

    def on_boolop(self, node):    # ('op', 'values')
        """Boolean operator."""
        val = self.run(node.values[0])
        is_and = ast.And == node.op.__class__
        if (is_and and val) or (not is_and and not val):
            for n in node.values[1:]:
                val = op2func(node.op)(val, self.run(n))
                if (is_and and not val) or (not is_and and val):
                    break
        return val

    def on_compare(self, node):  # ('left', 'ops', 'comparators')
        """comparison operators"""
        lval = self.run(node.left)
        out = True
        for op, rnode in zip(node.ops, node.comparators):
            rval = self.run(rnode)
            out = op2func(op)(lval, rval)
            lval = rval
            if self.use_numpy and isinstance(out, ndarray) and out.any():
                break
            elif not out:
                break
        return out

    def on_print(self, node):    # ('dest', 'values', 'nl')
        """Note: implements Python2 style print statement, not print()
        function.

        May need improvement....

        """
        dest = self.run(node.dest) or self.writer
        end = ''
        if node.nl:
            end = '\n'
        out = [self.run(tnode) for tnode in node.values]
        if out and len(self.error) == 0:
            self._printer(*out, file=dest, end=end)

    def _printer(self, *out, **kws):
        """Generic print function."""
        flush = kws.pop('flush', True)
        fileh = kws.pop('file', self.writer)
        sep = kws.pop('sep', ' ')
        end = kws.pop('sep', '\n')

        print(*out, file=fileh, sep=sep, end=end)
        if flush:
            fileh.flush()

    def on_if(self, node):    # ('test', 'body', 'orelse')
        """Regular if-then-else statement."""
        block = node.body
        if not self.run(node.test):
            block = node.orelse
        for tnode in block:
            self.run(tnode)

    def on_ifexp(self, node):    # ('test', 'body', 'orelse')
        """If expressions."""
        expr = node.orelse
        if self.run(node.test):
            expr = node.body
        return self.run(expr)

    def on_while(self, node):    # ('test', 'body', 'orelse')
        """While blocks."""
        while self.run(node.test):
            self._interrupt = None
            for tnode in node.body:
                self.run(tnode)
                if self._interrupt is not None:
                    break
            if isinstance(self._interrupt, ast.Break):
                break
        else:
            for tnode in node.orelse:
                self.run(tnode)
        self._interrupt = None

    def on_for(self, node):    # ('target', 'iter', 'body', 'orelse')
        """For blocks."""
        for val in self.run(node.iter):
            self.node_assign(node.target, val)
            self._interrupt = None
            for tnode in node.body:
                self.run(tnode)
                if self._interrupt is not None:
                    break
            if isinstance(self._interrupt, ast.Break):
                break
        else:
            for tnode in node.orelse:
                self.run(tnode)
        self._interrupt = None

    def on_listcomp(self, node):    # ('elt', 'generators')
        """List comprehension."""
        out = []
        for tnode in node.generators:
            if tnode.__class__ == ast.comprehension:
                for val in self.run(tnode.iter):
                    self.node_assign(tnode.target, val)
                    add = True
                    for cond in tnode.ifs:
                        add = add and self.run(cond)
                    if add:
                        out.append(self.run(node.elt))
        return out

    def on_excepthandler(self, node):  # ('type', 'name', 'body')
        """Exception handler..."""
        return (self.run(node.type), node.name, node.body)

    def on_try(self, node):    # ('body', 'handlers', 'orelse', 'finalbody')
        """Try/except/else/finally blocks."""
        no_errors = True
        for tnode in node.body:
            self.run(tnode, with_raise=False)
            no_errors = no_errors and len(self.error) == 0
            if len(self.error) > 0:
                e_type, e_value, e_tback = self.error[-1].exc_info
                for hnd in node.handlers:
                    htype = None
                    if hnd.type is not None:
                        htype = builtins.get(hnd.type.id, None)
                    if htype is None or isinstance(e_type(), htype):
                        self.error = []
                        if hnd.name is not None:
                            self.node_assign(hnd.name, e_value)
                        for tline in hnd.body:
                            self.run(tline)
                        break
        if no_errors and hasattr(node, 'orelse'):
            for tnode in node.orelse:
                self.run(tnode)

        if hasattr(node, 'finalbody'):
            for tnode in node.finalbody:
                self.run(tnode)

    def on_raise(self, node):    # ('type', 'inst', 'tback')
        """Raise statement: note difference for python 2 and 3."""
        if version_info[0] == 3:
            excnode = node.exc
            msgnode = node.cause
        else:
            excnode = node.type
            msgnode = node.inst
        out = self.run(excnode)
        msg = ' '.join(out.args)
        msg2 = self.run(msgnode)
        if msg2 not in (None, 'None'):
            msg = "%s: %s" % (msg, msg2)
        self.raise_exception(None, exc=out.__class__, msg=msg, expr='')

    def on_call(self, node):
        """Function execution."""
        #  ('func', 'args', 'keywords'. Py<3.5 has 'starargs' and 'kwargs' too)
        func = self.run(node.func)
        if not hasattr(func, '__call__') and not isinstance(func, type):
            msg = "'%s' is not callable!!" % (func)
            self.raise_exception(node, exc=TypeError, msg=msg)

        args = [self.run(targ) for targ in node.args]
        starargs = getattr(node, 'starargs', None)
        if starargs is not None:
            args = args + self.run(starargs)

        keywords = {}
        if six.PY3 and func == print:
            keywords['file'] = self.writer

        for key in node.keywords:
            if not isinstance(key, ast.keyword):
                msg = "keyword error in function call '%s'" % (func)
                self.raise_exception(node, msg=msg)
            if key.arg is None:   # Py3 **kwargs !
                keywords.update(self.run(key.value))
            else:
                keywords[key.arg] = self.run(key.value)

        kwargs = getattr(node, 'kwargs', None)
        if kwargs is not None:
            keywords.update(self.run(kwargs))

        try:
            return func(*args, **keywords)
        except:
            self.raise_exception(node, msg="Error running %s" % (func))

    def on_arg(self, node):    # ('test', 'msg')
        """Arg for function definitions."""
        return node.arg

    def on_functiondef(self, node):
        """Define procedures."""
        # ('name', 'args', 'body', 'decorator_list')
        if node.decorator_list:
            raise Warning("decorated procedures not supported!")
        kwargs = []

        offset = len(node.args.args) - len(node.args.defaults)
        for idef, defnode in enumerate(node.args.defaults):
            defval = self.run(defnode)
            keyval = self.run(node.args.args[idef+offset])
            kwargs.append((keyval, defval))

        if version_info[0] == 3:
            args = [tnode.arg for tnode in node.args.args[:offset]]
        else:
            args = [tnode.id for tnode in node.args.args[:offset]]

        doc = None
        nb0 = node.body[0]
        if isinstance(nb0, ast.Expr) and isinstance(nb0.value, ast.Str):
            doc = nb0.value.s

        varkws = node.args.kwarg
        vararg = node.args.vararg
        if version_info[0] == 3:
            if isinstance(vararg, ast.arg):
                vararg = vararg.arg
            if isinstance(varkws, ast.arg):
                varkws = varkws.arg

        self.symtable[node.name] = Procedure(node.name, self, doc=doc,
                                             lineno=self.lineno,
                                             body=node.body,
                                             args=args, kwargs=kwargs,
                                             vararg=vararg, varkws=varkws)
        if node.name in self.no_deepcopy:
            self.no_deepcopy.pop(node.name)


class Procedure(object):
    """Procedure: user-defined function for asteval.

    This stores the parsed ast nodes as from the 'functiondef' ast node
    for later evaluation.

    """

    def __init__(self, name, interp, doc=None, lineno=0,
                 body=None, args=None, kwargs=None,
                 vararg=None, varkws=None):
        """TODO: docstring in public method."""
        self.__ininit__ = True
        self.name = name
        self.__asteval__ = interp
        self.raise_exc = self.__asteval__.raise_exception
        self.__doc__ = doc
        self.body = body
        self.argnames = args
        self.kwargs = kwargs
        self.vararg = vararg
        self.varkws = varkws
        self.lineno = lineno
        self.__ininit__ = False

    def __setattr__(self, attr, val):
        if not getattr(self, '__ininit__', True):
            self.raise_exc(None, exc=TypeError,
                           msg= "procedure is read-only")
        self.__dict__[attr] = val

    def __dir__(self):
        return ['name']

    def __repr__(self):
        """TODO: docstring in magic method."""
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
        """TODO: docstring in public method."""
        symlocals = {}
        args = list(args)
        nargs = len(args)
        nkws = len(kwargs)
        nargs_expected = len(self.argnames)
        # check for too few arguments, but the correct keyword given
        if (nargs < nargs_expected) and nkws > 0:
            for name in self.argnames[nargs:]:
                if name in kwargs:
                    args.append(kwargs.pop(name))
            nargs = len(args)
            nargs_expected = len(self.argnames)
            nkws = len(kwargs)
        if nargs < nargs_expected:
            msg = "%s() takes at least %i arguments, got %i"
            self.raise_exc(None, exc=TypeError,
                           msg=msg % (self.name, nargs_expected, nargs))
        # check for multiple values for named argument
        if len(self.argnames) > 0 and kwargs is not None:
            msg = "multiple values for keyword argument '%s' in Procedure %s"
            for targ in self.argnames:
                if targ in kwargs:
                    self.raise_exc(None, exc=TypeError,
                                   msg=msg % (targ, self.name),
                                   lineno=self.lineno)

        # check more args given than expected, varargs not given
        if nargs != nargs_expected:
            msg = None
            if nargs < nargs_expected:
                msg = 'not enough arguments for Procedure %s()' % self.name
                msg = '%s (expected %i, got %i)' % (msg, nargs_expected, nargs)
                self.raise_exc(None, exc=TypeError, msg=msg)

        if nargs > nargs_expected and self.vararg is None:
            if nargs - nargs_expected > len(self.kwargs):
                msg = 'too many arguments for %s() expected at most %i, got %i'
                msg = msg % (self.name, len(self.kwargs)+nargs_expected, nargs)
                self.raise_exc(None, exc=TypeError, msg=msg)

            for i, xarg in enumerate(args[nargs_expected:]):
                kw_name = self.kwargs[i][0]
                if kw_name not in kwargs:
                    kwargs[kw_name] = xarg

        for argname in self.argnames:
            symlocals[argname] = args.pop(0)

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
                self.raise_exc(None, msg=msg, exc=TypeError,
                               lineno=self.lineno)

        except (ValueError, LookupError, TypeError,
                NameError, AttributeError):
            msg = 'incorrect arguments for Procedure %s' % self.name
            self.raise_exc(None, msg=msg, lineno=self.lineno)

        save_symtable = self.__asteval__.symtable.copy()
        self.__asteval__.symtable.update(symlocals)
        self.__asteval__.retval = None
        retval = None

        # evaluate script of function
        for node in self.body:
            self.__asteval__.run(node, expr='<>', lineno=self.lineno)
            if len(self.__asteval__.error) > 0:
                break
            if self.__asteval__.retval is not None:
                retval = self.__asteval__.retval
                self.__asteval__.retval = None
                if retval is ReturnedNone:
                    retval = None
                break

        self.__asteval__.symtable = save_symtable
        symlocals = None
        return retval
