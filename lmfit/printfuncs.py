# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:24:21 2012

@author: Tillsten

Changes:
  -  13-Feb-2013 M Newville
     complemented  "report_errors" and "report_ci" with
     "error_report" and "ci_report" (respectively) which
     return the text of the report.  Thus report_errors()
     is simply:
        def report_errors(params, modelpars=None, show_correl=True):
            print error_report(params, modelpars=modelpars,
                               show_correl=show_correl)
     and similar for report_ci() / ci_report()

"""

from __future__ import print_function
from .parameter import Parameters
import re

def alphanumeric_sort(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def getfloat_attr(obj, attr, fmt='%.3f'):
    "format an attribute of an object for printing"
    val = getattr(obj, attr, None)
    if val is None:
        return 'unknown'
    if isinstance(val, int):
        return '%d' % val
    if isinstance(val, float):
        return fmt % val
    else:
        return repr(val)

def gformat(val, length=11):
    """format a number with '%g'-like format, except that
    the return will be length ``length`` (default=12)
    and have at least length-6 significant digits
    """
    length = max(length, 7)
    fmt = '{: .%ig}' % (length-6)
    if isinstance(val, int):
        out = ('{: .%ig}' % (length-2)).format(val)
        if len(out) > length:
            out = fmt.format(val)
    else:
        out = fmt.format(val)
    if len(out) < length:
        if 'e' in out:
            ie = out.find('e')
            if '.' not in out[:ie]:
                out = out[:ie] + '.' + out[ie:]
            out = out.replace('e', '0'*(length-len(out))+'e')
        else:
            fmt = '{: .%ig}' % (length-1)
            out = fmt.format(val)[:length]
            if len(out) < length:
                pad = '0' if '.' in  out else ' '
                out += pad*(length-len(out))
    return out

CORREL_HEAD = '[[Correlations]] (unreported correlations are < % .3f)'

def fit_report(inpars, modelpars=None, show_correl=True, min_correl=0.1,
               sort_pars=False):
    """return text of a report for fitted params best-fit values,
    uncertainties and correlations

    arguments
    ----------
       inpars       Parameters from fit or Minizer object returned from a fit.
       modelpars    Optional Known Model Parameters [None]
       show_correl  whether to show list of sorted correlations [True]
       min_correl   smallest correlation absolute value to show [0.1]
       sort_pars    If True, then fit_report will show parameter names
                    sorted in alphanumerical order.  If False, then the
                    parameters will be listed in the order they were added to
                    the Parameters dictionary. If sort_pars is callable, then
                    this (one argument) function is used to extract a
                    comparison key from each list element.
    """
    if isinstance(inpars, Parameters):
        result, params = None, inpars
    if hasattr(inpars, 'params'):
        result = inpars
        params = inpars.params
    
    if sort_pars:
        if callable(sort_pars):
            key = sort_pars
        else:
            key = alphanumeric_sort
        parnames = sorted(params, key=key)
    else:
        # dict.keys() returns a KeysView in py3, and they're indexed further
        # down
        parnames = list(params.keys())

    buff = []
    add = buff.append
    if result is not None:
        add("[[Fit Statistics]]")
        add("    # function evals   = %s" % getfloat_attr(result, 'nfev'))
        add("    # data points      = %s" % getfloat_attr(result, 'ndata'))
        add("    # variables        = %s" % getfloat_attr(result, 'nvarys'))
        add("    chi-square         = %s" % getfloat_attr(result, 'chisqr'))
        add("    reduced chi-square = %s" % getfloat_attr(result, 'redchi'))

    namelen = max([len(n) for n in parnames])
    add("[[Variables]]")
    for name in parnames:
        par = params[name]
        space = ' '*(namelen+1-len(name))
        nout = "%s:%s" % (name, space)
        inval = '(init= ?)'
        if par.init_value is not None:
            inval = '(init=% .7g)' % par.init_value
        if modelpars is not None and name in modelpars:
            inval = '%s, model_value =% .7g' % (inval, modelpars[name].value)
        try:
            sval = gformat(par.value)
        except (TypeError, ValueError):
            sval = 'Non Numeric Value?'

        if par.stderr is not None:
            serr = gformat(par.stderr, length=9)

            try:
                spercent = '({:.2%})'.format(abs(par.stderr/par.value))
            except ZeroDivisionError:
                spercent = ''
            sval = '%s +/-%s %s' % (sval, serr, spercent)

        if par.vary:
            add("    %s %s %s" % (nout, sval, inval))
        elif par.expr is not None:
            add("    %s %s  == '%s'" % (nout, sval, par.expr))
        else:
            add("    %s % .7g (fixed)" % (nout, par.value))

    if show_correl:
        add(CORREL_HEAD % min_correl)
        correls = {}
        for i, name in enumerate(parnames):
            par = params[name]
            if not par.vary:
                continue
            if hasattr(par, 'correl') and par.correl is not None:
                for name2 in parnames[i+1:]:
                    if (name != name2 and name2 in par.correl and
                        abs(par.correl[name2]) > min_correl):
                        correls["%s, %s" % (name, name2)] = par.correl[name2]

        sort_correl = sorted(correls.items(), key=lambda it: abs(it[1]))
        sort_correl.reverse()
        for name, val in sort_correl:
            lspace = max(1, 25 - len(name))
            add('    C(%s)%s = % .3f ' % (name, (' '*30)[:lspace], val))
    return '\n'.join(buff)


def report_errors(params, **kws):
    """print a report for fitted params:  see error_report()"""
    print(fit_report(params, **kws))


def report_fit(params, **kws):
    """print a report for fitted params:  see error_report()"""
    print(fit_report(params, **kws))


def ci_report(ci):
    """return text of a report for confidence intervals"""
    maxlen = max([len(i) for i in ci])
    buff = []
    add = buff.append
    convp = lambda x: ("%.2f" % (x[0]*100))+'%'
    conv = lambda x: "%.5f" % x[1]
    title_shown = False
    for name, row in ci.items():
        if not title_shown:
            add("".join([''.rjust(maxlen)]+[i.rjust(10)
                                            for i in map(convp, row)]))
            title_shown = True
        add("".join([name.rjust(maxlen)]+[i.rjust(10)
                                          for i in map(conv,  row)]))
    return '\n'.join(buff)


def report_ci(ci):
    """print a report for confidence intervals"""
    print(ci_report(ci))
