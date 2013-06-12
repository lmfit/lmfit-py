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


def fit_report(params, modelpars=None, show_correl=True, min_correl=0.1):
    """return text of a report for fitted params best-fit values,
    uncertainties and correlations

    arguments
    ----------
       params       Parameters from fit
       modelpars    Optional Known Model Parameters [None]
       show_correl  whether to show list of sorted correlations [True]
       min_correl   smallest correlation absolute value to show [0.1]

    """
    parnames = sorted(params)
    buff = []
    add = buff.append
    namelen = max([len(n) for n in parnames])
    add("[[Variables]]")
    for name in parnames:
        par = params[name]
        space = ' '*(namelen+2 - len(name))
        nout = " %s: %s" % (name, space)
        initval = 'inital = ?'
        if par.init_value is not None:
            initval = 'initial = % .6f' % par.init_value
        if modelpars is not None and name in modelpars:
            initval = '%s, model_value =% .6f' % (initval, modelpars[name].value)

        try:
            sval = '% .6f' % par.value
        except (TypeError, ValueError):
            sval = 'Non Numeric Value?'

        if par.stderr is not None:
            sval = '% .6f +/- %.6f' % (par.value, par.stderr)
            try:
                sval = '%s (%.2f%%)' % (sval, abs(par.stderr/par.value)*100)
            except ZeroDivisionError:
                pass

        if par.vary:
            add("    %s %s %s" % (nout, sval, initval))
        elif par.expr is not None:
            add("    %s %s == '%s'" % (nout, sval, par.expr))
        else:
            add("    %s fixed" % (nout))

    if show_correl:
        add('[[Correlations]] (unreported correlations are < % .3f)' % min_correl)
        correls = {}
        for i, name in enumerate(parnames):
            par = params[name]
            if not par.vary:
                continue
            if hasattr(par, 'correl') and par.correl is not None:
                for name2 in parnames[i+1:]:
                    if name != name2 and name2 in par.correl:
                        correls["%s, %s" % (name, name2)] = par.correl[name2]

        sort_correl = sorted(correls.items(), key=lambda it: abs(it[1]))
        sort_correl.reverse()
        for name, val in sort_correl:
            if abs(val) < min_correl:
                break
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
            add("".join([''.rjust(maxlen)]+[i.rjust(10)   for i in map(convp, row)]))
            title_shown = True
        add("".join([name.rjust(maxlen)]+[i.rjust(10) for i in map(conv,  row)]))
    return '\n'.join(buff)

def report_ci(ci):
    """print a report for confidence intervals"""
    print(ci_report(ci))



