# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:24:21 2012

@author: Tillsten
"""

from __future__ import print_function

def report_errors(params, modelpars=None, show_correl=True):
    """Print a report for fitted params"""
    parnames = sorted(params)
    #print('   -------------------------------------')
    #print( '  Best Fit Values and Standard Errors:')
    namelen = max([len(n) for n in parnames])

    for name in parnames:
        par = params[name]
        # print( 'PAR : ', par, par.value, par.stderr, par.expr)
        
        space = ' '*(namelen+2 - len(name))
        nout = " %s: %s" % (name, space)
        initval = 'inital = ?'
        if par.init_value is not None:
            initval = 'initial = % .6f' % par.init_value
        if modelpars is not None and name in modelpars:
            initval = '%s, model_value =% .6f' % (initval, modelpars[name].value)

        sval = '% .6f' % par.value
        if par.stderr is not None:
            sval = '% .6f +/- %.6f (%.2f%%)' % (par.value, par.stderr,
                                                abs(par.stderr/par.value)*100)

        if par.vary:
            print(" %s %s %s" % (nout, sval, initval))
        elif par.expr is not None:
            print(" %s %s == '%s'" % (nout, sval, par.expr))
        else:
            print(" %s fixed" % (nout))

    if show_correl:
        print( 'Correlations:')
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
            lspace = max(1, 25 - len(name))
            print('    C(%s)%s = % .3f ' % (name, (' '*30)[:lspace], val))
    #print('-------------------------------------')


def report_ci(ci):
    """Print a report for confidence intervals"""
    max_name_length = max([len(i) for i in ci])
    for count, name in enumerate(ci):
        convp = lambda x: ("%.2f" % (x[0]*100))+'%'
        conv = lambda x: "%.5f" % x[1]
        row = ci[name]

        #Print title once
        if count == 0:
            print("".join([''.rjust(max_name_length)]+[i.rjust(10)   for i in map(convp, row)]))
        print("".join([name.rjust(max_name_length)]+[i.rjust(10) for i in map(conv,  row)]))






