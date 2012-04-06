from __future__ import print_function

def report_errors(params, modelpars=None, show_correl=True):
    """write report for fitted params"""
    parnames = sorted(params)
    #print('   -------------------------------------')
    #print( '  Best Fit Values and Standard Errors:')
    namelen = max([len(n) for n in parnames])

    for name in parnames:
        par = params[name]
        space = ' '*(namelen+2 - len(name))
        nout = " %s: %s" % (name, space)
        initval = 'inital= ?'
        if par.init_value is not None:
            initval = 'inital= % .6f' % par.init_value
        if modelpars is not None and name in modelpars:
            initval = '%s, model_value=% .6f' % (initval, modelpars[name].value)
        if par.vary:
            print(" %s % .5f+/- %.5f (%s)" % (nout, par.value,
                                               par.stderr, initval))

        elif par.expr is not None:
            print(" %s % .5f == '%s'" % (nout, par.value,
                                                par.expr))
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
