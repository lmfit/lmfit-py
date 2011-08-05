from __future__ import print_function

def report_errors(params):
    """write report for fitted params"""
    parnames = sorted(params)
    print('-------------------------------------')
    print( 'Best Fit Values and Standard Errors:')
    for name in parnames:
        par = params[name]
        if par.vary:
            print(" %s: % .6f +/- %.6f (inital = % .6f)" % (name,
                                                            par.value,
                                                            par.stderr,
                                                            par.init_value))
        elif par.expr is not None:
            print(" %s: % .6f   (== '%s')" % (name, par.value,
                                                   par.expr))
        else:
            print(" %s: fixed" % (name))

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
    print('-------------------------------------')
