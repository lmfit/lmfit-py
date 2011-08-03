from __future__ import print_function

def report_errors(params):
    """write report for fitted params"""
    parnames = sorted(params)
    print('-------------------------------------')
    print( 'Best Fit Values and Standard Errors:')
    for name in parnames:
        par = params[name]
        if par.vary:
            print(" %s: %.4g +/- %.3g" % (name, par.value, par.stderr))
        elif par.expr is not None:
            print(" %s: %.4g   == '%s'" % (name, par.value, par.expr))
        else:
            print(" %s: fixed" % (name))
            
    print( 'Correlations:')
    for i, name in enumerate(parnames):
        par = params[name]
        if not par.vary:
            continue
        if hasattr(par, 'correl') and par.correl is not None:
            for name2 in parnames[i+1:]:
                if name != name2 and name2 in par.correl:
                    print('    c(%s, %s) = %.3f ' % (name, name2,
                                                     par.correl[name2]))

    print('-------------------------------------')    
