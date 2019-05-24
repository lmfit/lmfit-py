"""Functions to display fitting results and confidence intervals."""
from math import log10
import re
import numpy as np

try:
    import numdifftools  # noqa: F401
    HAS_NUMDIFFTOOLS = True
except ImportError:
    HAS_NUMDIFFTOOLS = False


def alphanumeric_sort(s, _nsre=re.compile('([0-9]+)')):
    """Sort alphanumeric string."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def getfloat_attr(obj, attr, length=11):
    """Format an attribute of an object for printing."""
    val = getattr(obj, attr, None)
    if val is None:
        return 'unknown'
    elif isinstance(val, int):
        return '%d' % val
    elif isinstance(val, float):
        return gformat(val, length=length).strip()
    return repr(val)


def gformat(val, length=11):
    """Format a number with '%g'-like format.

    Except that:
        a) the length of the output string will be of the requested length.
        b) positive numbers will have a leading blank.
        b) the precision will be as high as possible.
        c) trailing zeros will not be trimmed.

    The precision will typically be length-7.

    Parameters
    ----------
    val : float
       Value to be formatted.
    length : int, optional
       Length of output string (default is 11).

    Returns
    -------
    str
        String of specified length.

    Notes
    ------
     Positive values will have leading blank.

    """
    try:
        expon = int(log10(abs(val)))
    except (OverflowError, ValueError):
        expon = 0
    length = max(length, 7)
    form = 'e'
    prec = length - 7
    if abs(expon) > 99:
        prec -= 1
    elif ((expon > 0 and expon < (prec+4)) or
          (expon <= 0 and -expon < (prec-1))):
        form = 'f'
        prec += 4
        if expon > 0:
            prec -= expon
    fmt = '{0: %i.%i%s}' % (length, prec, form)
    return fmt.format(val)[:length]


CORREL_HEAD = '[[Correlations]] (unreported correlations are < %.3f)'


def fit_report(inpars, modelpars=None, show_correl=True, min_correl=0.1,
               sort_pars=False):
    """Generate a report of the fitting results.

    The report contains the best-fit values for the parameters and their
    uncertainties and correlations.

    Parameters
    ----------
    inpars  : Parameters
       Input Parameters from fit or MinimizerResult returned from a fit.
    modelpars : Parameters, optional
       Known Model Parameters.
    show_correl : bool, optional
       Whether to show list of sorted correlations (default is True).
    min_correl : float, optional
       Smallest correlation in absolute value to show (default is 0.1).
    sort_pars : bool or callable, optional
       Whether to show parameter names sorted in alphanumerical order. If
       False (default), then the parameters will be listed in the order they
       were added to the Parameters dictionary. If callable, then this (one
       argument) function is used to extract a comparison key from each
       list element.

    Returns
    -------
    string
       Multi-line text of fit report.

    """
    from .parameter import Parameters
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
        # dict.keys() returns a KeysView in py3, and they're indexed
        # further down
        parnames = list(params.keys())

    buff = []
    add = buff.append
    namelen = max([len(n) for n in parnames])
    if result is not None:
        add("[[Fit Statistics]]")
        add("    # fitting method   = %s" % (result.method))
        add("    # function evals   = %s" % getfloat_attr(result, 'nfev'))
        add("    # data points      = %s" % getfloat_attr(result, 'ndata'))
        add("    # variables        = %s" % getfloat_attr(result, 'nvarys'))
        add("    chi-square         = %s" % getfloat_attr(result, 'chisqr'))
        add("    reduced chi-square = %s" % getfloat_attr(result, 'redchi'))
        add("    Akaike info crit   = %s" % getfloat_attr(result, 'aic'))
        add("    Bayesian info crit = %s" % getfloat_attr(result, 'bic'))
        if not result.errorbars:
            add("##  Warning: uncertainties could not be estimated:")
            if result.method in ('leastsq', 'least_squares') or HAS_NUMDIFFTOOLS:
                parnames_varying = [par for par in result.params
                                    if result.params[par].vary]
                for name in parnames_varying:
                    par = params[name]
                    space = ' '*(namelen-len(name))
                    if np.allclose(par.value, par.init_value):
                        add('    %s:%s  at initial value' % (name, space))
                    if (np.allclose(par.value, par.min) or np.allclose(par.value, par.min)):
                        add('    %s:%s  at boundary' % (name, space))
            else:
                add("    this fitting method does not natively calculate uncertainties")
                add("    and numdifftools is not installed for lmfit to do this. Use")
                add("    `pip install numdifftools` for lmfit to estimate uncertainties")
                add("    with this fitting method.")

    add("[[Variables]]")
    for name in parnames:
        par = params[name]
        space = ' '*(namelen-len(name))
        nout = "%s:%s" % (name, space)
        inval = '(init = ?)'
        if par.init_value is not None:
            inval = '(init = %.7g)' % par.init_value
        if modelpars is not None and name in modelpars:
            inval = '%s, model_value = %.7g' % (inval, modelpars[name].value)
        try:
            sval = gformat(par.value)
        except (TypeError, ValueError):
            sval = 'Non Numeric Value?'
        if par.stderr is not None:
            serr = gformat(par.stderr)
            try:
                spercent = '({:.2%})'.format(abs(par.stderr/par.value))
            except ZeroDivisionError:
                spercent = ''
            sval = '%s +/-%s %s' % (sval, serr, spercent)

        if par.vary:
            add("    %s %s %s" % (nout, sval, inval))
        elif par.expr is not None:
            add("    %s %s == '%s'" % (nout, sval, par.expr))
        else:
            add("    %s % .7g (fixed)" % (nout, par.value))

    if show_correl:
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
        if len(sort_correl) > 0:
            add(CORREL_HEAD % min_correl)
            maxlen = max([len(k) for k in list(correls.keys())])
        for name, val in sort_correl:
            lspace = max(0, maxlen - len(name))
            add('    C(%s)%s = % .3f' % (name, (' '*30)[:lspace], val))
    return '\n'.join(buff)


def fitreport_html_table(result, show_correl=True, min_correl=0.1):
    """Report minimizer result as an html table"""
    html = []
    add = html.append

    def stat_row(label, val, val2=''):
        add('<tr><td>%s</td><td>%s</td><td>%s</td></tr>' % (label, val, val2))
    add('<h2>Fit Statistics</h2>')
    add('<table>')
    stat_row('fitting method', result.method)
    stat_row('# function evals', result.nfev)
    stat_row('# data points', result.ndata)
    stat_row('# variables', result.nvarys)
    stat_row('chi-square', gformat(result.chisqr))
    stat_row('reduced chi-square', gformat(result.redchi))
    stat_row('Akaike info crit.', gformat(result.aic))
    stat_row('Bayesian info crit.', gformat(result.bic))
    add('</table>')
    add('<h2>Variables</h2>')
    add(result.params._repr_html_())
    if show_correl:
        correls = []
        parnames = list(result.params.keys())
        for i, name in enumerate(result.params):
            par = result.params[name]
            if not par.vary:
                continue
            if hasattr(par, 'correl') and par.correl is not None:
                for name2 in parnames[i+1:]:
                    if (name != name2 and name2 in par.correl and
                            abs(par.correl[name2]) > min_correl):
                        correls.append((name, name2, par.correl[name2]))
        if len(correls) > 0:
            sort_correls = sorted(correls, key=lambda val: abs(val[2]))
            sort_correls.reverse()
            extra = '(unreported correlations are < %.3f)' % (min_correl)
            add('<h2>Correlations %s</h2>' % extra)
            add('<table>')
            for name1, name2, val in sort_correls:
                stat_row(name1, name2, "%.4f" % val)
            add('</table>')
    return ''.join(html)


def params_html_table(params):
    """Returns a HTML representation of parameters data."""
    has_err = any([p.stderr is not None for p in params.values()])
    has_expr = any([p.expr is not None for p in params.values()])
    has_brute = any([p.brute_step is not None for p in params.values()])

    html = []
    add = html.append

    def cell(x, cat='td'):
        return add('<%s> %s </%s>' % (cat, x, cat))

    add('<table><tr>')
    headers = ['name', 'value']
    if has_err:
        headers.extend(['standard error', 'relative error'])
    headers.extend(['initial value', 'min', 'max', 'vary'])
    if has_expr:
        headers.append('expression')
    if has_brute:
        headers.append('brute step')
    for h in headers:
        cell(h, cat='th')
    add('</tr>')

    for par in params.values():
        rows = [par.name, gformat(par.value)]
        if has_err:
            serr = ''
            if par.stderr is not None:
                serr = gformat(par.stderr)
                try:
                    spercent = '({:.2%})'.format(abs(par.stderr/par.value))
                except ZeroDivisionError:
                    spercent = ''
            rows.extend([serr, spercent])
        rows.extend((par.init_value, gformat(par.min),
                     gformat(par.max), '%s' % par.vary))
        if has_expr:
            expr = ''
            if par.expr is not None:
                expr = par.expr
            rows.append(expr)

        if has_brute:
            brute_step = 'None'
            if par.brute_step is not None:
                brute_step = gformat(par.brute_step)
            rows.append(brute_step)

        add('<tr>')
        for r in rows:
            cell(r)
        add('</tr>')
    add('</table>')
    return ''.join(html)


def report_errors(params, **kws):
    """Print a report for fitted params: see error_report()."""
    print(fit_report(params, **kws))


def report_fit(params, **kws):
    """Print a report for fitted params: see error_report()."""
    print(fit_report(params, **kws))


def ci_report(ci, with_offset=True, ndigits=5):
    """Return text of a report for confidence intervals.

    Parameters
    ----------
    with_offset : bool, optional
        Whether to subtract best value from all other values (default is True).
    ndigits : int, optional
        Number of significant digits to show (default is 5).

    Returns
    -------
    str
       Text of formatted report on confidence intervals.

    """
    maxlen = max([len(i) for i in ci])
    buff = []
    add = buff.append

    def convp(x):
        """TODO: function docstring."""
        if abs(x[0]) < 1.e-2:
            return "_BEST_"
        return "%.2f%%" % (x[0]*100)

    title_shown = False
    fmt_best = fmt_diff = "{0:.%if}" % ndigits
    if with_offset:
        fmt_diff = "{0:+.%if}" % ndigits
    for name, row in ci.items():
        if not title_shown:
            add("".join([''.rjust(maxlen+1)] + [i.rjust(ndigits+5)
                                                for i in map(convp, row)]))
            title_shown = True
        thisrow = [" %s:" % name.ljust(maxlen)]
        offset = 0.0
        if with_offset:
            for cval, val in row:
                if abs(cval) < 1.e-2:
                    offset = val
        for cval, val in row:
            if cval < 1.e-2:
                sval = fmt_best.format(val)
            else:
                sval = fmt_diff.format(val-offset)
            thisrow.append(sval.rjust(ndigits+5))
        add("".join(thisrow))

    return '\n'.join(buff)


def report_ci(ci):
    """Print a report for confidence intervals."""
    print(ci_report(ci))
