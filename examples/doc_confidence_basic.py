#!/usr/bin/env python

# <examples/doc_confidence_basic.py>
import lmfit
import numpy as np

x = np.linspace(0.3, 10, 100)
np.random.seed(0)
y = 1/(0.1*x) + 2 + 0.1*np.random.randn(x.size)

pars = lmfit.Parameters()
pars.add_many(('a', 0.1), ('b', 1))


def residual(p):
    return 1/(p['a']*x) + p['b'] - y


mini = lmfit.Minimizer(residual, pars)
result = mini.minimize()
print(lmfit.fit_report(result.params))

ci = lmfit.conf_interval(mini, result)
lmfit.printfuncs.report_ci(ci)
# <end examples/doc_confidence_basic.py>
