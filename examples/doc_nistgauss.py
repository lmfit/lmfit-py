#!/usr/bin/env python
#<examples/doc_stepmodel.py>
import numpy as np
from lmfit import fit_report
from lmfit.models import GaussianModel, ExponentialModel

import matplotlib.pyplot as plt

dat = np.loadtxt('NIST_Gauss2.dat')
x = dat[:, 1]
y = dat[:, 0]

exp_mod = ExponentialModel(prefix='exp_')
gauss1  = GaussianModel(prefix='g1_')
gauss2  = GaussianModel(prefix='g2_')
exp_mod.guess_starting_values(y, x=x)
gauss1.set_paramval('center',    105, min=100, max=110)
gauss1.set_paramval('sigma',      12, min=3)
gauss1.set_paramval('amplitude', 500, min=10)
gauss2.set_paramval('center',    150, min=140, max=160)
gauss2.set_paramval('sigma',      12, min=3)
gauss2.set_paramval('amplitude', 500, min=10)

mod = gauss1 + gauss2 + exp_mod
out = mod.fit(y, x=x)

print(fit_report(out))

plt.plot(x, y)
plt.plot(x, out.init_fit, 'k--')
plt.plot(x, out.best_fit, 'r-')
plt.show()
#<end examples/doc_stepmodel.py>
