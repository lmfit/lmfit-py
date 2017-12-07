#!/usr/bin/env python

# <examples/doc_model_loadmodelresult2.py>
import matplotlib.pyplot as plt
import numpy as np

from lmfit.model import load_modelresult

dat = np.loadtxt('NIST_Gauss2.dat')
x = dat[:, 1]
y = dat[:, 0]

result = load_modelresult('nistgauss_modelresult.sav')
print(result.fit_report())

plt.plot(x, y, 'bo')
plt.plot(x, result.best_fit, 'r-')
plt.show()
# <end examples/doc_model_loadmodelresult2.py>
