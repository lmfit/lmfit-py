# <examples/doc_model_loadmodelresult2.py>
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from lmfit.model import load_modelresult

if not os.path.exists('nistgauss_modelresult.sav'):
    os.system(f"{sys.executable} doc_model_savemodelresult2.py")

dat = np.loadtxt('NIST_Gauss2.dat')
x = dat[:, 1]
y = dat[:, 0]

result = load_modelresult('nistgauss_modelresult.sav')
print(result.fit_report())

plt.plot(x, y, 'o')
plt.plot(x, result.best_fit, '-')
plt.show()
# <end examples/doc_model_loadmodelresult2.py>
