# <examples/doc_model_savemodelresult2.py>
import numpy as np

from lmfit.model import save_modelresult
from lmfit.models import ExponentialModel, GaussianModel

dat = np.loadtxt('NIST_Gauss2.dat')
x = dat[:, 1]
y = dat[:, 0]

exp_mod = ExponentialModel(prefix='exp_')
pars = exp_mod.guess(y, x=x)

gauss1 = GaussianModel(prefix='g1_')
pars.update(gauss1.make_params(center=dict(value=105, min=75, max=125),
                               sigma=dict(value=15, min=0),
                               amplitude=dict(value=2000, min=0)))

gauss2 = GaussianModel(prefix='g2_')
pars.update(gauss2.make_params(center=dict(value=155, min=125, max=175),
                               sigma=dict(value=15, min=0),
                               amplitude=dict(value=2000, min=0)))

mod = gauss1 + gauss2 + exp_mod

init = mod.eval(pars, x=x)

result = mod.fit(y, pars, x=x)

save_modelresult(result, 'nistgauss_modelresult.sav')

print(result.fit_report())
# <end examples/doc_model_savemodelresult2.py>
