#!/usr/bin/env python
#<examples/doc_stepmodel.py>
import numpy as np
from lmfit.models import StepModel, ConstantModel

import matplotlib.pyplot as plt

x = np.linspace(0, 10, 201)
y = np.ones_like(x)
y[:48] = 0.0
y[48:77] = np.arange(77-48)/(77.0-48)
y = 110.2 * (y + 9e-3*np.random.randn(len(x))) + 12.0

step_mod = StepModel(form='erf')
offset = ConstantModel()
step_mod.guess_starting_values(y, x)
offset.set_param('c', y.min())

mod = step_mod + offset
out = mod.fit(y, x=x)

print(mod.fit_report())

plt.plot(x, y)
plt.plot(x, out.init_fit, 'k--')
plt.plot(x, out.best_fit, 'r-')
plt.show()
#<end examples/doc_stepmodel.py>
