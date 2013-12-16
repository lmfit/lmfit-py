import numpy as np
from lmfit.models1d import  GaussianModel
import matplotlib.pyplot as plt

data = np.loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]

model = GaussianModel()
model.guess_starting_values(y, x=x)
# model.params['amplitude'].value=6.0

init_fit = model.model(x=x)
model.fit(y, x=x)

print model.fit_report(min_correl=0.25)

final_fit = model.model(x=x)

plt.plot(x, final_fit, 'r-')
plt.plot(x, init_fit, 'k--')
plt.plot(x, y,         'bo')
plt.show()
