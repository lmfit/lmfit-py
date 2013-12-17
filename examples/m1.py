
import numpy as np
from lmfit.models1d import  GaussianModel
import matplotlib.pyplot as plt

data = np.loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]

model = GaussianModel()  # background='linear'

# model.guess_starting_values(y, x, negative=False)
# model.params['bkg_offset'].value=min(y)

init_fit = model.model(x=x) + model.calc_background(x)
model.fit(y, x=x)

print model.fit_report()

final_fit = model.model(x=x)

plt.plot(x, y)
plt.plot(x, init_fit)
plt.plot(x, final_fit)
plt.show()

