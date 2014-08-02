import numpy as np
from lmfit.old_models1d import  GaussianModel, VoigtModel
import matplotlib.pyplot as plt

data = np.loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]

model = VoigtModel(background='linear')

# get default starting values, but then alter them
model.guess_starting_values(y, x=x)
model.params['amplitude'].value = 2.0

init_fit = model.model(x=x)

# the actual fit
model.fit(y, x=x)

print model.fit_report(min_correl=0.25)

vfit = model.model(x=x)


mod2 = GaussianModel(background='linear')

mod2.fit(y, x=x)
gfit = mod2.model(x=x)

print mod2.fit_report(min_correl=0.25)

print 'Voigt    Sum of Squares: ', ((vfit - y)**2).sum()
print 'Gaussian Sum of Squares: ', ((gfit - y)**2).sum()

plt.plot(x, vfit, 'r-')
plt.plot(x, gfit, 'b-')
plt.plot(x, y,    'bo')
plt.show()
