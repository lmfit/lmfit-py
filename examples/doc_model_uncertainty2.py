# <examples/doc_model_uncertainty2.py>
import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ExponentialModel, GaussianModel

dat = np.loadtxt('NIST_Gauss2.dat')
x = dat[:, 1]
y = dat[:, 0]

model = (GaussianModel(prefix='g1_') +
         GaussianModel(prefix='g2_') +
         ExponentialModel(prefix='bkg_'))

params = model.make_params(bkg_amplitude=100, bkg_decay=80,
                           g1_amplitude=3000,
                           g1_center=100,
                           g1_sigma=10,
                           g2_amplitude=3000,
                           g2_center=150,
                           g2_sigma=10)

result = model.fit(y, params, x=x)
print(result.fit_report(min_correl=0.5))

comps = result.eval_components(x=x)
dely = result.eval_uncertainty(sigma=3)

fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6))

axes[0][0].plot(x, y, 'o', color='#99002299', markersize=3, label='data')
axes[0][0].plot(x, result.best_fit, '-', label='best fit')
axes[0][0].plot(x, result.init_fit, '--', label='initial fit')
axes[0][0].set_title('data, initial fit, and best-fit')
axes[0][0].legend()

axes[0][1].plot(x, y, 'o', color='#99002299', markersize=3, label='data')
axes[0][1].plot(x, result.best_fit, '-', label='best fit')
axes[0][1].fill_between(x, result.best_fit-dely, result.best_fit+dely,
                        color="#8A8A8A", label=r'3-$\sigma$ band')
axes[0][1].set_title('data, best-fit, and uncertainty band')
axes[0][1].legend()

axes[1][0].plot(x, result.best_fit, '-', label=r'best fit, 3-$\sigma$ band')
axes[1][0].fill_between(x,
                        result.best_fit-result.dely,
                        result.best_fit+result.dely,
                        color="#8A8A8A")

axes[1][0].plot(x, comps['bkg_'], label=r'background, 3-$\sigma$ band')
axes[1][0].fill_between(x,
                        comps['bkg_']-result.dely_comps['bkg_'],
                        comps['bkg_']+result.dely_comps['bkg_'],
                        color="#8A8A8A")

axes[1][0].plot(x, comps['g1_'], label=r'Gaussian #1, 3-$\sigma$ band')
axes[1][0].fill_between(x,
                        comps['g1_']-result.dely_comps['g1_'],
                        comps['g1_']+result.dely_comps['g1_'],
                        color="#8A8A8A")

axes[1][0].plot(x, comps['g2_'], label=r'Gaussian #2, 3-$\sigma$ band')
axes[1][0].fill_between(x,
                        comps['g2_']-result.dely_comps['g2_'],
                        comps['g2_']+result.dely_comps['g2_'],
                        color="#8A8A8A")
axes[1][0].set_title('model components with uncertainty bands')
axes[1][0].legend()

axes[1][1].plot(x, result.best_fit, '-', label='best fit')
axes[1][1].plot(x, 10*result.dely, label=r'3-$\sigma$ total (x10)')
axes[1][1].plot(x, 10*result.dely_comps['bkg_'], label=r'3-$\sigma$ background (x10)')
axes[1][1].plot(x, 10*result.dely_comps['g1_'], label=r'3-$\sigma$ Gaussian #1 (x10)')
axes[1][1].plot(x, 10*result.dely_comps['g2_'], label=r'3-$\sigma$ Gaussian #2 (x10)')
axes[1][1].set_title('uncertainties for model components')
axes[1][1].legend()

plt.show()
# <end examples/doc_model_uncertainty2.py>
