# <examples/doc_uvars_params.py>
import numpy as np

from lmfit.models import ExponentialModel, GaussianModel

dat = np.loadtxt('NIST_Gauss2.dat')
x = dat[:, 1]
y = dat[:, 0]

bkg = ExponentialModel(prefix='exp_')
gauss1 = GaussianModel(prefix='g1_')
gauss2 = GaussianModel(prefix='g2_')

mod = gauss1 + gauss2 + bkg

pars = mod.make_params(g1_center=105, g1_amplitude=4000, g1_sigma={'value': 15, 'min': 0},
                       g2_center=155, g2_amplitude=2000, g2_sigma={'value': 15, 'min': 0},
                       exp_amplitude=100, exp_decay=100)

out = mod.fit(y, pars, x=x)

print(out.fit_report(min_correl=0.5))

# Area and Centroid of the combined peaks
# option 1: from output uncertainties values

uvar_g1amp = out.uvars['g1_amplitude']
uvar_g2amp = out.uvars['g2_amplitude']
uvar_g1cen = out.uvars['g1_center']
uvar_g2cen = out.uvars['g2_center']

print("### Peak Area: ")
print(f"Peak1: {out.params['g1_amplitude'].value:.5f}+/-{out.params['g1_amplitude'].stderr:.5f}")
print(f"Peak2: {out.params['g2_amplitude'].value:.5f}+/-{out.params['g2_amplitude'].stderr:.5f}")
print(f"Sum  : {(uvar_g1amp + uvar_g2amp):.5f}")


area_quadrature = np.sqrt(out.params['g1_amplitude'].stderr**2 + out.params['g2_amplitude'].stderr**2)

print(f"Correlation of peak1 and peak2 areas:            {out.params['g1_amplitude'].correl['g2_amplitude']:.5f}")
print(f"Stderr Quadrature sum for peak1 and peak2 area2: {area_quadrature:.5f}")
print(f"Stderr of peak1 + peak2 area, with correlation:  {(uvar_g1amp+uvar_g2amp).std_dev:.5f}")

print("### Peak Centroid: ")

centroid = (uvar_g1amp * uvar_g1cen + uvar_g2amp * uvar_g2cen) / (uvar_g1amp + uvar_g2amp)

print(f"Peak Centroid: {centroid:.5f}")

# option 2: define corresponding variables after fit


def post_fit(result):
    "example post fit function"
    result.params.add('post_area', expr='g1_amplitude + g2_amplitude')
    result.params.add('post_centroid', expr='(g1_amplitude*g1_center+ g2_amplitude*g2_center)/(g1_amplitude + g2_amplitude)')


mod.post_fit = post_fit

out = mod.fit(y, pars, x=x)
print(out.fit_report(min_correl=0.5))

# option 3: define corresponding variables before fit
pars.add('area', expr='g1_amplitude + g2_amplitude')
pars.add('centroid', expr='(g1_amplitude*g1_center+ g2_amplitude*g2_center)/(g1_amplitude + g2_amplitude)')

out = mod.fit(y, pars, x=x)
print(out.fit_report(min_correl=0.5))
# <end examples/doc_uvars_params.py>
