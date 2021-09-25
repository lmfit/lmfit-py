"""
Emcee and the Model Interface
=============================

"""
import corner
import matplotlib.pyplot as plt
import numpy as np

import lmfit


###############################################################################
# Set up a double-exponential function and create a Model:
def double_exp(x, a1, t1, a2, t2):
    return a1*np.exp(-x/t1) + a2*np.exp(-(x-0.1) / t2)


model = lmfit.Model(double_exp)

###############################################################################
# Generate some fake data from the model with added noise:
truths = (3.0, 2.0, -5.0, 10.0)
x = np.linspace(1, 10, 250)
np.random.seed(0)
y = double_exp(x, *truths)+0.1*np.random.randn(x.size)

###############################################################################
# Create model parameters and give them initial values:
p = model.make_params(a1=4, t1=3, a2=4, t2=3)

###############################################################################
# Fit the model using a traditional minimizer, and show the output:
result = model.fit(data=y, params=p, x=x, method='Nelder', nan_policy='omit')

lmfit.report_fit(result)
result.plot()

###############################################################################
# Calculate parameter covariance using ``emcee``:
#
#  - start the walkers out at the best-fit values
#  - set ``is_weighted`` to ``False`` to estimate the noise weights
#  - set some sensible priors on the uncertainty to keep the MCMC in check

emcee_kws = dict(steps=5000, burn=500, thin=20, is_weighted=False,
                 progress=False)
emcee_params = result.params.copy()
emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))

###############################################################################
# run the MCMC algorithm and show the results:
result_emcee = model.fit(data=y, x=x, params=emcee_params, method='emcee',
                         nan_policy='omit', fit_kws=emcee_kws)

###############################################################################
lmfit.report_fit(result_emcee)

###############################################################################
result_emcee.plot_fit()
plt.plot(x, model.eval(params=result.params, x=x), '--', label='Nelder')
plt.legend()

###############################################################################
# Check the acceptance fraction to see whether ``emcee`` performed well:
plt.plot(result_emcee.acceptance_fraction, 'o')
plt.xlabel('walker')
plt.ylabel('acceptance fraction')

###############################################################################
# Try to compute the autocorrelation time:
if hasattr(result_emcee, "acor"):
    print("Autocorrelation time for the parameters:")
    print("----------------------------------------")
    for i, p in enumerate(result.params):
        print(f'{p} = {result_emcee.acor[i]:.3f}')

###############################################################################
# Plot the parameter covariances returned by ``emcee`` using ``corner``:
emcee_corner = corner.corner(result_emcee.flatchain, labels=result_emcee.var_names,
                             truths=list(result_emcee.params.valuesdict().values()))

###############################################################################
print("\nmedian of posterior probability distribution")
print('--------------------------------------------')
lmfit.report_fit(result_emcee.params)

###############################################################################
# Find the maximum likelihood solution:
highest_prob = np.argmax(result_emcee.lnprob)
hp_loc = np.unravel_index(highest_prob, result_emcee.lnprob.shape)
mle_soln = result_emcee.chain[hp_loc]
print("\nMaximum Likelihood Estimation (MLE):")
print('----------------------------------')
for ix, param in enumerate(emcee_params):
    print(f"{param}: {mle_soln[ix]:.3f}")

quantiles = np.percentile(result_emcee.flatchain['t1'], [2.28, 15.9, 50, 84.2, 97.7])
print(f"\n\n1 sigma spread = {0.5 * (quantiles[3] - quantiles[1]):.3f}")
print(f"2 sigma spread = {0.5 * (quantiles[4] - quantiles[0]):.3f}")
