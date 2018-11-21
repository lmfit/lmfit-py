#!/usr/bin/env python

import numpy as np

import lmfit

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

try:
    import corner
    HASCORNER = True
except ImportError:
    HASCORNER = False


# Set up a double exponential function
def double_exp(x, a1, t1, a2, t2):
    return a1*np.exp(-x/t1) + a2*np.exp(-(x-0.1) / t2)


# Create a lmfit Model from it
model = lmfit.Model(double_exp)

# Generate some fake data from the model with added noise
truths = (3.0, 2.0, -5.0, 10.0)
x = np.linspace(1, 10, 250)
np.random.seed(0)
y = double_exp(x, *truths)+0.1*np.random.randn(len(x))

if HASPYLAB:
    plt.plot(x, y, 'b')
    # plt.savefig('../doc/_images/emcee_dbl_exp.png')
    plt.show()

# Create model parameters and give them initial values
p = model.make_params(a1=4, t1=3, a2=4, t2=3)

# Fit the model and plot the results using a traditional minimizer
result = model.fit(data=y, params=p, x=x, method='Nelder', nan_policy='omit')

lmfit.report_fit(result)

if HASPYLAB:
    result.plot()
    plt.show()

# Calculate parameter covariance using emcee
# Start the walkers out at the best-fit values
# Set is_weighted to False to estimate the noise weights
emcee_kws = dict(steps=1000, burn=300, thin=20, is_weighted=False)
emcee_params = result.params.copy()

# Set some sensible priors on the uncertainty to keep the MCMC in check
emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))
result_emcee = model.fit(data=y, x=x, params=emcee_params,
                         method='emcee', nan_policy='omit',
                         fit_kws=emcee_kws)

lmfit.report_fit(result_emcee)

# Plot the emcee result and compare it with best fit from Nelder method
if HASPYLAB:
    ax = plt.plot(x, model.eval(params=result.params, x=x), label='Nelder', zorder=100)
    result_emcee.plot_fit(ax=ax, data_kws=dict(color='gray', markersize=2))
    plt.show()

# Plot the parameter covariances returned by emcee using corner
if HASPYLAB and HASCORNER:
    emcee_corner = corner.corner(result_emcee.flatchain, labels=result_emcee.var_names,
                                 truths=list(result_emcee.params.valuesdict().values()))
    plt.show()


print("\nmedian of posterior probability distribution")
print('--------------------------------------------')
lmfit.report_fit(result_emcee.params)

# find the maximum likelihood solution
highest_prob = np.argmax(result_emcee.lnprob)
hp_loc = np.unravel_index(highest_prob, result_emcee.lnprob.shape)
mle_soln = result_emcee.chain[hp_loc]
print("\nMaximum likelihood Estimation")
print('-----------------------------')
for ix, param in enumerate(emcee_params):
    print(param + ': ' + str(mle_soln[ix]))

quantiles = np.percentile(result_emcee.flatchain['t1'], [2.28, 15.9, 50, 84.2, 97.7])
print("1 sigma spread", 0.5 * (quantiles[3] - quantiles[1]))
print("2 sigma spread", 0.5 * (quantiles[4] - quantiles[0]))
