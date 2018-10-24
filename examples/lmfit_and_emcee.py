#!/usr/bin/env python

import numpy as np

import emcee
import lmfit

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False


def create_prior(params):
    """
    emccee uses a uniform prior for every variable.
    Here we create a functions which checks the bounds
    and returns np.inf if a value is outside of its
    allowed range. WARNING: A uniform prior may not be
    what you want!
    """
    none_to_inf = lambda x, sign=1: sign*np.inf if x is None else x
    lower_bounds = np.array([none_to_inf(i.min, -1) for i in params.values() if i.vary])
    upper_bounds = np.array([none_to_inf(i.max, 1) for i in params.values() if i.vary])

    def bounds_prior(values):
        values = np.asarray(values)
        is_ok = np.all((lower_bounds < values) & (values < upper_bounds))
        return 0 if is_ok else -np.inf
    return bounds_prior


def create_lnliklihood(mini, sigma=None):
    """create a normal-likihood from the residuals"""
    def lnprob(vals, sigma=sigma):
        for v, p in zip(vals, [p for p in mini.params.values() if p.vary]):
            p.value = v
        residuals = mini.residual
        if not sigma:
            # sigma is either the error estimate or it will
            # be part of the sampling.
            sigma = vals[-1]
        val = -0.5*np.sum(np.log(2*np.pi*sigma**2) + (residuals/sigma)**2)
        return val
    return lnprob


def starting_guess(mini, estimate_sigma=True):
    """
    Use best a fit as a starting point for the samplers.
    If no sigmas are given, it is assumed that
    all points have the same uncertainty which will
    be also part of the sampled parameters.
    """
    vals = [i.value for i in mini.params.values() if i.vary]
    if estimate_sigma:
        vals.append(mini.residual.std())
    return vals


def create_all(mini, sigma=None):
    """
    creates the log-poposterior function from a minimizer.
    sigma should is either None or an array with the
    1-sigma uncertainties of each residual point. If None,
    sigma will be assumed the same for all residuals and
    is added to the sampled parameters.
    """
    sigma_given = not sigma is None
    lnprior = create_prior(mini.params)
    lnprob = create_lnliklihood(mini, sigma=sigma)
    guess = starting_guess(mini, not sigma_given)
    if sigma_given:
        func = lambda x: lnprior(x[:]) + lnprob(x)
    else:
        func = lambda x: lnprior(x[:-1]) + lnprob(x)
    return func, guess


# setup example problem.
params = lmfit.Parameters()
params.add_many(('a', 5),
                ('b', -5),
                ('t1', 1, 1, 0),
                ('t2', 15, 1, 0))

x = np.linspace(0, 20, 350)
a, b, t1, t2 = 2, 3, 2, 10  # Real values
y_true = a * np.exp(-x/t1) + b * np.exp(-x/t2)
sigma = 0.02
y = y_true + np.random.randn(x.size)*sigma


def residuals(paras):
    a = paras['a'].value
    b = paras['b'].value
    t1 = paras['t1'].value
    t2 = paras['t2'].value
    return a * np.exp(-x/t1) + b * np.exp(-x/t2) - y


# fit the data with lmfit.
mini = lmfit.Minimizer(residuals, params)
result = mini.leastsq()
lmfit.report_errors(result.params)

# create lnfunc and starting distribution.
lnfunc, guess = create_all(result)
nwalkers, ndim = 30, len(guess)
p0 = emcee.utils.sample_ball(guess, 0.1*np.array(guess), nwalkers)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnfunc)
steps = 500
sampler.run_mcmc(p0, steps)

if HASPYLAB:
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 9))
    for (i, name, rv) in zip(range(5), list(params.keys()) + ['sigma'], [a, b, t1, t2, sigma]):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.05)
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))
        axes[i].axhline(rv, color="#888888", lw=2)
        axes[i].set_ylabel("$%s$" % name)
    axes[-1].set_xlabel("Steps")

    plt.figure()

    try:
        import corner  # use pip install corner
        burnin = 100
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        corner.corner(samples, labels=list(result.params.keys())+['sigma'],
                      truths=[a, b, t1, t2, sigma])
    except ImportError:
        print("Please install corner for a nice overview graphic")
