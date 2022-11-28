.. _confidence_chapter:

Calculation of confidence intervals
===================================

.. module:: lmfit.confidence

The lmfit :mod:`confidence` module allows you to explicitly calculate
confidence intervals for variable parameters. For most models, it is not
necessary since the estimation of the standard error from the estimated
covariance matrix is normally quite good.

But for some models, the sum of two exponentials for example, the approximation
begins to fail. For this case, lmfit has the function :func:`conf_interval`
to calculate confidence intervals directly. This is substantially slower
than using the errors estimated from the covariance matrix, but the results
are more robust.


Method used for calculating confidence intervals
------------------------------------------------

The F-test is used to compare our null model, which is the best fit we have
found, with an alternate model, where one of the parameters is fixed to a
specific value. The value is changed until the difference between :math:`\chi^2_0`
and :math:`\chi^2_{f}` can't be explained by the loss of a degree of freedom
within a certain confidence.

.. math::

 F(P_{fix},N-P) = \left(\frac{\chi^2_f}{\chi^2_{0}}-1\right)\frac{N-P}{P_{fix}}

``N`` is the number of data points and ``P`` the number of parameters of the null model.
:math:`P_{fix}` is the number of fixed parameters (or to be more clear, the
difference of number of parameters between our null model and the alternate
model).

Adding a log-likelihood method is under consideration.

A basic example
---------------

First we create an example problem:

.. jupyter-execute::

    import numpy as np

    import lmfit

    x = np.linspace(0.3, 10, 100)
    np.random.seed(0)
    y = 1/(0.1*x) + 2 + 0.1*np.random.randn(x.size)
    pars = lmfit.Parameters()
    pars.add_many(('a', 0.1), ('b', 1))


    def residual(p):
        return 1/(p['a']*x) + p['b'] - y


before we can generate the confidence intervals, we have to run a fit, so
that the automated estimate of the standard errors can be used as a
starting point:

.. jupyter-execute::

    mini = lmfit.Minimizer(residual, pars)
    result = mini.minimize()

    print(lmfit.fit_report(result.params))

Now it is just a simple function call to calculate the confidence
intervals:

.. jupyter-execute::

    ci = lmfit.conf_interval(mini, result)
    lmfit.printfuncs.report_ci(ci)

This shows the best-fit values for the parameters in the ``_BEST_`` column,
and parameter values that are at the varying confidence levels given by
steps in :math:`\sigma`. As we can see, the estimated error is almost the
same, and the uncertainties are well behaved: Going from 1-:math:`\sigma`
(68% confidence) to 3-:math:`\sigma` (99.7% confidence) uncertainties is
fairly linear. It can also be seen that the errors are fairly symmetric
around the best fit value. For this problem, it is not necessary to
calculate confidence intervals, and the estimates of the uncertainties from
the covariance matrix are sufficient.

Working without standard error estimates
----------------------------------------

Sometimes the estimation of the standard errors from the covariance
matrix fails, especially if values are near given bounds. Hence, to
find the confidence intervals in these cases, it is necessary to set
the errors by hand. Note that the standard error is only used to find an
upper limit for each value, hence the exact value is not important.

To set the step-size to 10% of the initial value we loop through all
parameters and set it manually:

.. jupyter-execute::

    for p in result.params:
        result.params[p].stderr = abs(result.params[p].value * 0.1)


..  _label-confidence-advanced:

An advanced example for evaluating confidence intervals
-------------------------------------------------------

Now we look at a problem where calculating the error from approximated
covariance can lead to misleading result -- the same double exponential
problem shown in :ref:`label-emcee`. In fact such a problem is particularly
hard for the Levenberg-Marquardt method, so we first estimate the results
using the slower but robust Nelder-Mead method. We can then compare the
uncertainties computed (if the ``numdifftools`` package is installed) with
those estimated using Levenberg-Marquardt around the previously found
solution. We can also compare to the results of using ``emcee``.


.. jupyter-execute::
    :hide-code:

    import warnings
    warnings.filterwarnings(action="ignore")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.dpi'] = 150
    %matplotlib inline
    %config InlineBackend.figure_format = 'svg'

.. jupyter-execute:: ../examples/doc_confidence_advanced.py
    :hide-output:

which will report:

.. jupyter-execute::
    :hide-code:

    lmfit.report_fit(out2.params, min_correl=0.5)
    print('')
    lmfit.printfuncs.report_ci(ci)

Again we called :func:`conf_interval`, this time with tracing and only for
1- and 2-:math:`\sigma`. Comparing these two different estimates, we see
that the estimate for ``a1`` is reasonably well approximated from the
covariance matrix, but the estimates for ``a2`` and especially for ``t1``, and
``t2`` are very asymmetric and that going from 1 :math:`\sigma` (68%
confidence) to 2 :math:`\sigma` (95% confidence) is not very predictable.

Plots of the confidence region are shown in the figures below for ``a1`` and
``t2`` (left), and ``a2`` and ``t2`` (right):

.. jupyter-execute::
    :hide-code:

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a1', 't2', 30, 30)
    ctp = axes[0].contourf(cx, cy, grid, np.linspace(0, 1, 11))
    fig.colorbar(ctp, ax=axes[0])
    axes[0].set_xlabel('a1')
    axes[0].set_ylabel('t2')

    cx, cy, grid = lmfit.conf_interval2d(mini, out2, 'a2', 't2', 30, 30)
    ctp = axes[1].contourf(cx, cy, grid, np.linspace(0, 1, 11))
    fig.colorbar(ctp, ax=axes[1])
    axes[1].set_xlabel('a2')
    axes[1].set_ylabel('t2')

    plt.show()

Neither of these plots is very much like an ellipse, which is implicitly
assumed by the approach using the covariance matrix. The plots actually
look quite a bit like those found with MCMC and shown in the "corner plot"
in :ref:`label-emcee`. In fact, comparing the confidence interval results
here with the results for the 1- and 2-:math:`\sigma` error estimated with
``emcee``, we can see that the agreement is pretty good and that the
asymmetry in the parameter distributions are reflected well in the
asymmetry of the uncertainties.

The trace returned as the optional second argument from
:func:`conf_interval` contains a dictionary for each variable parameter.
The values are dictionaries with arrays of values for each variable, and an
array of corresponding probabilities for the corresponding cumulative
variables. This can be used to show the dependence between two
parameters:

.. jupyter-execute::
    :hide-output:

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    cx1, cy1, prob = trace['a1']['a1'], trace['a1']['t2'], trace['a1']['prob']
    cx2, cy2, prob2 = trace['t2']['t2'], trace['t2']['a1'], trace['t2']['prob']

    axes[0].scatter(cx1, cy1, c=prob, s=30)
    axes[0].set_xlabel('a1')
    axes[0].set_ylabel('t2')

    axes[1].scatter(cx2, cy2, c=prob2, s=30)
    axes[1].set_xlabel('t2')
    axes[1].set_ylabel('a1')

    plt.show()

which shows the trace of values:

.. jupyter-execute::
    :hide-code:

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    cx1, cy1, prob = trace['a1']['a1'], trace['a1']['t2'], trace['a1']['prob']
    cx2, cy2, prob2 = trace['t2']['t2'], trace['t2']['a1'], trace['t2']['prob']
    axes[0].scatter(cx1, cy1, c=prob, s=30)
    axes[0].set_xlabel('a1')
    axes[0].set_ylabel('t2')
    axes[1].scatter(cx2, cy2, c=prob2, s=30)
    axes[1].set_xlabel('t2')
    axes[1].set_ylabel('a1')
    plt.show()

As an alternative/complement to the confidence intervals, the :meth:`Minimizer.emcee`
method uses Markov Chain Monte Carlo to sample the posterior probability distribution.
These distributions demonstrate the range of solutions that the data supports and we
refer to :ref:`label-emcee` where this methodology was used on the same problem.

Credible intervals (the Bayesian equivalent of the frequentist confidence
interval) can be obtained with this method. MCMC can be used for model
selection, to determine outliers, to marginalize over nuisance parameters, etcetera.
For example, you may have fractionally underestimated the uncertainties on a
dataset. MCMC can be used to estimate the true level of uncertainty on each
data point. A tutorial on the possibilities offered by MCMC can be found at [1]_.

.. [1] https://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/


Confidence Interval Functions
-----------------------------

.. autofunction:: lmfit.conf_interval

.. autofunction:: lmfit.conf_interval2d

.. autofunction:: lmfit.ci_report
