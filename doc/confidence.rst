Calculation of confidence intervals
====================================

.. py:module:: confidence

Since version `0.5`, lmfit is also capable of calculating the confidence
intervals directly. For most models, it is not necessary: the estimation
of the standard error from the estimated covariance matrix is normally quite
good.

But for some models, e.g. a sum of two exponentials, the approximation
begins to fail. For this case, lmfit has the function :func:`conf_interval`
to calculate confidence intervals directly.  This is substantially slower
than using the errors estimated from the covariance matrix, but the results
are more robust.


Method used for calculating confidence intervals
-------------------------------------------------
The F-test is used to compare our null model, which is the best fit we have
found, with an alternate model, where one of the parameters is fixed to a
specific value. The value is changed until the difference between :math:`\chi^2_0`
and :math:`\chi^2_{f}` can't be explained by the loss of a degree of freedom
within a certain confidence.

.. math::

 F(P_{fix},N-P) = \left(\frac{\chi^2_f}{\chi^2_{0}}-1\right)\frac{N-P}{P_{fix}}

N is the number of data-points, P the number of parameter of the null model.
:math:`P_{fix}` is the number of fixed parameters (or to be more clear, the
difference of number of parameters between our null model and the alternate
model).

A log-likelihood method will be added soon.

A basic example
---------------

First we create a toy problem::


    >>> import lmfit
    >>> import numpy as np
    >>> x = np.linspace(0.3,10,100)
    >>> y = 1/(0.1*x)+2+0.1*np.random.randn(x.size)
    >>> p = lmfit.Parameters()
    >>> p.add_many(('a',0.1),('b',1))
    >>> def residual(p):
    ...    a = p['a'].value
    ...    b = p['b'].value
    ...    return 1/(a*x)+b-y


We have to fit it, before we can generate the confidence intervals::


    >>> mi = lmfit.minimize(residual, p)
    >>> mi.leastsq()
    >>> lmfit.printfuncs.report_fit(mi.params)
    [[Variables]]
         a:     0.09978076 +/- 0.0002112132 (0.21%) initial =  0.09978076
         b:     1.992907 +/- 0.0132743 (0.67%) initial =  1.992907
    [[Correlations]] (unreported correlations are <  0.100)
        C(a, b)                      =  0.601



Now it just a simple function call to start the calculation::

    >>> ci = lmfit.conf_interval(mi)
    >>>lmfit.printfuncs.report_ci(ci)
         99.70%    95.00%    67.40%     0.00%    67.40%    95.00%    99.70%
    a   0.09960   0.09981   0.10000   0.10019   0.10039   0.10058   0.10079
    b   1.97035   1.98326   1.99544   2.00008   2.01936   2.03154   2.04445


As we can see, the estimated error is almost the same:  it is not necessary to calculate ci's for this problem.

An advanced example
-------------------

Now we look at a problem, where calculating the error from approximated
covariance can lead to wrong results::

    >>> y = 3*np.exp(-x/2.)-5*np.exp(-x/10.)+0.2*np.random.randn(x.size)
    >>> p = lmfit.Parameters()
    >>> p.add_many(('a1', 5), ('a2', -5), ('t1', 2), ('t2', 5))
    >>> def residual(p):
    ...    a1, a2, t1, t2 = [i.value for i in p.values()]
    ...    return a1*np.exp(-x/t1)+a2*np.exp(-x/t2)-y

    >>> mi = lmfit.minimize(residual, p)
    >>> mi.leastsq()
    >>> lmfit.printfuncs.report_fit(mi.params, show_correl=False)

    [[Variables]]
         a1:     2.611013 +/- 0.3279648 (12.56%) initial =  2.611013
         a2:    -4.512928 +/- 0.3991997 (8.85%) initial = -4.512928
         t1:     1.569477 +/- 0.3345078 (21.31%) initial =  1.569477
         t2:     10.96137 +/- 1.263874 (11.53%) initial =  10.96137


Again we call :func:`conf_interval`, this time with tracing and only for 1-
and 2-sigma::

    >>> ci, trace = lmfit.conf_interval(mi, sigmas=[0.68,0.95], trace=True, verbose=False)
    >>> lmfit.printfuncs.report_ci(ci)
          95.00%    68.00%     0.00%    68.00%    95.00%
    a1   2.11679   2.33696   2.61101   3.06631   4.28694
    a2  -6.39449  -5.05982  -4.20173  -4.19528  -3.97850
    t2   8.00414   9.62688  12.17331  12.17886  13.34857
    t1   1.07009   1.28482   1.37407   1.97509   2.64341

If you compare the calculated error estimates, you will see that the
regular estimate is too small. Now let's plot a confidence region::

    >>> import matplotlib.pylab as plt
    >>> x, y, grid = lmfit.conf_interval2d(mi,'a1','t2',30,30)
    >>> plt.contourf(x, y, grid, np.linspace(0,1,11))
    >>> plt.xlabel('a1')
    >>> plt.colorbar()
    >>> plt.ylabel('t2')
    >>> plt.show()

which shows:

.. image:: _images/conf_interval1.png
   :target: _images/conf_interval1.png
   :width: 85%


Remember the trace? It shows the dependence between two parameters::

    >>> x, y, prob = trace['a1']['a1'], trace['a1']['t2'],trace['a1']['prob']
    >>> x2, y2, prob2 = trace['t2']['t2'], trace['t2']['a1'],trace['t2']['prob']
    >>> plt.scatter(x, y, c=prob ,s=30)
    >>> plt.scatter(x2, y2, c=prob2, s=30)

which shows

.. image:: _images/conf_interval2.png
   :target: _images/conf_interval2.png
   :width: 85%


Documentation of methods
------------------------

.. autofunction:: lmfit.conf_interval
.. autofunction:: lmfit.conf_interval2d



