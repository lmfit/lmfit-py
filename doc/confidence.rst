Calculation of confidence intervals
====================================

.. py:module:: confidence

Since version `0.5`, lmfit is also capable of calculating the confidence 
intervals directly. For most models, it is not necessery: the estimation
of the standard error from the estimated covariance matrix is normally quite 
good. 

But for some models, e.g. a sum of two exponentials, the approximation
begins to fail. For this case, lmfit has the function :func:`conf_interval`
to calculate confidence inverals directly.  This is substantially slower
than using the errors estimated from the covariance matrix, but the results
are more robust.


Method used for calculating confidence intervals
-------------------------------------------------
The F-test is used to compare our null model, which is the best fit we found,
with an alternate model, where on of the parameters is fixed to a specific value. 
The value is changed util the differnce between :math:`\chi^2_0` and 
:math:`\chi^2_{f}` can't be explained by the loss of a degree of freedom with
a certain confidence. 

.. math::

 F(P_{fix},N-P) = \left(\frac{\chi^2_f}{\chi^2_{0}}-1\right)\frac{N-P}{P_{fix}}

N is the number of data-points, P the number of parameter of the null model.
:math:`P_{fix}` is the number of fixed parameters (or to be more clear, the 
difference of number of parameters betweeen our null model and the alternate
model).

A log-likelihood method will be added soon.

A basic example
---------------

First we generate a toy problem.

.. ipython:: python
   
    import lmfit
    import numpy as np     
    x=np.linspace(0.3,10,100)
    y=1/(0.1*x)+2+0.1*np.random.randn(x.size)
    p=lmfit.Parameters()
    p.add_many(('a',0.1),('b',1))    
    def residual(p):
        a=p['a'].value
        b=p['b'].value
        return 1/(a*x)+b-y

We have to fit it before we can generate the confidence intervals.

.. ipython:: python
        
    mi=lmfit.minimize(residual, p)
    mi.leastsq()
    lmfit.printfuncs.report_errors(mi.params)

Now it just a simple function call:

.. ipython:: python    

    ci=lmfit.conf_interval(mi)
    lmfit.printfuncs.report_ci(ci)

As we can see, it is not necessery to caclulate ci's for this problem.

An advanced example
-------------------
Now we look at a problem, where calculating the error from approimated 
covariance can lead to wrong results.

.. ipython:: python

    @suppress
    np.random.seed(1)
    y=3*np.exp(-x/2.)-5*np.exp(-x/10.)+0.2*np.random.randn(x.size)
    p=lmfit.Parameters()
    p.add_many(('a1',5),('a2',-5),('t1',2),('t2',5))
    def residual(p):
        a1,a2,t1,t2=[i.value for i in p.values()]        
        return a1*np.exp(-x/t1)+a2*np.exp(-x/t2)-y

Now lets fit it:

.. ipython:: python

    mi=lmfit.minimize(residual, p)
    mi.leastsq() 
    lmfit.printfuncs.report_errors(mi.params, show_correl=False)

Again we call :func:`conf_interval`, this time with tracing and only for 1-
and 2-sigma:

.. ipython:: python
    
    ci, trace = lmfit.conf_interval(mi,sigmas=[0.68,0.95],trace=True, verbose=0)
    lmfit.printfuncs.report_ci(ci)

If you compare the calculated error estimates, you will see that the 
regular estimate is too small. Now let's plot a coninfidance region:

.. ipython:: python
    
    import matplotlib.pylab as plt
    x, y, grid=lmfit.conf_interval2d(mi,'a1','t2',30,30)    
    plt.contourf(x,y,grid,np.linspace(0,1,11))    
    plt.xlabel('a1');
    plt.colorbar();    
    @savefig conf_interval.png width=7in
    plt.ylabel('t2');

Remember the trace? 

.. ipython:: python
    
    @suppress
    plt.contourf(x,y,grid,np.linspace(0,1,11))    
    @suppress
    plt.xlabel('a1')
    @suppress
    plt.colorbar()
    @suppress  
    plt.ylabel('t2')

    
    x,y,prob=trace['a1']['a1'], trace['a1']['t2'],trace['a1']['prob']
    x2,y2,prob2=trace['t2']['t2'], trace['t2']['a1'],trace['t2']['prob']
    @savefig conf_interval2.png width=7in
    plt.scatter(x,y,c=prob,s=30)
    plt.scatter(x2,y2,c=prob2,s=30)
    


Documentation of methods
------------------------

.. autofunction:: conf_interval
.. autofunction:: conf_interval2d


    
