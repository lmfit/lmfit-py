# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 19:47:45 2012

@author: Tillsten
"""
import numpy as np
from lmfit import Parameters, minimize, conf_interval, report_errors, report_ci

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq

try:
    import matplotlib.pyplot as plt
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False


p_true = Parameters()
p_true.add('amp', value=14.0)
p_true.add('period', value=5.33)
p_true.add('shift', value=0.123)
p_true.add('decay', value=0.010)

def residual(pars, x, data=None):
    amp = pars['amp'].value
    per = pars['period'].value
    shift = pars['shift'].value
    decay = pars['decay'].value

    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = amp*sin(shift + x/per) * exp(-x*x*decay*decay)
    if data is None:
        return model
    return (model - data)

n = 2500
xmin = 0.
xmax = 250.0
noise = random.normal(scale=0.7215, size=n)
x     = linspace(xmin, xmax, n)
data  = residual(p_true, x) + noise

fit_params = Parameters()
fit_params.add('amp', value=13.0)
fit_params.add('period', value=2)
fit_params.add('shift', value=0.0)
fit_params.add('decay', value=0.02)

out = minimize(residual, fit_params, args=(x,), kws={'data':data})

fit = residual(fit_params, x)

print( ' N fev = ', out.nfev)
print( out.chisqr, out.redchi, out.nfree)

report_errors(fit_params)
#ci=calc_ci(out)
ci, tr = conf_interval(out, trace=True)
report_ci(ci)
    
if HASPYLAB:
    names=fit_params.keys()
    i=0  
    gs=pylab.GridSpec(4,4)
    sx={}
    sy={}
    for fixed in names:   
        j=0        
        for free in names:                                         
            if j in sx and i in sy:                
                ax=pylab.subplot(gs[i,j],sharex=sx[j],sharey=sy[i])                                        
            elif i in sy:
                ax=pylab.subplot(gs[i,j],sharey=sy[i])
                sx[j]=ax
            elif j in sx:
                ax=pylab.subplot(gs[i,j],sharex=sx[j])
                sy[i]=ax
            else:
                ax=pylab.subplot(gs[i,j])
                sy[i]=ax
                sx[j]=ax
            if i<3:
                pylab.setp( ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(free)

            if j>0:                    
                pylab.setp( ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(fixed)        

            res=tr[fixed]                
            prob=res['prob']
            f=prob<0.96
            
            x,y=res[free], res[fixed]
            ax.scatter(x[f],y[f],
                  c=1-prob[f],s=200*(1-prob[f]+0.5))
            ax.autoscale(1,1)
            
               
            
            j=j+1         
        i=i+1
    
    pylab.show()


