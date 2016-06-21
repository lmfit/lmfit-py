#!/usr/bin/env python

from lmfit import Parameters, Minimizer, conf_interval, conf_interval2d, minimize
import numpy as np
from scipy.interpolate import interp1d

try:
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

np.random.seed(1)

p_true = Parameters()
p_true.add('amp', value=14.0)
p_true.add('decay', value=0.010)
p_true.add('amp2', value=-10.0)
p_true.add('decay2', value=0.050)


def residual(pars, x, data=None):
    model = pars['amp']*np.exp(-x*pars['decay'])
    model += pars['amp2']*np.exp(-x*pars['decay2'])
    if data is None:
        return model
    return (model - data)

n = 200
xmin = 0.
xmax = 250.0
noise = np.random.normal(scale=0.7215, size=n)
x     = np.linspace(xmin, xmax, n)
data  = residual(p_true, x) + noise

fit_params = Parameters()
fit_params.add('amp', value=14.0)
fit_params.add('decay', value=0.010)
fit_params.add('amp2', value=-10.0)
fit_params.add('decay2', value=0.050)

mini = Minimizer(residual, fit_params,
                 fcn_args=(x,), fcn_kws={'data':data})

out = mini.leastsq()

ci, trace = conf_interval(mini, out, trace=True)

names = out.params.keys()

if HASPYLAB:
    pylab.rcParams['font.size']=8
    pylab.plot(x, data)
    pylab.figure()
    cm=pylab.cm.coolwarm
    for i in range(4):
        for j in range(4):
            pylab.subplot(4,4,16-j*4-i)
            if i!=j:
                x,y,m = conf_interval2d(mini, out, names[i],names[j],20,20)
                pylab.contourf(x,y,m,np.linspace(0,1,10),cmap=cm)
                pylab.xlabel(names[i])
                pylab.ylabel(names[j])

                x=trace[names[i]][names[i]]
                y=trace[names[i]][names[j]]
                pr=trace[names[i]]['prob']
                s=np.argsort(x)
                pylab.scatter(x[s],y[s],c=pr[s],s=30,lw=1, cmap=cm)
            else:
                x=trace[names[i]][names[i]]
                y=trace[names[i]]['prob']

                t,s=np.unique(x,True)
                f=interp1d(t,y[s],'slinear')
                xn=np.linspace(x.min(),x.max(),50)
                pylab.plot(xn,f(xn),'g',lw=1)
                pylab.xlabel(names[i])
                pylab.ylabel('prob')

    pylab.show()
