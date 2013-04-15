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
    amp = pars['amp'].value
    decay = pars['decay'].value
    amp2 = pars['amp2'].value
    decay2 = pars['decay2'].value


    model = amp*np.exp(-x*decay)+amp2*np.exp(-x*decay2)
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

out = minimize(residual, fit_params, args=(x,), kws={'data':data})
out.leastsq()
ci, trace = conf_interval(out, trace=True)


names=fit_params.keys()

if HASPYLAB:
    pylab.rcParams['font.size']=8
    pylab.plot(x,data)
    pylab.figure()
    cm=pylab.cm.coolwarm
    for i in range(4):
        for j in range(4):
            pylab.subplot(4,4,16-j*4-i)
            if i!=j:
                x,y,m = conf_interval2d(out,names[i],names[j],20,20)
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


    



