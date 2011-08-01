import pylab
from numpy import linspace, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq, curve_fit

## Generating noisy data to fit
n = 2500
xmin = 0.
xmax = 250.0

vtrue = [14.0,  4.5, 0.123456, 0.010]
vinit = [5,  2.2,   0.0, 1.e-7]

x     = linspace(xmin, xmax, n)
noise = random.normal(scale=3.5, size=n)

def model(x, *vars):
    amp, period, shift, decay = vars
    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    return amp*sin(shift + x/period) * exp(-x*x*decay*decay)

def misfit(vars, x, data):
    return data - model(x, *vars)

data  = model(x, *vtrue) + noise

# vout, cov = curve_fit(model, x, data, vinit, maxfev=70000)
# lsout = leastsq(misfit, vinit, args=(x, data), full_output=1,
#                 maxfev=1000000, xtol=1.e-4, ftol=1.e-4)
# vout, cov, infodict, errmsg, ier = lsout
# print vout,infodict['nfev'], ier
#

lsout = leastsq(misfit, vinit, args=(x, data), full_output=1,
                maxfev=1000000, xtol=1.e-7, ftol=1.e-7)
vout, cov, infodict, errmsg, ier = lsout

if (len(data) > len(vout)) and cov is not None:
    cov = cov * (misfit(vout, *(x, data))**2).sum()/(len(data)-len(vout))

print infodict['nfev'], ier
print errmsg

for i in range(4):
    print "  %.3f  / %.3f => %.3f +/- %.5f" % (vinit[i], vtrue[i], vout[i], sqrt(cov[i,i]))
pylab.plot(x, data, 'ro')
pylab.plot(x, model(x, *vinit), 'k.')
pylab.plot(x, model(x, *vout), 'b-')
pylab.show()

## Plot
def plot_fit():
    print 'Estimater parameters: ', v
    print 'Real parameters: ', v_real
    X = linspace(xmin,xmax,n*5)
    plot(x,y,'ro', X, fp(v,X))
#
# print v
# print success
# plot_fit()
# show()
