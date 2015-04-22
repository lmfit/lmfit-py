import lmfit
import numpy as np

x = np.linspace(0.3,10,100)
np.random.seed(0)

y = 1/(0.1*x)+2+0.1*np.random.randn(x.size)

p = lmfit.Parameters()
p.add_many(('a', 0.1), ('b', 1))

def residual(p):
   a = p['a'].value
   b = p['b'].value

   return 1/(a*x)+b-y

minimizer = lmfit.Minimizer(residual, p)
out = minimizer.leastsq()
lmfit.printfuncs.report_fit(out.params)

ci = lmfit.conf_interval(minimizer, out)
lmfit.printfuncs.report_ci(ci)
