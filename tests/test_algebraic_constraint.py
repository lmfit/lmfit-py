from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from lmfit import Parameters, Parameter, Minimizer, Model
from lmfit.lineshapes import gaussian, lorentzian, pvoigt
from lmfit.printfuncs import report_fit

def test_constraints1():
    def residual(pars, x, sigma=None, data=None):
        yg = gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g'])
        yl = lorentzian(x, pars['amp_l'], pars['cen_l'], pars['wid_l'])

        model =  yg +  yl + pars['line_off'] + x * pars['line_slope']
        if data is None:
            return model
        if sigma is None:
            return (model - data)
        return (model - data)/sigma


    n = 601
    xmin = 0.
    xmax = 20.0
    x = linspace(xmin, xmax, n)

    data = (gaussian(x, 21, 8.1, 1.2) +
            lorentzian(x, 10, 9.6, 2.4) +
            random.normal(scale=0.23,  size=n) +
            x*0.5)


    pfit = Parameters()
    pfit.add(name='amp_g',  value=10)
    pfit.add(name='cen_g',  value=9)
    pfit.add(name='wid_g',  value=1)

    pfit.add(name='amp_tot',  value=20)
    pfit.add(name='amp_l',  expr='amp_tot - amp_g')
    pfit.add(name='cen_l',  expr='1.5+cen_g')
    pfit.add(name='wid_l',  expr='2*wid_g')

    pfit.add(name='line_slope', value=0.0)
    pfit.add(name='line_off', value=0.0)

    sigma = 0.021  # estimate of data error (for all data points)

    myfit = Minimizer(residual, pfit,
                      fcn_args=(x,), fcn_kws={'sigma':sigma, 'data':data},
                      scale_covar=True)

    myfit.prepare_fit()
    init = residual(myfit.params, x)

    result = myfit.leastsq()

    print(' Nfev = ', result.nfev)
    print( result.chisqr, result.redchi, result.nfree)

    report_fit(result.params)
    pfit= result.params
    fit = residual(result.params, x)
    assert(pfit['cen_l'].value == 1.5 + pfit['cen_g'].value)
    assert(pfit['amp_l'].value == pfit['amp_tot'].value - pfit['amp_g'].value)
    assert(pfit['wid_l'].value == 2 * pfit['wid_g'].value)

def test_constraints2():
    """add a user-defined function to symbol table"""
    def residual(pars, x, sigma=None, data=None):
        yg = gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g'])
        yl = lorentzian(x, pars['amp_l'], pars['cen_l'], pars['wid_l'])

        model =  yg +  yl + pars['line_off'] + x * pars['line_slope']
        if data is None:
            return model
        if sigma is None:
            return (model - data)
        return (model - data)/sigma


    n = 601
    xmin = 0.
    xmax = 20.0
    x = linspace(xmin, xmax, n)

    data = (gaussian(x, 21, 8.1, 1.2) +
            lorentzian(x, 10, 9.6, 2.4) +
            random.normal(scale=0.23,  size=n) +
            x*0.5)

    pfit = Parameters()
    pfit.add(name='amp_g',  value=10)
    pfit.add(name='cen_g',  value=9)
    pfit.add(name='wid_g',  value=1)

    pfit.add(name='amp_tot',  value=20)
    pfit.add(name='amp_l',  expr='amp_tot - amp_g')
    pfit.add(name='cen_l',  expr='1.5+cen_g')
    pfit.add(name='line_slope', value=0.0)
    pfit.add(name='line_off', value=0.0)

    sigma = 0.021  # estimate of data error (for all data points)

    myfit = Minimizer(residual, pfit,
                      fcn_args=(x,), fcn_kws={'sigma':sigma, 'data':data},
                      scale_covar=True)

    def width_func(wpar):
        """ """
        return 2*wpar

    myfit.params._asteval.symtable['wfun'] = width_func

    try:
        myfit.params.add(name='wid_l', expr='wfun(wid_g)')
    except:
        assert(False)

    result = myfit.leastsq()

    print(' Nfev = ', result.nfev)
    print( result.chisqr, result.redchi, result.nfree)
    report_fit(result.params)
    pfit= result.params
    fit = residual(result.params, x)
    assert(pfit['cen_l'].value == 1.5 + pfit['cen_g'].value)
    assert(pfit['amp_l'].value == pfit['amp_tot'].value - pfit['amp_g'].value)
    assert(pfit['wid_l'].value == 2 * pfit['wid_g'].value)


def test_constraints3():
    """test a constraint with simple function call"""
    x = [1723, 1773, 1823, 1523, 1773, 1033.03078,
         1042.98077, 1047.90937, 1053.95899, 1057.94906,
         1063.13788, 1075.74218, 1086.03102]
    y = [0.79934, -0.31876, -0.46852, 0.05, -0.21,
         11.1708, 10.31844, 9.73069, 9.21319, 9.12457,
         9.05243, 8.66407, 8.29664]

    def VFT(T, ninf=-3, A=5e3, T0=800):
        return ninf + A/(T-T0)

    vftModel = Model(VFT)
    vftModel.set_param_hint('D', vary=False, expr=r'A*log(10)/T0')
    result = vftModel.fit(y, T=x)
    assert(result.params['A'].value > 2600.0)
    assert(result.params['A'].value < 2650.0)
    assert(result.params['D'].value > 7.0)
    assert(result.params['D'].value < 7.5)

if __name__ == '__main__':
    test_constraints1()
    test_constraints2()
    test_constraints3()
