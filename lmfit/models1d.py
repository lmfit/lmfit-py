"""
Basic Fitting Models for 1-D data usingsimplifying standard line shapes.

All include optional background that can be
   ('constant', 'linear', 'parabolic')

Models:
     Linear
     Quadratic
     Exponential
     Gaussian
     Lorentzian
     Voigt

Coming Soon....:
     * Step (linear / errorfunc / atan)
     * Rectangular (linear / errorfunc / atan)
     * Sine
     * ErrorFunction

  Original concept and code by Tillsten,
  adopted and expanded by Matt Newville
"""

import numpy as np
from scipy.special import gamma, gammaln, beta, betaln, erf, erfc, wofz

import lmfit
from lmfit import Parameter, Parameters, Minimizer

VALID_BKGS = ('constant', 'linear', 'quadratic')

LOG2 = np.log(2)
SQRT2   = np.sqrt(2))
SQRT2PI = np.sqrt(2*np.pi)
SQRTPI  = np.sqrt(np.pi)

class FitModel(object):
    """base class for fitting models"""
    invalid_bkg_msg = """Warning: unrecoginzed background option '%s'
expected one of the following:
   %s
"""
    def __init__(self, background=None, **kws):
        self.params = Parameters()
        self.initialize_background(background=background, **kws)

    def initialize_background(self, background=None,
                              offset=0, slope=0, quad=0):
        """initialize background parameters"""
        if background is None:
            return
        if background not in VALID_BKGS:
            print self.invalid_bkg_msg % (repr(background),
                                          ', '.join(VALID_BKGS))

        self.params.add('bkg_offset', value=offset, vary=True)
        if background.startswith('line'):
            self.params.add('bkg_slope', value=slope, vary=True)
        elif background.startswith('quad'):
            self.params.add('bkg_slope', value=slope, vary=True)
            self.params.add('bkg_quad', value=slope, vary=True)


    def calc_background(self, x):
        bkg = np.zeros_like(x)
        if 'bkg_offset' in self.params:
            bkg += self.params['bkg_offset'].value
        if 'bkg_slope' in self.params:
            bkg += x*self.params['bkg_slope'].value
        if 'bkg_quad' in self.params:
            bkg += x*x*self.params['bkg_quad'].value
        return bkg

    def __objective(self, params, y=None, x=None, dy=None, **kws):
        """fit objective function"""
        bkg = 0
        if x is not None: bkg = self.calc_background(x)
        if y is None:     y   = 0.0
        if dy is None:    dy  = 1.0
        model = self.model(self.params, x=x, dy=dy, **kws)
        return (model + bkg - y)/dy

    def model(self, params, x=None, **kws):
        raise NotImplementedError

    def guess_starting_values(self, params, y, x=None, **kws):
        raise NotImplementedError

    def fit_report(self, params=None):
        if params is None:
            params = self.params
        return lmfit.fit_report(params)

    def fit(self, y, x=None, dy=None, **kws):
        fcn_kws={'y':y, 'x':x, 'dy':dy}
        fcn_kws.update(kws)
        self.minimizer = Minimizer(self.__objective, self.params,
                                   fcn_kws=fcn_kws, scale_covar=True)
        self.minimizer.prepare_fit()
        self.init = self.model(self.params, x=x, **kws)
        self.minimizer.leastsq()

class LinearModel(FitModel):
    """Linear Model: slope, offset, no background"""
    def __init__(self, offset=0, slope=0, **kws):
        FitModel.__init__(self, background=None, **kws)
        self.params.add('offset', value=offset)
        self.params.add('slope',  value=slope)

    def guess_starting_values(self, y, x):
        sval, oval = np.polyfit(x, y, 1)
        self.params['offset'].value = oval
        self.params['slope'].value = sval

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        return params['offset'].value +  x * params['slope'].value

class QuadraticModel(FitModel):
    """Quadratic Model: slope, offset, quad, no background"""
    def __init__(self, offset=0, slope=0, quad=0, **kws):
        FitModel.__init__(self, background=None, **kws)
        self.params.add('offset', value=offset)
        self.params.add('slope',  value=slope)
        self.params.add('quad',  value=quad)

    def guess_starting_values(self, y, x):
        qval, sval, oval = np.polyfit(x, y, 2)
        self.params['offset'].value = oval
        self.params['slope'].value = sval
        self.params['quad'].value = qval

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        return params['offset'].value +  x * (params['slope'].value +
                                              x * params['quad'].value)

class ExponentialModel(FitModel):
    """Exponential Model: amplitude, decay, optional background"""
    def __init__(self, amplitude=1, decay=1, background=None, **kws):
        FitModel.__init__(self, background=background, **kws)
        self.params.add('amplitude', value=amplitude)
        self.params.add('decay',  value=decay)

    def guess_starting_values(self, y, x):
        try:
            sval, oval = np.polyfit(x, np.log(abs(y)), 1)
        except:
            sval, oval = 1., np.log(abs(max(y)+1.e-9))
        self.params['amplitude'].value = np.exp(oval)
        self.params['decay'].value = (max(x)-min(x))/10.

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        amp   = params['amplitude'].value
        decay = params['decay'].value
        return amp*np.exp(-x / decay)

class GaussianModel(PeakModel):
    """Generalization for Gaussian/Lorenztian/Voigt Model:
       amplitude, center, sigma, optional background
       """
    def __init__(self, amplitude=1, center=0, sigma=1,
                 background=None, **kws):
        FitModel.__init__(self, background=background, **kws)
        self.params.add('amplitude', value=amplitude)
        self.params.add('center',  value=center)
        self.params.add('sigma',  value=sigma, min=0)

    def guess_starting_values(self, y, x):
        """could probably improve this"""
        maxy = max(y)
        self.params['amplitude'].value =(maxy - min(y))*3.0
        imaxy = np.abs(y - maxy).argmin()
        # print imaxy, np.abs(y - maxy).argmin()
        self.params['center'].value = x[imaxy]
        self.params['sigma'].value = (max(x)-min(x))/4.0

    def model(self, params=None, x=None, **kws):
        pass

class GaussianModel(PeakModel):
    """Gaussian Model:
    amplitude, center, sigma, optional background"""
    def __init__(self, amplitude=1, center=0, sigma=1,
                 background=None, **kws):
        PeakModel.__init__(self, amplitude=1, center=0, sigma=1,
                           background=background, **kws)

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        amp = params['amplitude'].value
        cen = params['center'].value
        sig = params['sigma'].value
        amp = amp/(2*SQRT2PI*sig)
        return amp * np.exp(-(x-cen)**2 / (2*sig)**2)

class LorenztianModel(PeakModel):
    """Lorenztian Model:
    amplitude, center, sigma, optional background"""
    def __init__(self, amplitude=1, center=0, sigma=1,
                 background=None, **kws):
        PeakModel.__init__(self, amplitude=1, center=0, sigma=1,
                           background=background, **kws)

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        amp = params['amplitude'].value
        cen = params['center'].value
        sig = params['sigma'].value
        return (amp/(1 + ((x-cen)/sig)**2))/(np.pi*sig)

class VoigtModel(PeakModel):
    """Voigt Model:
    amplitude, center, sigma, optional background
    this version sets gamma=sigma
    """
    def __init__(self, amplitude=1, center=0, sigma=1,
                 background=None, **kws):
        PeakModel.__init__(self, amplitude=1, cen<ter=0, sigma=1,
                           background=background, **kws)

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        amp = params['amplitude'].value
        cen = params['center'].value
        sig = params['sigma'].value
        gam = sig
        z = (x-cen + 1j*gam)/ (sig*SQRT2)
        return wofz(z).real / (sig*SQRT2PI)
