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
     Step (linear / erf / atan)
     Rectangular (linear / erf / atan)

  Original concept and code by Tillsten,
  adopted and expanded by Matt Newville
"""

import numpy as np
from scipy.special import gamma, gammaln, beta, betaln, erf, erfc, wofz

from . import Parameter, Parameters, Minimizer
from . import fit_report as lmfit_report

VALID_BKGS = ('constant', 'linear', 'quadratic')

LOG2 = np.log(2)
SQRT2   = np.sqrt(2)
SQRT2PI = np.sqrt(2*np.pi)
SQRTPI  = np.sqrt(np.pi)

def index_of(arr, val):
    """return index of array nearest to a value
    """
    if val < min(arr):
        return 0
    return np.abs(arr-val).argmin()

class FitBackground(object):
    """base class for fitting models
    needs to overwrite calculate() method
    """
    def __init__(self, **kws):
        self.params = Parameters()
        for key, val in kws.items():
            if val is not None:
                self.params.add('bkg_%s' % key, value=val, vary=True)

    def calculate(self, x):
        pass

class PolyBackground(FitBackground):
    """polynomial background: constant, linear, or quadratic"""
    def __init__(self, offset=None, slope=None, quad=None):
        FitBackground.__init__(self, offset=offset, slope=slope, quad=quad)

    def calculate(self, x):
        bkg = np.zeros_like(x)
        if 'bkg_offset' in self.params:
            bkg += self.params['bkg_offset'].value
        if 'bkg_slope' in self.params:
            bkg += x*self.params['bkg_slope'].value
        if 'bkg_quad' in self.params:
            bkg += x*x*self.params['bkg_quad'].value
        return bkg


class FitModel(object):
    """base class for fitting models

    only supports polynomial background (offset, slop, quad)

    """
    invalid_bkg_msg = """Warning: unrecoginzed background option '%s'
expected one of the following:
   %s
"""
    def __init__(self, background=None, **kws):
        self.params = Parameters()
        self.has_initial_guess = False
        self.bkg = None
        self.initialize_background(background=background, **kws)

    def initialize_background(self, background=None,
                              offset=0, slope=0, quad=0):
        """initialize background parameters"""
        if background is None:
            return
        if background not in VALID_BKGS:
            print( self.invalid_bkg_msg % (repr(background),
                                          ', '.join(VALID_BKGS)))

        kwargs = {'offset':offset}
        if background.startswith('line'):
            kwargs['slope'] = slope
        if background.startswith('quad'):
            kwargs['quad'] = quad

        self.bkg = PolyBackground(**kwargs)

        for nam, par in self.bkg.params.items():
            self.params[nam] = par

    def calc_background(self, x):
        if self.bkg is None:
            return 0
        return self.bkg.calculate(x)

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

    def fit_report(self, params=None, **kws):
        if params is None:
            params = self.params
        return lmfit_report(params, **kws)

    def fit(self, y, x=None, dy=None, **kws):
        fcn_kws = {'y': y, 'x': x, 'dy': dy}
        fcn_kws.update(kws)
        if not self.has_initial_guess:
            self.guess_starting_values(y, x=x, **kws)
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

    def guess_starting_values(self, y, x=None, **kws):
        if x is None:
            sval, oval = 0., 0.
        else:
            sval, oval = np.polyfit(x, y, 1)
        self.params['offset'].value = oval
        self.params['slope'].value = sval
        self.has_initial_guess = True

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

    def guess_starting_values(self, y, x=None, **kws):
        if x is None:
            qval, sval, oval = 0., 0., 0.
        else:
            qval, sval, oval = np.polyfit(x, y, 2)
        self.params['offset'].value = oval
        self.params['slope'].value = sval
        self.params['quad'].value = qval
        self.has_initial_guess = True

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

    def guess_starting_values(self, y, x=None, **kws):
        try:
            sval, oval = np.polyfit(x, np.log(abs(y)), 1)
        except:
            sval, oval = 1., np.log(abs(max(y)+1.e-9))
        self.params['amplitude'].value = np.exp(oval)
        self.params['decay'].value = (max(x)-min(x))/10.
        self.has_initial_guess = True

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        amp   = params['amplitude'].value
        decay = params['decay'].value
        return amp*np.exp(-x / decay)

class PeakModel(FitModel):
    """Generalization for Gaussian/Lorentzian/Voigt Model:
       amplitude, center, sigma, optional background
       sets bounds: sigma >= 0
       """
    fwhm_factor = 2.0
    def __init__(self, amplitude=1, center=0, sigma=1,
                 background=None, **kws):
        FitModel.__init__(self, background=background, **kws)
        self.params.add('amplitude', value=amplitude)
        self.params.add('center',  value=center)
        self.params.add('sigma',  value=sigma, min=0)
        self.params.add('fwhm',  expr='%.6f*sigma' % self.fwhm_factor)

    def guess_starting_values(self, y, x=None, negative=False, **kws):
        """could probably improve this"""
        if x is None:
            return
        maxy, miny = max(y), min(y)
        extremey = maxy
        self.params['amplitude'].value =(maxy - miny)*1.5
        if negative:
            extremey = miny
            self.params['amplitude'].value = -(maxy - miny)*1.5
        imaxy = index_of(y, extremey)
        sigma_guess = (max(x)-min(x))/6.0
        halfmax_vals = np.where(y > extremey/2.0)[0]
        if len(halfmax_vals) > 3:
            sigma_guess = (x[halfmax_vals[-1]] - x[halfmax_vals[0]])/self.fwhm_factor

        self.params['center'].value = x[imaxy]
        self.params['sigma'].value = sigma_guess
        if 'bkg_offset' in self.params:
            bkg_off = miny
            if negative:  bkg_off = maxy
            self.params['bkg_offset'].value = bkg_off
        self.has_initial_guess = True

    def model(self, params=None, x=None, **kws):
        pass

class GaussianModel(PeakModel):
    """Gaussian Model:
    amplitude, center, sigma, optional background"""
    fwhm_factor = 2.354820
    def __init__(self, amplitude=1, center=0, sigma=1,
                 background=None, **kws):
        PeakModel.__init__(self, amplitude=1, center=0, sigma=1,
                           background=background, **kws)
        self.params.add('fwhm',  expr='%g*sigma' % self.fwhm_factor)

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        amp = params['amplitude'].value
        cen = params['center'].value
        sig = params['sigma'].value
        amp = amp/(SQRT2PI*sig)
        return amp * np.exp(-(x-cen)**2 / (2*sig**2))

class LorentzianModel(PeakModel):
    """Lorentzian Model:
    amplitude, center, sigma, optional background"""
    fwhm_factor = 2.0
    def __init__(self, amplitude=1, center=0, sigma=1,
                 background=None, **kws):
        PeakModel.__init__(self, amplitude=1, center=0, sigma=1,
                           background=background, **kws)
        self.params.add('fwhm',  expr='%.6f*sigma' % self.fwhm_factor)

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
    fwhm_factor = 3.60131
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
        z = (x-cen + 1j*sig) / (sig*SQRT2)
        return amp*wofz(z).real / (sig*SQRT2PI)

class StepModel(FitModel):
    """Step Model: height, center, width, optional background
    a step can have a form of 'linear' (default), 'atan', or 'erfc'
    which will give the functional form for going from 0 to height
   """
    def __init__(self, height=1, center=0, width=1, form='linear',
                 background=None, **kws):
        FitModel.__init__(self, background=background, **kws)
        self.params.add('height', value=height)
        self.params.add('center',  value=center)
        self.params.add('width',  value=width, min=0)
        self.form = form

    def guess_starting_values(self, y, x=None, **kws):
        if x is None:
            return
        ymin, ymax = min(y), max(y)
        xmin, xmax = min(x), max(x)
        self.params['height'].value = (ymax-ymin)
        self.params['center'].value = (xmax+xmin)/2.0
        self.params['width'].value  = (xmax-xmin)/7.0
        self.has_initial_guess = True

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        height = params['height'].value
        center = params['center'].value
        width  = params['width'].value
        out = (x - center)/max(width, 1.e-13)
        if self.form == 'linear':
            out[np.where(out<0)] = 0.0
            out[np.where(out>1)] = 1.0
        elif self.form == 'atan':
            out = 0.5 + np.arctan(out)/np.pi
        elif self.form == 'erf':
            out = 0.5*(1 + erf(out))
        return height*out

class RectangularModel(FitModel):
    """Rectangular Model:  a step up and a step down:

    height, center1, center2, width1, width2, optional background

    a step can have a form of 'linear' (default), 'atan', or 'erfc'
    which will give the functional form for going from 0 to height
   """
    def __init__(self, height=1, center1=0, width1=1,
                 center2=1, width2=1,
                 form='linear',
                 background=None, **kws):
        FitModel.__init__(self, background=background, **kws)
        self.params.add('height',   value=height)
        self.params.add('center1',  value=center1)
        self.params.add('width1',   value=width1, min=0)
        self.params.add('center2',  value=center2)
        self.params.add('width2',   value=width2, min=0)
        self.params.add('midpoint',   expr='(center1+center2)/2.0')
        self.form = form

    def guess_starting_values(self, y, x=None, **kws):
        if x is None:
            return
        ymin, ymax = min(y), max(y)
        xmin, xmax = min(x), max(x)
        self.params['height'].value = (ymax-ymin)
        self.params['center1'].value = (xmax+xmin)/4.0
        self.params['width1'].value  = (xmax-xmin)/7.0
        self.params['center2'].value = 3*(xmax+xmin)/4.0
        self.params['width2'].value  = (xmax-xmin)/7.0
        self.has_initial_guess = True

    def model(self, params=None, x=None, **kws):
        if params is None:
            params = self.params
        height  = params['height'].value
        center1 = params['center1'].value
        width1  = params['width1'].value
        center2 = params['center2'].value
        width2  = params['width2'].value
        arg1 = (x - center1)/max(width1, 1.e-13)
        arg2 = (center2 - x)/max(width2, 1.e-13)
        if self.step == 'atan':
            out = (np.arctan(arg1) + np.arctan(arg2))/np.pi
        elif self.step == 'erf':
            out = 0.5*(erf(arg1) + erf(arg2))
        else: # 'linear'
            arg1[np.where(arg1 < 0)] = 0.0
            arg1[np.where(arg1 > 1)] = 1.0
            arg2[np.where(arg2 < -1)] = -1.0
            arg2[np.where(arg2 > 0)] = 0.0
            out = arg1 + arg2
        return height*out
