

try:
    import numpy
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import scipy
    from scipy.special import gamma
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    

CUSTOM_FUNCTIONS = {}
if HAS_NUMPY:
    log2 = numpy.log(2)
    pi = numpy.pi
    
    def gauss(x, amp, cen, wid):
        return amp * numpy.exp(-log2 * (x-cen) **2 / wid**2)

    def loren(x, amp, cen, wid):
        return (amp  / (1 + ((x-cen)/wid)**2))
            
    def gauss_area(x, amp, cen, wid):
        return numpy.sqrt(log2/pi) * gauss(x, amp, cen, wid) / wid

    def loren_area(x, amp, cen, wid):
        return loren(x, amp, cen, wid) / (pi*wid)
    
    def pvoigt(x, amp, cen, wid, frac):
        return amp * (gauss(x, (1-frac), cen, wid) +
                      loren(x, frac,     cen, wid))
    
    def pvoigt_area(x, amp, cen, wid, frac):
        return amp * (gauss_area(x, (1-frac), cen, wid) +
                      loren_area(x, frac,     cen, wid))
    
    def pearson7(x, amp, cen, wid, expon):
        xp = 1.0 * expon
        return amp / (1 + ( ((x-cen)/wid)**2) * (2**(1/xp) -1) )**xp
    
    
    def pearson7_area(x, amp, cen, wid, expon):
        scale = gamma(expon) * sqrt((2**(1/expon) -1)) / (gamma(expon-0.5))
        return scale * pearson7(x, amp, cen, wid, expon) / (wid*sqrt(pi))

    CUSTOM_FUNCTIONS = {'gauss': gauss, 'gauss_area': gauss_area,
                  'loren': loren, 'loren_area': loren_area,
                  'pvoigt': pvoigt, 'pvoigt_area': pvoigt_area,
                  'pearson7': pearson7}
    
    if HAS_SCIPY:
        def pearson7_area(x, amp, cen, sigma, expon):
            scale = gamma(expon) * sqrt((2**(1/expon) -1)) / (gamma(expon-0.5))
            return scale * pearson7(x, amp, cen, sigma, expon) / (sigma*sqrt(pi))
        CUSTOM_FUNCTIONS['pearson7_area'] = pearson7_area

