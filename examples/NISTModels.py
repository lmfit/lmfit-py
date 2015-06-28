import os
import sys
from numpy import exp, log, log10, sin, cos, arctan, array
from lmfit import Parameters
thisdir, thisfile = os.path.split(__file__)
NIST_DIR = os.path.join(thisdir, '..', 'NIST_STRD')

def read_params(params):
    if isinstance(params, Parameters):
        return [par.value for par in params.values()]
    else:
        return params

def Bennet5(b, x, y=0):
    b = read_params(b)
    return y - b[0] * (b[1]+x)**(-1/b[2])

def BoxBOD(b, x, y=0):
    b = read_params(b)
    return y - b[0]*(1-exp(-b[1]*x))

def Chwirut(b, x, y=0):
    b = read_params(b)
    return y - exp(-b[0]*x)/(b[1]+b[2]*x)

def DanWood(b, x, y=0):
    b = read_params(b)
    return y - b[0]*x**b[1]

def ENSO(b, x, y=0):
    b = read_params(b)
    pi = 3.141592653589793238462643383279

    return y - b[0] + (b[1]*cos( 2*pi*x/12 ) + b[2]*sin( 2*pi*x/12 ) +
                       b[4]*cos( 2*pi*x/b[3] ) + b[5]*sin( 2*pi*x/b[3] ) +
                       b[7]*cos( 2*pi*x/b[6] ) + b[8]*sin( 2*pi*x/b[6] ) )

def Eckerle4(b, x, y=0):
    b = read_params(b)
    return y - (b[0]/b[1]) * exp(-0.5*((x-b[2])/b[1])**2)

def Gauss(b, x, y=0):
    b = read_params(b)
    return y - b[0]*exp( -b[1]*x ) + (b[2]*exp( -(x-b[3])**2 / b[4]**2 ) +
                                      b[5]*exp( -(x-b[6])**2 / b[7]**2 ) )

def Hahn1(b, x, y=0):
    b = read_params(b)
    return y - ((b[0]+b[1]*x+b[2]*x**2+b[3]*x**3) /
                (1+b[4]*x+b[5]*x**2+b[6]*x**3)  )

def Kirby(b, x, y=0):
    b = read_params(b)
    return y - (b[0] + b[1]*x + b[2]*x**2) / (1 + b[3]*x + b[4]*x**2)

def Lanczos(b, x, y=0):
    b = read_params(b)
    return y - b[0]*exp(-b[1]*x) + b[2]*exp(-b[3]*x) + b[4]*exp(-b[5]*x)

def MGH09(b, x, y=0):
    b = read_params(b)
    return y - b[0]*(x**2+x*b[1]) / (x**2+x*b[2]+b[3])

def MGH10(b, x, y=0):
    b = read_params(b)
    return y - b[0] * exp( b[1]/(x+b[2]) )

def MGH17(b, x, y=0):
    b = read_params(b)
    return y - b[0] + b[1]*exp(-x*b[3]) + b[2]*exp(-x*b[4])

def Misra1a(b, x, y=0):
    b = read_params(b)
    return y - b[0]*(1-exp(-b[1]*x))

def Misra1b(b, x, y=0):
    b = read_params(b)
    return y - b[0] * (1-(1+b[1]*x/2)**(-2))

def Misra1c(b, x, y=0):
    b = read_params(b)
    return y - b[0] * (1-(1+2*b[1]*x)**(-.5))

def Misra1d(b, x, y=0):
    b = read_params(b)
    return y - b[0]*b[1]*x*((1+b[1]*x)**(-1))

def Nelson(b, x, y=None):
    b = read_params(b)
    x1 = x[:,0]
    x2 = x[:,1]
    if y is None:
        return  - exp(b[0] - b[1]*x1 * exp(-b[2]*x2))
    return log(y) - (b[0] - b[1]*x1 * exp(-b[2]*x2) )

def Rat42(b, x, y=0):
    b = read_params(b)
    return  y - b[0] / (1+exp(b[1]-b[2]*x))

def Rat43(b, x, y=0):
    b = read_params(b)
    return  y - b[0] / ((1+exp(b[1]-b[2]*x))**(1/b[3]))

def Roszman1(b, x, y=0):
    b = read_params(b)
    pi = 3.141592653589793238462643383279
    return y - b[0] - b[1]*x - arctan(b[2]/(x-b[3]))/pi

def Thurber(b, x, y=0):
    b = read_params(b)
    return y - ( (b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3) /
                 (1 + b[4]*x + b[5]*x**2 + b[6]*x**3) )

#  Model name        fcn,    #fitting params, dim of x
Models = {'Bennett5':  (Bennet5,  3, 1),
          'BoxBOD':    (BoxBOD,   2, 1),
          'Chwirut1':  (Chwirut,  3, 1),
          'Chwirut2':  (Chwirut,  3, 1),
          'DanWood':   (DanWood,  2, 1),
          'ENSO':      (ENSO,     9, 1),
          'Eckerle4':  (Eckerle4, 3, 1),
          'Gauss1':    (Gauss,    8, 1),
          'Gauss2':    (Gauss,    8, 1),
          'Gauss3':    (Gauss,    8, 1),
          'Hahn1':     (Hahn1,    7, 1),
          'Kirby2':    (Kirby,    5, 1),
          'Lanczos1':  (Lanczos,  6, 1),
          'Lanczos2':  (Lanczos,  6, 1),
          'Lanczos3':  (Lanczos,  6, 1),
          'MGH09':     (MGH09,    4, 1),
          'MGH10':     (MGH10,    3, 1),
          'MGH17':     (MGH17,    5, 1),
          'Misra1a':   (Misra1a,  2, 1),
          'Misra1b' :  (Misra1b,  2, 1),
          'Misra1c' :  (Misra1c,  2, 1),
          'Misra1d' :  (Misra1d,  2, 1),
          'Nelson':    (Nelson,   3, 2),
          'Rat42':     (Rat42,    3, 1),
          'Rat43':     (Rat43,    4, 1),
          'Roszman1':  (Roszman1, 4, 1),
          'Thurber':   (Thurber,  7, 1) }

def ReadNistData(dataset):
    """NIST STRD data is in a simple, fixed format with
    line numbers being significant!
    """
    finp = open(os.path.join(NIST_DIR, "%s.dat" % dataset), 'r')
    lines = [l[:-1] for l in finp.readlines()]
    finp.close()
    ModelLines = lines[30:39]
    ParamLines = lines[40:58]
    DataLines = lines[60:]

    words = ModelLines[1].strip().split()
    nparams = int(words[0])

    start1 = [0]*nparams
    start2 = [0]*nparams
    certval = [0]*nparams
    certerr = [0]*nparams
    for i, text in enumerate(ParamLines[:nparams]):
        [s1, s2, val, err] = [float(x) for x in text.split('=')[1].split()]
        start1[i] = s1
        start2[i] = s2
        certval[i] = val
        certerr[i] = err

    #
    for t in ParamLines[nparams:]:
        t =  t.strip()
        if ':' not in t:
            continue
        val = float(t.split(':')[1])
        if t.startswith('Residual Sum of Squares'):
            sum_squares = val
        elif t.startswith('Residual Standard Deviation'):
            std_dev = val
        elif t.startswith('Degrees of Freedom'):
            nfree = int(val)
        elif t.startswith('Number of Observations'):
            ndata = int(val)

    y, x = [], []
    for d in DataLines:
        vals = [float(i) for i in d.strip().split()]
        y.append(vals[0])
        if len(vals) > 2:
            x.append(vals[1:])
        else:
            x.append(vals[1])

    y = array(y)
    x = array(x)
    out = {'y': y, 'x': x, 'nparams': nparams, 'ndata': ndata,
           'nfree': nfree, 'start1': start1, 'start2': start2,
           'sum_squares': sum_squares, 'std_dev': std_dev,
           'cert': certval,  'cert_values': certval,  'cert_stderr': certerr }
    return out
