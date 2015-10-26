#
# test speed of building complex model
#
import time
import sys
import numpy as np
from lmfit import Model
from lmfit.lineshapes import gaussian
from copy import deepcopy


sys.setrecursionlimit(2000)

def test_manypeaks_speed():
    x  = np.linspace( -5, 5, 251)
    model = None
    t0 = time.time()
    for i in np.arange(500):
        g = Model(gaussian, prefix='g%i_' % i)
        if model is None:
            model = g
        else:
            model += g
    t1 = time.time()
    pars = model.make_params()
    t2 = time.time()
    cpars = deepcopy(pars)
    t3 = time.time()

    # these are very conservative tests that 
    # should be satisfied on nearly any machine
    assert((t3-t2) < 0.5)
    assert((t2-t1) < 0.5)
    assert((t1-t0) < 5.0)

if __name__ == '__main__':
    test_manypeaks_speed()
