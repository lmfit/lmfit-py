"""Test speed of building complex model."""
from copy import deepcopy
import sys
import time

import numpy as np
import pytest

from lmfit import Model
from lmfit.lineshapes import gaussian

sys.setrecursionlimit(2000)


@pytest.mark.flaky(max_runs=5)
def test_manypeaks_speed():
    model = None
    t0 = time.time()
    for i in np.arange(500):
        g = Model(gaussian, prefix=f'g{i}')
        if model is None:
            model = g
        else:
            model += g
    t1 = time.time()
    pars = model.make_params()
    t2 = time.time()
    _cpars = deepcopy(pars)  # noqa: F841
    t3 = time.time()

    # these are very conservative tests that
    # should be satisfied on nearly any machine
    assert (t3-t2) < 0.5
    assert (t2-t1) < 0.5
    assert (t1-t0) < 5.0
