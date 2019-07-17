# <examples/doc_model_savemodel.py>
import numpy as np

from lmfit.model import Model, save_model


def mysine(x, amp, freq, shift):
    return amp * np.sin(x*freq + shift)


sinemodel = Model(mysine)
pars = sinemodel.make_params(amp=1, freq=0.25, shift=0)

save_model(sinemodel, 'sinemodel.sav')
# <end examples/doc_model_savemodel.py>
