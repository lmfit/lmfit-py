#!/usr/bin/env python
#<examples/doc_save_model.py>
from lmfit.models import LorentzianModel
from lmfit.model import save_model

smodel = LorentzianModel()

savefile = 'tmp_save_model.sav'

save_model(smodel, savefile)

print("mode saved to '%s'" % savefile)

#<end examples/doc_save_model.py>
