"""
Example process of ivcurves from NIST dataset.
"""

import os
import pandas as pd
import numpy as np
from pvlib.ivtools.sdm import fit_desoto_sandia
from pvlib.temperature import sapm_cell_from_module

import matplotlib
import matplotlib.pyplot as plt

# Used simple algorithm to extract v_oc, v_mp, i_mp, i_sc. Can overwrite with
# better extraction method if necessary.

ivcurves = pd.read_pickle(os.path.join('data', 'nist_iv_curve_sample.pkl'))

plt.figure(0)
plt.clf()
k = 20
plt.plot(ivcurves.loc[k, 'v'], ivcurves.loc[k, 'i'])

specs = {'cells_in_series': 60,
         'alpha_sc': 0.053 * 1e-2 * 8.6,
         'beta_voc': -0.351 * 1e-2 * 37.0}

desoto = fit_desoto_sandia(ivcurves, specs)
