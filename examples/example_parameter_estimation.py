"""
Example estimation of single diode model parameters

@author: toddkarin
"""
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import synthetic data
df = pd.read_pickle('synth01_out.pkl')

from pvpro.estimate import estimate_singlediode_params, estimate_imp_ref, \
    estimate_resistance_series, estimate_saturation_current_ref
from pvlib.temperature import sapm_cell_from_module
from pvlib.ivtools import fit_sdm_desoto, fit_sdm_cec_sam
from pvterms import rename
from pvlib.pvsystem import calcparams_desoto

delta_T = 3

est = estimate_singlediode_params(
    irradiance_poa=df['poa_meas'],
    temperature_module=df['temperature_module_meas'],
    vmp=df['v_dc'],
    imp=df['i_dc'],
    delta_T=delta_T,
    figure=True
)

# Format output to compare
compare = pd.DataFrame(est, index=['Estimate'])

for k in est.keys():
    if k in df:
        compare.loc['True', k] = df[k].mean()
print(compare.transpose().to_string())
