"""

This is a simple and fast demonstration for estimating the number of modules
in series and parallel in a power block.

Run synth01_generate_synthetic_data.py first to generate a sample data file.

@author: toddkarin
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvpro.fit import production_data_curve_fit
from pvpro.classify import classify_operating_mode
from pvpro.estimate import estimate_imp_ref, estimate_vmp_ref
from pvlib.temperature import sapm_cell_from_module

# Import synthetic data for a single module.
df = pd.read_pickle('synth01_out.pkl')

# Clip dataframe shorter.
df = df[:5000]

# Multiply by a known number of modules in parallel and series
parallel_strings = 5
modules_in_series = 22
df['v_dc'] = df['v_dc'] * parallel_strings
df['i_dc'] = df['i_dc'] * modules_in_series

# Datasheet values
imp_ref_datasheet = df['i_mp_ref'][0]
vmp_ref_datasheet = df['v_mp_ref'][0]

# Estimate cell temperature.
df['temperature_cell_meas'] = sapm_cell_from_module(
    module_temperature=df['temperature_module_meas'],
    poa_global=df['poa_meas'],
    deltaT=3)


# Classify operating modes, this is a simple fast algorithm to find MPP.
df['operating_cls'] = classify_operating_mode(voltage=df['v_dc'],
                                              current=df['i_dc'],
                                              power_clip=np.inf)
# Mask off MPP points
mpp = df['operating_cls'] == 0

# Estimate imp ref
est_imp = estimate_imp_ref(poa=df.loc[mpp, 'poa_meas'],
                           temperature_cell=df.loc[
                               mpp, 'temperature_cell_meas'],
                           imp=df.loc[mpp, 'i_dc'],
                           figure=True
                           )
print('Imp ref estimate: {}'.format(est_imp['i_mp_ref']))

# Estimate parallel strings
print('Parallel strings: {}'.format(est_imp['i_mp_ref'] / imp_ref_datasheet))

# Estimate Vmp ref
est_vmp = estimate_vmp_ref(poa=df.loc[mpp, 'poa_meas'],
                           temperature_cell=df.loc[
                               mpp, 'temperature_cell_meas'],
                           vmp=df.loc[mpp, 'v_dc'],
                           figure=True,
                           )
print('Vmp ref estimate: {}'.format(est_vmp['v_mp_ref']))
print('Modules in series: {}'.format(est_vmp['v_mp_ref'] / vmp_ref_datasheet))
