"""

This is a simple and fast demonstration for estimating the number of modules
in series and parallel in a power block. First, estimate imp_ref and vmp_ref,
then divide by datahseet values to get estimated number of modules in series
and parallel.

Data is taken from the NIST ground array [1], resampled to 15 minutes.

[1] Matthew T. Boyd. High-Speed Monitoring of Multiple Grid-Connected
Photovoltaic Array Configurations. NIST Technical Note 1896. [2015]
https://doi.org/10.6028/NIST.TN.1896

@author: toddkarin
"""

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvpro.classify import classify_operating_mode
from pvpro.estimate import estimate_imp_ref, estimate_vmp_ref
from pvlib.temperature import sapm_cell_from_module

# Import NIST ground array test data
df = pd.read_csv('nist_ground_resampled-15-min.csv')

# Keys of interest in data frame
current_key = 'InvIDCin_Avg'
voltage_key = 'InvVDCin_Avg'
temperature_module_key = 'RTD_C_Avg_4'
poa_key = 'RefCell1_Wm2_Avg'

# Clip dataframe to avoid bad data on startup. Could improve fit using
# better filtering of bad data
df = df[np.logical_and.reduce((
    df[temperature_module_key] < 50,
    df[temperature_module_key] > 15,
    df.index > df.index[20000],
    df.index < df.index[25000],
))
]

# Datasheet values (Sharp NU-U235F2)'
imp_ref_datasheet = 7.84
vmp_ref_datasheet = 30

# Estimate cell temperature.
df['temperature_cell'] = sapm_cell_from_module(
    module_temperature=df[temperature_module_key],
    poa_global=df[poa_key],
    deltaT=3)

# Classify operating modes, this is a simple fast algorithm to find MPP points.
df['operating_cls'] = classify_operating_mode(voltage=df[voltage_key],
                                              current=df[current_key],
                                              power_clip=np.inf)
# Mask off MPP points
mpp = df['operating_cls'] == 0

# Estimate imp ref
est_imp = estimate_imp_ref(poa=df.loc[mpp, poa_key],
                           temperature_cell=df.loc[mpp, 'temperature_cell'],
                           imp=df.loc[mpp, current_key],
                           figure=True
                           )

# Estimate Vmp ref
est_vmp = estimate_vmp_ref(poa=df.loc[mpp, poa_key],
                           temperature_cell=df.loc[mpp, temperature_module_key],
                           vmp=df.loc[mpp, voltage_key],
                           figure=True,
                           )

# Compare estimated values and ground truth
info = {
    'Imp_ref power block estimate': est_imp['i_mp_ref'],
    'Imp_ref datasheet': imp_ref_datasheet,
    'parallel_strings estimate': est_imp['i_mp_ref'] / imp_ref_datasheet,
    'parallel_strings true': 96,
    'Vmp_ref power block estimate': est_vmp['v_mp_ref'],
    'Vmp_ref datasheet': vmp_ref_datasheet,
    'modules_per_string estimate': est_vmp['v_mp_ref'] / vmp_ref_datasheet,
    'modules_per_string true': 12,
}

print(pd.Series(info))
