"""
Example estimation of single diode model parameters

@author: toddkarin
"""

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import shuntetic data
df = pd.read_pickle('synth01_out.pkl')

from pvpro.estimate import estimate_singlediode_params, estimate_imp_ref
from pvlib.temperature import sapm_cell_from_module

deltaT=3
temperature_cell = sapm_cell_from_module(df['temperature_module_meas'],
                                         poa_global=df['poa_meas'],
                                         deltaT=deltaT)

estimate_imp_ref(irradiance_poa=df['poa_meas'],
                 temperature_cell=temperature_cell,
                 imp=df['i_operation'],
                 figure=True)


est = estimate_singlediode_params(
    irradiance_poa=df['poa_meas'],
    temperature_module=df['temperature_module_meas'],
    vmp=df['v_operation'],
    imp=df['i_operation']
)

print('Estimated photocurrent_ref: {:.2f} A'.format(est['photocurrent_ref']))
print('True photocurrent_ref: {:.2f} A'.format(df['photocurrent_ref'].mean()))

print('Estimated imp_ref: {:.2f} A'.format(est['imp_ref']))
print('True imp_ref: {:.2f} A'.format(df['i_mp_ref'].mean()))

print('Estimated vmp_ref: {:.2f} A'.format(est['vmp_ref']))
print('True vmp_ref: {:.2f} A'.format(df['v_mp_ref'].mean()))