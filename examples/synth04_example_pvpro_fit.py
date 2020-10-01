"""
"Minimal" example of pvpro fit for a synthetic dataset.

@author: toddkarin
"""

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvpro.fit import production_data_curve_fit
from pvpro.classify import classify_operating_mode

# Import synthetic data
df = pd.read_pickle('synth01_out.pkl')

from pvlib.temperature import sapm_cell_from_module

# Estimate cell temperature.
df['temperature_cell_meas'] = sapm_cell_from_module(
    module_temperature=df['temperature_module_meas'],
    poa_global=df['poa_meas'],
    deltaT=3)

# Classify operating modes.
df['operating_cls'] = classify_operating_mode(voltage=df['v_dc'],
                                              current=df['i_dc'],
                                              power_clip=np.inf)

# Clip dataframe shorter.
df = df[:5000]

# Run the fit
pfit, residual, ret = production_data_curve_fit(
    temperature_cell=df['temperature_cell_meas'],
    effective_irradiance=df['poa_meas'],
    operating_cls=df['operating_cls'],
    voltage=df['v_dc'],
    current=df['i_dc'],
    alpha_isc=0.001,
    resistance_shunt_ref=400,
    p0=None,
    cells_in_series=60,
    band_gap_ref=1.121,
    verbose=True,
    solver='nelder-mead',
    singlediode_method='newton',
    method='minimize',
    use_mpp_points=True,
    use_voc_points=False,
    use_clip_points=False,
)

# Make best fit comparison
pfit_compare = pd.DataFrame(pfit, index=['Best Fit'])
for k in pfit.keys():
    pfit_compare.loc['True',k] = df[k].mean()
print('Best fit:')
print(pfit_compare.transpose())