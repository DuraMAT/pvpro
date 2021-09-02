"""Example of analyzing a particular pvpro fit and how the remaining fit
parameters depend on fixing one fit parameter.

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

from tqdm import tqdm

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
df = df[-2000:]

fit_params = ['diode_factor',
              'photocurrent_ref',
              'resistance_series_ref',
              'conductance_shunt_extra']

# Can set a custom startpoint if auto-chosen startpoint isn't great.
p0 = {'diode_factor': 1.10,
          'photocurrent_ref': 6.0,
          'resistance_series_ref': 0.4,
          'conductance_shunt_extra': 0.001}

saturation_current_list = np.logspace(-10,-8,20)
result = pd.DataFrame()
for j in tqdm(range(len(saturation_current_list))):
    # Run the fit
    out = production_data_curve_fit(
        temperature_cell=df['temperature_cell_meas'],
        effective_irradiance=df['poa_meas'],
        operating_cls=df['operating_cls'],
        voltage=df['v_dc'],
        current=df['i_dc'],
        alpha_isc=0.001,
        resistance_shunt_ref=400,
        saturation_current_ref=saturation_current_list[j],
        p0=p0,
        cells_in_series=60,
        band_gap_ref=1.121,
        verbose=False,
        solver='L-BFGS-B',
        # solver='Nelder-Mead',
        # singlediode_method='lambertw',
        singlediode_method='fast',
        method='minimize',
        use_mpp_points=True,
        use_voc_points=False,
        use_clip_points=False,
    )

    result.loc[j, 'saturation_current_ref'] = saturation_current_list[j]
    for k in out['p']:
        result.loc[j,k] = out['p'][k]
    result.loc[j,'residual'] = out['residual']


saturation_current_ref_true = np.mean(df['saturation_current_ref'])

#
# # Make best fit comparison
# pfit_compare = pd.DataFrame(pfit, index=['Best Fit'])
# for k in pfit.keys():
#     pfit_compare.loc['True',k] = df[k].mean()
# print('Best fit:')
# print(pfit_compare.transpose())

n=0
for k in result:
    plt.figure(n,figsize=(4,2.2))
    plt.clf()
    plt.plot(saturation_current_list, result[k],'k-',label='Fit')

    if k in df:
        plt.axhline(y=df[k].mean(),c='r',label='True')

    plt.axvline(x=df['saturation_current_ref'].mean(), c='r')

    plt.xscale('log')
    plt.xlabel('saturation-current_ref')
    plt.ylabel(k)
    plt.legend()
    plt.show()
    plt.savefig('figures/synth05_loss_vs_saturation_{}.pdf'.format(k),
                bbox_inches='tight')

    n=n+1