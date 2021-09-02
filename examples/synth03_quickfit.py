"""
Example running quick estimate algorithm on synthetic data.

@author: toddkarin
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvpro import PvProHandler
from pvpro.preprocess import Preprocessor
from pvpro.postprocess import analyze_yoy
from pvpro.plotting import plot_results_timeseries

# Import synthetic data
df = pd.read_pickle('synth01_out.pkl')

# Load preprocessor.
pre = Preprocessor(df,
                   voltage_dc_key='v_dc',
                   current_dc_key='i_dc',
                   temperature_module_key='temperature_module_meas',
                   irradiance_poa_key='poa_meas',
                   modules_per_string=1,
                   parallel_strings=1,
                   )

# Calculate cell temperature from moduule temperature and POA.
pre.calculate_cell_temperature(delta_T=3)

# Two preprocessing modes, 'fast' and 'sdt'. Since this is clean synthetic
# data, we will use the 'fast' pipeline.
method='fast'
if method=='sdt':
    pre.run_preprocess_sdt(correct_dst=True)
    pre.classify_points_sdt()
    pre.build_operating_cls()

elif method=='fast':
    pre.classify_points_pva()
    pre.build_operating_cls()

# Make PvProHandler object to store data.
pvp = PvProHandler(pre.df,
                   system_name='synthetic',
                   cells_in_series=60,
                   resistance_shunt_ref=df['resistance_shunt_ref'].mean(),
                   alpha_isc=0.001,
                   voltage_key='v_dc',
                   current_key='i_dc',
                   temperature_cell_key='temperature_cell',
                   irradiance_poa_key='poa_meas',
                   modules_per_string=1,
                   parallel_strings=1,
                   )

# Estimate startpoint.
pvp.estimate_p0()
print('Estimated startpoint:')
print(pvp.p0)



# Can set a custom startpoint if auto-chosen startpoint isn't great.
pvp.p0 = {'diode_factor': 1.12,
          'photocurrent_ref': 5.9,
          'saturation_current_ref': 2e-9,
          'resistance_series_ref': 0.4,
          'conductance_shunt_extra': 0.001}


# Plot startpoint on top of data.
# plt.figure(0)
# plt.clf()
# pvp.plot_Vmp_Imp_scatter(df=pvp.df[:5000],
#                          p_plot=pvp.p0,
#                          figure_number=4,
#                          plot_vmp_max=37,
#                          plot_imp_max=6)
# plt.title('Startpoint')
# plt.show()

# Set boolean mask for which points to include.
boolean_mask = pvp.df['poa_meas'] > 100

ret = pvp.quick_parameter_extraction(freq='Y',
                                     verbose=False,
                                     figure=True,
                                     )
pfit = ret['p']
pfit
# from pvpro.plotting import plot_results_timeseries

df['t_years'] = np.array(
            [t.year + (t.dayofyear - 1) / 365.25 for t in df.index])
downsample = 100

plt.figure(0,figsize=(3.5,3))
plt.clf()
plt.plot(pfit['t_years'],pfit['diode_factor'],'.',label='estimate')
plt.plot(df['t_years'][::downsample], df['diode_factor'][::downsample],label='true')
plt.xlabel('time')
plt.ylabel('Diode factor')
plt.legend()
plt.show()
plt.savefig('figures/estimated_diode_factor_extraction.pdf',
            bbox_inches='tight')
# plot_results_timeseries(ret['p'])