"""
Example full run of pv-pro analysis using synthetic data.

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

# Set hyperparameters for running model.
hyperparams = {
    'use_voc_points': False,
    'use_mpp_points': True,
    'use_clip_points': False,
    # 'method': 'basinhopping',
    'method': 'minimize',
    'solver': 'L-BFGS-B',
    # 'solver': 'nelder-mead',
    'days_per_run': 30,
    'iterations_per_year': 6,
    'save_figs': False,
    'verbose' : False,
    # 'saturation_current_multistart':[0.8,1.2],
    'start_point_method': 'last',
    'save_figs_directory': 'figures',
    'plot_imp_max': 7,
    'plot_vmp_max': 35,
    'boolean_mask': boolean_mask,
    'singlediode_method':'fast'
}

ret = pvp.execute(iteration='all',
                  **hyperparams)

# Get results
pfit = pvp.result['p']
pfit.index = pfit['t_start']
print(pfit)

# Analyze year-on-year trend.
yoy_result = analyze_yoy(pfit)

extra_text = 'System: {}\n'.format(pvp.system_name) + \
                 'Use mpp points: {}\n'.format(hyperparams['use_mpp_points']) + \
                 'Use voc points: {}\n'.format(hyperparams['use_voc_points']) + \
                 'Use clip points: {}\n'.format(hyperparams['use_clip_points']) + \
                 'Irrad: {}\n'.format(pvp.irradiance_poa_key) + \
                 'Days per run: {}\n'.format(hyperparams['days_per_run']) + \
                 'start point method: {}\n'.format(hyperparams['start_point_method']) + \
                 'Median residual: {:.4f}\n'.format(1000*np.median(pfit['residual']))


compare = pvp.df.resample('M').mean()
compare['t_years'] = np.array(
            [t.year + (t.dayofyear - 1) / 365.25 for t in compare.index])

# Plot results
plot_results_timeseries(pfit,yoy_result=None,
                        extra_text=extra_text,
                        compare=compare)
# plt.savefig('figures/synth02_result.pdf',bbox_inches='tight')