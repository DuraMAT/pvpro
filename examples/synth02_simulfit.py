"""
Example full run of pv-pro analysis using synthetic data.

@author: toddkarin
"""

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

# import seaborn as sns
from pvpro import PvProHandler

# Import synthetic data
df = pd.read_pickle('synth01_out.pkl')

save_figs_directory = 'figures'

# Create a boolean mask to not use values with extra low irradiance

# Make PvProHandler object to store data.
pvp = PvProHandler(df,
                   system_name='synthetic',
                   delta_T=3,
                   use_clear_times=False,
                   cells_in_series=60,
                   resistance_shunt_ref=df['resistance_shunt_ref'].mean(),
                   alpha_isc=0.001,
                   voltage_key='v_dc',
                   current_key='i_dc',
                   temperature_module_key='temperature_module_meas',
                   irradiance_poa_key='poa_meas',
                   modules_per_string=1,
                   parallel_strings=1,
                   )

# Preprocess
pvp.run_preprocess(run_solar_data_tools=False)

# Find clear times (not necessary for synthetic data)
"""
pvp.find_clear_times(smoothness_hyperparam=1000)
pvp.dh.plot_daily_signals(boolean_mask=pvp.dh.boolean_masks.clear_times,
                          start_day=400)
plt.show()
"""

# Can set a custom startpoint if auto-chosen startpoint isn't great.
"""
pvp.p0 = {'diode_factor': 1.15,
          'photocurrent_ref': 5.7,
          'saturation_current_ref': 10e-10,
          'resistance_series_ref': 0.4,
          'conductance_shunt_extra': 0.001}
"""

# Plot startpoint on top of data.
pvp.plot_Vmp_Imp_scatter(df=pvp.df,
                         p_plot=pvp.p0,
                         figure_number=4,
                         plot_vmp_max=4,
                         plot_imp_max=25)
plt.title('Startpoint')

# Set boolean mask for which points to include.
boolean_mask = pvp.df['poa_meas'] > 10

# Set hyperparameters for running model.
hyperparams = {
    'use_voc_points': True,
    'use_mpp_points': True,
    'use_clip_points': False,
    'method': 'minimize',
    'solver': 'L-BFGS-B',
    'days_per_run': 30,
    'time_step_between_iterations_days': 30,
    'start_point_method': 'last',
    'save_figs_directory': save_figs_directory,
    'plot_imp_max': 7,
    'plot_vmp_max': 35,
    'boolean_mask': boolean_mask
}


run_all = False
if run_all:
    # Run on all iterations.
    ret = pvp.execute(iteration='all',
                  save_figs=True,
                  verbose=False,
                  **hyperparams)
else:
    # Run on first iteration
    ret = pvp.execute(iteration=[0,1],
                      save_figs=True,
                      verbose=False,
                      **hyperparams)
# Get results
pfit = pvp.result['p']
print(pfit)


# ------------------------------------------------------------------------------
# Make degradation plots.
# ------------------------------------------------------------------------------

n = 2
figure = plt.figure(21, figsize=(7.5, 5.5))
plt.clf()

figure.subplots(nrows=4, ncols=3, sharex='all')
# plt.subplots(sharex='all')
plt.subplots_adjust(wspace=0.6, hspace=0.1)

ylabel = {'diode_factor': 'Diode factor',
          'photocurrent_ref': 'Photocurrent ref (A)',
          'saturation_current_ref': 'I sat ref (nA)',
          'resistance_series_ref': 'R series ref (Ohm)',
          'resistance_shunt_ref': 'R shunt ref (Ohm)',
          'conductance_shunt_ref': 'G shunt ref (1/Ohm)',
          'conductance_shunt_extra': 'G shunt extra (1/Ohm)',
          'i_sc_ref': 'I sc ref (A)',
          'v_oc_ref': 'V oc ref (V)',
          'i_mp_ref': 'I mp ref (A)',
          'p_mp_ref': 'P mp ref (W)',
          'i_x_ref': 'I x ref (A)',
          'v_mp_ref': 'V mp ref (V)',
          'residual': 'Residual (AU)',
          }

plt.subplot(4, 3, 1)
ax = plt.gca()
plt.axis('off')
plt.text(-0.2, 0,
         'System: {}\n'.format(pvp.system_name) + \
         'Use clear times: {}\n'.format(pvp.use_clear_times) + \
         'Use mpp points: {}\n'.format(hyperparams['use_mpp_points']) + \
         'Use voc points: {}\n'.format(hyperparams['use_voc_points']) + \
         'Use clip points: {}\n'.format(hyperparams['use_clip_points']) + \
         'Temp: {}\n'.format(pvp.temperature_module_key) + \
         'Irrad: {}\n'.format(pvp.irradiance_poa_key) + \
         'Days per run: {}\n'.format(hyperparams['days_per_run']) + \
         'start point method: {}\n'.format(hyperparams['start_point_method']) + \
         'Minimize method: {}\n'.format(hyperparams['solver'])
         , fontsize=8
         )

for k in ['diode_factor', 'photocurrent_ref', 'saturation_current_ref',
          'resistance_series_ref', 'conductance_shunt_extra', 'i_mp_ref',
          'v_mp_ref',
          'p_mp_ref', 'i_sc_ref', 'v_oc_ref', 'residual', ]:

    ax = plt.subplot(4, 3, n)

    if k == 'saturation_current_ref':
        scale = 1e9
    elif k == 'residual':
        scale = 1e3
    else:
        scale = 1

    plt.plot(pfit['t_mean'], pfit[k] * scale, '.',
             color=[0, 0, 0.8],
             label='pvpro')
    if k in ['i_mp_ref','v_mp_ref','p_mp_ref']:
        plt.plot(pfit['t_mean'], pfit[k + '_est'],'.',
                 color=[0, 0.8, 0.8],
                 label='pvpro-fast')

    ylims = scale * np.array([pfit[k].min(), pfit[k].max()])

    if k in df.keys():
        plt.plot(df.index, df[k] * scale, '--',
                 color=[1, 0.2, 0.2],
                 label=True)
        ylims[0] = np.min([ylims[0], df[k].min() * scale])
        ylims[1] = np.max([ylims[1], df[k].max() * scale])

    plt.ylabel(ylabel[k], fontsize=9)
    # plt.gca().fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')

    if np.nanmax(pfit[k]) > np.nanmin(pfit[k]) * 1.2:
        plt.ylim(pfit[k].mean() * np.array([0.9, 1.1]))
    date_form = matplotlib.dates.DateFormatter("%Y")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9, rotation=90)

    ylims = ylims + 0.1 * np.array([-1, 1]) * (ylims.max() - ylims.min())
    plt.ylim(ylims)
    if k=='p_mp_ref':
        plt.legend(loc=[0, 3.2],
                   fontsize=8)

    # for y in [df.index.]:
    #     # plt.plot([y,y], [pfit[k].min()*scale, pfit[k].max()*scale] ,'--')
    #     plt.axvline(y,'--')
    # mask = np.logical_and(df.index.month == 1, df.index.day == 1)
    # day_ticks = np.arange(len(df))[mask]
    # plt.xticks(ticks=df.index[day_ticks].year,
    #            labels=df.index[day_ticks].year)

    n = n + 1
# figure.tight_layout(pad=5)
plt.show()

plt.savefig(
    '{}/synth02_degradation_{}.pdf'.format(save_figs_directory,
                                           pvp.system_name),
    bbox_inches='tight')
