"""
Example full run of pv-pro analysis using synthetic data.

@author: toddkarin
"""

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pvpro import PvProHandler


# Import synthetic data
df = pd.read_pickle('synth01_out.pkl')

# Make PvProHandler object to store data.
pvp = PvProHandler(df,
                   system_name='synthetic',
                   delta_T=3,
                   days_per_run=60,
                   time_step_between_iterations_days=60,
                   use_clear_times=True,
                   irradiance_lower_lim=0.1,
                   temperature_cell_upper_lim=500,
                   cells_in_series=60,
                   alpha_isc=0.001,
                   voltage_key='v_operation',
                   current_key='i_operation',
                   temperature_module_key='temperature_module_meas',
                   irradiance_poa_key='poa_meas',
                   modules_per_string=1,
                   parallel_strings=1,
                   solver='L-BFGS-B',
                   start_point_method='last',
                   )

# Preprocess
pvp.simultation_setup()
pvp.run_preprocess()

# Find clear times
pvp.find_clear_times(smoothness_hyperparam=1000)

# Inspect clear time detection.
pvp.dh.plot_daily_signals(boolean_mask=pvp.dh.boolean_masks.clear_times,
                          start_day=400)
plt.show()


# Plot startpoint on top of data.
iteration = 0
pvp.plot_Vmp_Imp_scatter(p_plot=pvp.p0,
                         figure_number=4,
                         iteration=0)
plt.title('Startpoint')

# Check execution on first iteration
ret = pvp.execute(iteration=[0],
                  verbose=False,
                  method='minimize',
                  save_figs_directory='figures/synth02')

print('Best fit:')
print(pvp.result['p'].loc[iteration, :])


iteration = 0
pvp.plot_Vmp_Imp_scatter(p_plot=pvp.result['p'].loc[iteration, :],
                         figure_number=5,
                         iteration=iteration)
plt.title('Best fit')
pvp.plot_suns_voc_scatter(p_plot=pvp.result['p'].loc[iteration, :],
                          figure_number=3,
                          iteration=iteration)
plt.title('Best fit')

# plt.savefig('figures/synth02_MPP-scatter_{}'.format(pvp.system_name,
#                                                                 iteration),
#             bbox_inches='tight')

# Run on all iterations.
ret = pvp.execute(iteration='all',
                  verbose=False,
                  method='minimize',
                  save_figs_directory='figures/synth02')

# Make degradation plots.

n = 1
figure = plt.figure(21, figsize=(10, 7))
plt.clf()

for k in pvp.result['p'].keys():

    plt.subplot(2, 3, n)
    plt.plot(pd.Series(pvp.time), pvp.result['p'][k])

    if k in df.keys():
        plt.plot(df.index, df[k], 'k--')
    plt.ylabel(k)
    # plt.gca().fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')

    date_form = matplotlib.dates.DateFormatter("%Y")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=90)
    # mask = np.logical_and(df.index.month == 1, df.index.day == 1)
    # day_ticks = np.arange(len(df))[mask]
    # plt.xticks(ticks=df.index[day_ticks].year,
    #            labels=df.index[day_ticks].year)

    n = n + 1
figure.tight_layout(pad=2.0)
plt.show()

# plt.savefig(
#     'figures/synth02_degradation_{}.png'.format(pvp.system_name),
#     bbox_inches='tight')
