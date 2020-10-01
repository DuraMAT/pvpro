"""
Example running quick estimate algorithm on synthetic data.

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
import seaborn as sns
from pvpro import PvProHandler
import pvpro
from pvlib.pvsystem import singlediode

# Import synthetic data
df = pd.read_pickle('synth01_out.pkl')

save_figs_directory = 'figures/synth03'

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
pvp.run_preprocess()

ret = pvp.quick_parameter_extraction(freq='M',
                                     verbose=False,
                                     figure=True
                                     )

# print(ret['p'])

pfit = ret['p']

n = 2
figure = plt.figure(0, figsize=(7.5, 5.5))
plt.clf()

figure.subplots(nrows=4, ncols=3, sharex='all')
# plt.subplots(sharex='all')
plt.subplots_adjust(wspace=0.6, hspace=0.1)

# df['conductance_shunt_ref'] = 1/df['resistance_shunt_ref']
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
         'QUICK ESTIMATE\n' + \
         'System: {}\n'.format(pvp.system_name) + \
         'Use clear times: {}\n'.format(pvp.use_clear_times) + \
         'Temp: {}\n'.format(pvp.temperature_module_key) + \
         'Irrad: {}\n'.format(pvp.irradiance_poa_key)
         , fontsize=8
         )

for k in ['diode_factor', 'photocurrent_ref', 'saturation_current_ref',
          'resistance_series_ref',
          'conductance_shunt_extra', 'i_mp_ref',
          'v_mp_ref',
          'p_mp_ref', 'i_sc_ref', 'v_oc_ref', ]:

    ax = plt.subplot(4, 3, n)

    if k == 'saturation_current_ref':
        scale = 1e9
    elif k == 'residual':
        scale = 1e3
    else:
        scale = 1

    plt.plot(pfit.index, pfit[k] * scale, '.',
             color=[0, 0, 0.8],
             label='pvpro')
    ylims = scale * np.array([np.nanmax(pfit[k]), np.nanmin(pfit[k])])

    if k in df.keys():
        plt.plot(df.index, df[k] * scale, '--',
                 color=[1, 0.2, 0.2],
                 label=True)
        ylims[0] = np.min([ylims[0], df[k].min() * scale])
        ylims[1] = np.max([ylims[1], df[k].max() * scale])

    plt.ylabel(ylabel[k], fontsize=9)
    # plt.gca().fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')

    if np.nanmax(pfit[k]) < np.nanmin(pfit[k])*1.2:
        plt.ylim(pfit[k].mean() * np.array([0.9, 1.1]))
    else:
        ylims = ylims + 0.1 * np.array([-1, 1]) * (ylims.max() - ylims.min())
        plt.ylim(ylims)
    date_form = matplotlib.dates.DateFormatter("%Y")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9, rotation=90)


    if n == 3:
        plt.legend(loc=[0, 1.2])

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

save_figs = False

if save_figs:
    plt.savefig(
        '{}/synth01_quickfit_degradation_{}.pdf'.format(save_figs_directory,
                                                        pvp.system_name),
        bbox_inches='tight')

    for f in range(20,25):
        plt.figure(f)
        plt.savefig(
            '{}/synth03_estimate_{}.pdf'.format(save_figs_directory,
                                                            f),
            bbox_inches='tight')


