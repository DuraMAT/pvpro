import pvlib
import numpy as np
import pandas as pd
# import pytz

import datetime
import os
import warnings
import time
from tqdm import tqdm

import matplotlib.pyplot as plt


def plot_results_timeseries(pfit, yoy_result=None,
                            extra_text='',
                            nrows=5,
                            ncols=2,
                            wspace=0.4,
                            hspace=0.1,
                            keys_to_plot=None):
    n = 1
    figure = plt.figure(21, figsize=(7, 6))
    # plt.clf()

    figure.subplots(nrows=nrows, ncols=ncols, sharex=True)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    ylabel = {'diode_factor': 'Diode factor',
              'photocurrent_ref': 'Photocurrent ref (A)',
              'saturation_current_ref': 'I sat ref (nA)',
              'resistance_series_ref': 'R series ref (Ω)',
              'resistance_shunt_ref': 'R shunt ref (Ω)',
              'conductance_shunt_ref': 'G shunt ref (1/Ω)',
              'conductance_shunt_extra': 'G shunt extra (1/kΩ)',
              'i_sc_ref': 'I sc ref (A)',
              'v_oc_ref': 'V oc ref (V)',
              'i_mp_ref': 'I mp ref (A)',
              'p_mp_ref': 'P mp ref (W)',
              'i_x_ref': 'I x ref (A)',
              'v_mp_ref': 'V mp ref (V)',
              'residual': 'Residual (AU)',
              }

    if keys_to_plot is None:
        keys_to_plot = ['photocurrent_ref',
                        'i_mp_ref',
                        'diode_factor',
                        'v_mp_ref',
                        'saturation_current_ref', 'p_mp_ref',
                        'resistance_series_ref',
                        'v_oc_ref',
                        'conductance_shunt_extra',
                        'residual',
                        ]

    for k in keys_to_plot:

        ax = plt.subplot(5, 2, n)

        if n == 1:
            plt.text(0, 1.1, extra_text
                     , fontsize=8,
                     transform=ax.transAxes)

        if k == 'saturation_current_ref':
            scale = 1e9
        elif k == 'residual':
            scale = 1e3
        elif k == 'conductance_shunt_extra':
            scale = 1e3

        else:
            scale = 1

        plt.plot(pfit['t_years'], pfit[k] * scale, '.',
                 color=[0, 0, 0.8],
                 label='pvpro')
        ylims = scale * np.array([pfit[k].min(), pfit[k].max()])

        #     if k in df.keys():
        #         plt.plot(df.index, df[k] * scale, '--',
        #                  color=[1, 0.2, 0.2],
        #                  label=True)
        #         ylims[0] = np.min([ylims[0], df[k].min() * scale])
        #         ylims[1] = np.max([ylims[1], df[k].max() * scale])

        if k in ['i_mp_ref', 'v_mp_ref', 'p_mp_ref']:
            plt.plot(pfit['t_years'], pfit[k + '_est'], '.',
                     color=[0, 0.8, 0.8],
                     label='pvpro-fast')

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel(ylabel[k], fontsize=8)

        # plt.gca().fmt_xdata = matplotlib.dates.DateFormatter('%Y-%m-%d')

        #     if (pfit[k].max() - pfit[k].min())/pfit[k].mean() < 1.2:
        #         plt.ylim(pfit[k].mean() * np.array([0.9, 1.1]))

        v_mp_med = np.median(pfit['v_mp_ref'])
        i_mp_med = np.median(pfit['i_mp_ref'])
        p_mp_med = np.median(pfit['p_mp_ref'])

        if k in ['conductance_shunt_extra']:
            #         1% power loss
            G_shunt_crit = 0.01 * p_mp_med / v_mp_med ** 2
            plt.plot(pfit['t_years'],
                     np.zeros(len(pfit['t_years'])) + G_shunt_crit * scale,
                     '--')
            plt.text(pfit['t_years'][0], G_shunt_crit * scale * 1.05,
                     '1% power loss', fontsize=7)
            ylims[1] = np.max([ylims[1], G_shunt_crit * scale * 1.1])
        if k in ['resistance_series_ref']:
            R_series_crit = 0.10 * v_mp_med / i_mp_med
            plt.plot(pfit['t_years'],
                     np.zeros(len(pfit['t_years'])) + R_series_crit * scale,
                     '--')
            print(R_series_crit)
            plt.text(pfit['t_years'][0], R_series_crit * scale * 1.05,
                     '10% power loss', fontsize=7)
            ylims[1] = np.max([ylims[1], R_series_crit * scale * 1.1])

        #     plt.plot()
        # date_form = matplotlib.dates.DateFormatter("%Y")
        # plt.gca().xaxis.set_major_formatter(date_form)
        plt.yticks(fontsize=9)
        plt.xticks(fontsize=9, rotation=90)

        ylims = ylims + 0.1 * np.array([-1, 1]) * (ylims.max() - ylims.min())
        plt.ylim(ylims)

        if k not in ['residual'] and yoy_result is not None:
            t_smooth = np.linspace(pfit['t_years'].min(), pfit['t_years'].max(),
                                   2)
            t_mean = np.mean(pfit['t_years'])

            plt.plot(t_smooth,
                     scale * pfit[k].median() * (1 + (t_smooth - t_mean) * (
                                 yoy_result[k]['percent_per_year_CI'][
                                     0] * 1e-2)),
                     color=[1, 0.5, 0, 0.3])
            plt.plot(t_smooth,
                     scale * pfit[k].median() * (1 + (t_smooth - t_mean) * (
                                 yoy_result[k]['percent_per_year_CI'][
                                     1] * 1e-2)),
                     color=[1, 0.5, 0, 0.3])
            plt.plot(t_smooth,
                     scale * pfit[k].median() * (1 + (t_smooth - t_mean) * (
                                 yoy_result[k]['percent_per_year'] * 1e-2)),
                     color=[1, 0.5, 0, 0.8],
                     label='YOY trend')
            plt.text(0.5, 0.1, '{:.2f}%/yr\n{:.2f} to {:.2f}%/yr'.format(
                yoy_result[k]['percent_per_year'],
                yoy_result[k]['percent_per_year_CI'][0],
                yoy_result[k]['percent_per_year_CI'][1]),
                     transform=plt.gca().transAxes,
                     backgroundcolor=[1, 1, 1, 0.6],
                     fontsize=8)

        if n == 2:
            plt.legend(loc=[0, 1.2])

        n = n + 1

# def plot_voltage_current_scatter(current,
#                          voltage,
#                          temperature_cell,
#                          irradiance_poa,
#                          operating_cls=None,
#                          p_plot=None,
#                       system_name='',
#                          figure_number=None,
#                          filter_mpp=True,
#                          vmin=0,
#                          vmax=70,
#                          plot_imp_max=8,
#                          plot_vmp_max=40,
#                          figsize=(6.5, 3.5)):
#     """
#     Make Vmp, Imp scatter plot.
#
#     Parameters
#     ----------
#     p_plot
#     figure_number
#     iteration
#     vmin
#     vmax
#
#     Returns
#     -------
#
#     """
#
#     # if figure_number is not None:
#     # Make figure for inverter on.
#     fig = plt.figure(figure_number, figsize=figsize)
#     # plt.clf()
#     # ax = plt.axes()
#
#     temp_limits = np.linspace(vmin, vmax, 8)
#
#
#     if operating_cls is None:
#         operating_cls = np.zeros_like(current).astype('int')
#
#     if filter_mpp:
#         mpp = operating_cls==0
#         voltage = np.array(voltage[mpp])
#         current = np.array(current[mpp])
#         temperature_cell = temperature_cell[mpp]
#
#     # Make scatterplot
#     h_sc = plt.scatter(voltage, current,
#                        c=temperature_cell,
#                        s=0.2,
#                        cmap='jet',
#                        vmin=0,
#                        vmax=70)
#
#     if p_plot is not None:
#         # Plot one sun
#         one_sun_points = np.logical_and.reduce((
#             operating_cls == 0,
#             irradiance_poa > 995,
#             irradiance_poa < 1005,
#         ))
#         if len(one_sun_points) > 0:
#             # print('number one sun points: ', len(one_sun_points))
#             plt.scatter(voltage[one_sun_points],
#                         current[one_sun_points],
#                         c='k',
#                         edgecolors='k',
#                         s=0.2)
#
#         # Plot temperature scan
#         temperature_smooth = np.linspace(0, 70, 20)
#
#         for effective_irradiance in [100, 1000]:
#             voltage_plot, current_plot = pv_system_single_diode_model(
#                 effective_irradiance=np.array([effective_irradiance]),
#                 temperature_cell=temperature_smooth,
#                 operating_cls=np.zeros_like(temperature_smooth),
#                 cells_in_series=p_plot['cells_in_series'],
#                 alpha_isc=p_plot['alpha_isc'],
#                 resistance_shunt_ref=p_plot['resistance_shunt_ref'],
#                 diode_factor=p_plot['diode_factor'],
#                 photocurrent_ref=p_plot['photocurrent_ref'],
#                 saturation_current_ref=p_plot['saturation_current_ref'],
#                 resistance_series_ref=p_plot['resistance_series_ref'],
#                 conductance_shunt_extra=p_plot['conductance_shunt_extra']
#             )
#
#             plt.plot(voltage_plot, current_plot, 'k:')
#
#             plt.text(voltage_plot[-1] - 0.5, current_plot[-1],
#                      '{:.1g} sun'.format(effective_irradiance / 1000),
#                      horizontalalignment='right',
#                      verticalalignment='center',
#                      fontsize=8)
#
#         # Plot irradiance scan
#         for j in np.flip(np.arange(len(temp_limits))):
#             temp_curr = temp_limits[j]
#             irrad_smooth = np.linspace(1, 1000, 500)
#
#             voltage_plot, current_plot = pv_system_single_diode_model(
#                 effective_irradiance=irrad_smooth,
#                 temperature_cell=temp_curr + np.zeros_like(irrad_smooth),
#                 operating_cls=np.zeros_like(irrad_smooth),
#                 cells_in_series=p_plot['cells_in_series'],
#                 alpha_isc=p_plot['alpha_isc'],
#                 resistance_shunt_ref=p_plot['resistance_shunt_ref'],
#                 diode_factor=p_plot['diode_factor'],
#                 photocurrent_ref=p_plot['photocurrent_ref'],
#                 saturation_current_ref=p_plot['saturation_current_ref'],
#                 resistance_series_ref=p_plot['resistance_series_ref'],
#                 conductance_shunt_extra=p_plot['conductance_shunt_extra']
#             )
#
#             # out = pvlib_fit_fun( np.transpose(np.array(
#             #     [irrad_smooth,temp_curr + np.zeros_like(irrad_smooth), np.zeros_like(irrad_smooth) ])),
#             #                     *p_plot)
#
#             # Reshape to get V, I
#             # out = np.reshape(out,(2,int(len(out)/2)))
#
#             # find the right color to plot.
#             # norm_temp = (temp_curr-df[temperature].min())/(df[temperature].max()-df[temperature].min())
#             norm_temp = (temp_curr - vmin) / (vmax - vmin)
#             line_color = np.array(h_sc.cmap(norm_temp))
#             # line_color[0:3] =line_color[0:3]*0.9
#
#             line_color[3] = 0.3
#
#             plt.plot(voltage_plot, current_plot,
#                      label='Fit {:2.0f} C'.format(temp_curr),
#                      color=line_color,
#                      # color='C' + str(j)
#                      )
#
#         text_str = 'System: {}\n'.format(system_name) + \
#                    'Start: {}\n'.format(
#                        df.index[0].strftime("%m/%d/%Y, %H:%M:%S")) + \
#                    'End: {}\n'.format(
#                        df.index[-1].strftime("%m/%d/%Y, %H:%M:%S")) + \
#                    'Current: {}\n'.format(self.current_key) + \
#                    'Voltage: {}\n'.format(self.voltage_key) + \
#                    'Temperature: {}\n'.format(self.temperature_module_key) + \
#                    'Irradiance: {}\n'.format(self.irradiance_poa_key) + \
#                    'Temperature module->cell delta_T: {}\n'.format(
#                        self.delta_T) + \
#                    'n_diode: {:1.2f} \n'.format(p_plot['diode_factor']) + \
#                    'reference_photocurrent: {:1.2f} A\n'.format(
#                        p_plot['photocurrent_ref']) + \
#                    'saturation_current_ref: {:1.2f} nA\n'.format(
#                        p_plot['saturation_current_ref'] * 1e9) + \
#                    'resistance_series: {:1.2f} Ohm\n'.format(
#                        p_plot['resistance_series_ref']) + \
#                    'Conductance shunt extra: {:1.2f} 1/Ohm\n\n'.format(
#                        p_plot['conductance_shunt_extra'])
#     else:
#         text_str = 'System: {}\n'.format(self.system_name) + \
#                    'Start: {}\n'.format(
#                        df.index[0].strftime("%m/%d/%Y, %H:%M:%S")) + \
#                    'End: {}\n'.format(
#                        df.index[-1].strftime("%m/%d/%Y, %H:%M:%S")) + \
#                    'Current: {}\n'.format(self.current_key) + \
#                    'Voltage: {}\n'.format(self.voltage_key) + \
#                    'Temperature: {}\n'.format(self.temperature_module_key) + \
#                    'Irradiance: {}\n'.format(self.irradiance_poa_key)
#
#     plt.text(0.05, 0.95, text_str,
#              horizontalalignment='left',
#              verticalalignment='top',
#              transform=plt.gca().transAxes,
#              fontsize=8)
#
#     plt.xlim([0, plot_vmp_max])
#     plt.ylim([0, plot_imp_max])
#     plt.xticks(fontsize=9)
#     plt.yticks(fontsize=9)
#     pcbar = plt.colorbar(h_sc)
#     pcbar.set_label('Cell Temperature (C)')
#
#     plt.xlabel('Vmp (V)', fontsize=9)
#     plt.ylabel('Imp (A)', fontsize=9)
#
#     plt.show()
#
#     return fig
