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
from pvpro.singlediode import pvlib_single_diode
from sklearn.linear_model import LinearRegression


def plot_results_timeseries(pfit, yoy_result=None,
                            compare=None,
                            compare_label='True Values',
                            compare_plot_style='.',
                            extra_text='',
                            nrows=5,
                            ncols=2,
                            wspace=0.4,
                            hspace=0.1,
                            keys_to_plot=None,
                            yoy_CI = False,
                            plot_est=True):
    n = 1
    figure = plt.figure(21, figsize=(7, 6))

    figure.subplots(nrows=nrows, ncols=ncols, sharex=True)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    ylabel = {'diode_factor': 'Diode factor',
              'photocurrent_ref': 'Iph (A)',
              'saturation_current_ref': 'I0 (A)',
              'resistance_series_ref': 'Rs (Ω)',
              'resistance_shunt_ref': 'Rsh (Ω)',
              'conductance_shunt_ref': 'G shunt ref (1/Ω)',
              'conductance_shunt_extra': 'G shunt extra (1/kΩ)',
              'i_sc_ref': 'Isc (A)',
              'v_oc_ref': 'Voc (V)',
              'i_mp_ref': 'Imp (A)',
              'p_mp_ref': 'Pmp (W)',
              'i_x_ref': 'I x ref (A)',
              'v_mp_ref': 'Vmp (V)',
              'residual': 'Residual (AU)',
              }

    if keys_to_plot is None:
        keys_to_plot = ['i_mp_ref', 'photocurrent_ref',
                        'v_mp_ref', 'saturation_current_ref',
                        'i_sc_ref', 'diode_factor',
                        'v_oc_ref', 'resistance_series_ref',
                        'p_mp_ref', 'resistance_shunt_ref'
                        ]

    for k in keys_to_plot:
        if k in pfit:

            ax = plt.subplot(5, 2, n)

            if n == 1:
                plt.text(0, 1.1, extra_text
                         , fontsize=8,
                         transform=ax.transAxes)

            if k == 'saturation_current_ref':
                scale = 1e0
            elif k == 'residual':
                scale = 1e3
            elif k == 'conductance_shunt_extra':
                scale = 1e3

            else:
                scale = 1

            plt.plot(pfit['t_years'], pfit[k] * scale, '.',
                     color=[0, 0, 0.8],
                     label='pvpro')

            ylims = scale * np.array(
                [np.nanmin(pfit[k]), np.nanmax(pfit[k])]
            )
            if compare is not None:
                if k in compare:
                    plt.plot(compare['t_years'], compare[k] * scale,
                             compare_plot_style,
                             color=[0.8, 0, 0.8, 0.5],
                             label=compare_label,
                             )

                    ylims[0] = np.min([ylims[0], np.nanmin(compare[k]) * scale])
                    ylims[1] = np.max([ylims[1], np.nanmax(compare[k]) * scale])

            v_mp_med = np.median(pfit['v_mp_ref'])
            i_mp_med = np.median(pfit['i_mp_ref'])
            p_mp_med = np.median(pfit['p_mp_ref'])

            if k in ['diode_factor']:
                ylims[0] = np.nanmin([0.7, ylims[0]])
                ylims[1] = np.nanmax([1.5, ylims[1]])

            if k in ['saturation_current_ref']:
                ylims = ylims * np.array([0.5, 1.5])
            else:
                ylims = ylims + 0.1 * np.array([-1, 1]) * (ylims[1] - ylims[0])

            plt.ylim(ylims)

            if k not in ['residual' ] and yoy_result is not None:
                t_smooth = np.linspace(pfit['t_years'].min(),
                                       pfit['t_years'].max(),
                                       20)
                t_mean = np.mean(pfit['t_years'])

                # linear fitted yoy trend
                x = pfit['t_years'].to_numpy().reshape(-1,1)
                y = pfit[k].to_numpy().reshape(-1,1)
                inx = ~np.isnan(y)
                x = x[inx].reshape(-1,1)
                y = y[inx].reshape(-1,1)
                reg = LinearRegression().fit(x ,y)
                rate_esti = reg.coef_[0][0]/y[0]*100
                yoy_result[k]['percent_per_year'] = rate_esti[0]

                
                plt.plot(t_smooth,
                         scale * pfit[k].median() * (1 + (t_smooth - t_mean) * (
                                 yoy_result[k]['percent_per_year'] * 1e-2)),
                         color=[1, 0.5, 0, 0.8],
                         label='YOY trend')
                
                if yoy_CI:
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

                    plt.text(0.5, 0.1, '{:.2f}%/yr\n{:.2f} to {:.2f}%/yr'.format(
                    yoy_result[k]['percent_per_year'],
                    yoy_result[k]['percent_per_year_CI'][0],
                    yoy_result[k]['percent_per_year_CI'][1]),
                         transform=plt.gca().transAxes,
                         backgroundcolor=[1, 1, 1, 0.6],
                         fontsize=8)
                else:
                    plt.text(0.1, 0.1, 'Rate: {:.2f}%/yr'.format(
                    yoy_result[k]['percent_per_year']),
                         transform=plt.gca().transAxes,
                         backgroundcolor=[1, 1, 1, 0.6],
                         fontsize=8)

            plt.xticks(fontsize=8, rotation=45)
            plt.yticks(fontsize=8)
            plt.ylabel(ylabel[k], fontsize=8, fontweight = 'bold')

            if n == 2:
                plt.legend(loc=[0, 1.2])

            n = n + 1

def plot_results_timeseries_error(pfit, df = None, yoy_result=None,
                            compare=None,
                            compare_label='True value',
                            compare_plot_style='.',
                            extra_text='',
                            nrows=5,
                            ncols=2,
                            wspace=0.4,
                            hspace=0.1,
                            keys_to_plot=None,
                            plot_est=True,
                            yoy_plot = False,
                            linestyle = '.',
                            figsize = (8, 9),
                            legendloc = [0.3, -1.7],
                            ncol = 3,
                            cal_error_synthetic = False,
                            cal_error_real = False,
                            xticks = None,
                            nylim = None):
    n = 1
    figure = plt.figure(21, figsize=figsize)

    warnings.filterwarnings("ignore")

    figure.subplots(nrows=nrows, ncols=ncols, sharex=True)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    ylabel = {'diode_factor': 'Diode factor',
              'photocurrent_ref': 'Iph (A)',
              'saturation_current_ref': 'I0 (A)',
              'resistance_series_ref': 'Rs (Ω)',
              'resistance_shunt_ref': 'Rsh (Ω)',
              'conductance_shunt_ref': 'G shunt ref (1/Ω)',
              'conductance_shunt_extra': 'G shunt extra (1/kΩ)',
              'i_sc_ref': 'Isc (A)',
              'v_oc_ref': 'Voc (V)',
              'i_mp_ref': 'Imp (A)',
              'p_mp_ref': 'Pmp (W)',
              'i_x_ref': 'I x ref (A)',
              'v_mp_ref': 'Vmp (V)',
              'residual': 'Residual (AU)',
              }

    if keys_to_plot is None:
        keys_to_plot = ['i_mp_ref', 'photocurrent_ref',
                        'v_mp_ref', 'saturation_current_ref',
                        'i_sc_ref', 'diode_factor',
                        'v_oc_ref', 'resistance_series_ref',
                        'p_mp_ref', 'resistance_shunt_ref'
                        ]

    for k in keys_to_plot:
        if k in pfit:
            ax = plt.subplot(nrows, 2, n)

            if k == 'saturation_current_ref':
                scale = 1e0
            elif k == 'residual':
                scale = 1e3
            elif k == 'conductance_shunt_extra':
                scale = 1e3

            else:
                scale = 1

            x_show = pfit['t_years']- pfit['t_years'][0]
            plt.plot(x_show, pfit[k] * scale, linestyle,
                     color=np.array([6,86,178])/256,
                     label='PVPRO')

            ylims = scale * np.array(
                [np.nanmin(pfit[k]), np.nanmax(pfit[k])]
            )
            if compare is not None:
                x_real_show = compare['t_years']-compare['t_years'][0]
                if k in compare:
                    plt.plot(x_real_show, compare[k] * scale,
                             linestyle,
                            color = 'deepskyblue',
                             label=compare_label,
                             )

                    ylims[0] = np.min([ylims[0], np.nanmin(compare[k]) * scale])
                    ylims[1] = np.max([ylims[1], np.nanmax(compare[k]) * scale])

            v_mp_med = np.median(pfit['v_mp_ref'])
            i_mp_med = np.median(pfit['i_mp_ref'])
            p_mp_med = np.median(pfit['p_mp_ref'])

            if k in ['diode_factor']:
                ylims[0] = np.nanmin([1.08, ylims[0]])
                ylims[1] = np.nanmax([1.11, ylims[1]])
                if nylim:
                    ylims[0] = nylim[0]
                    ylims[1] = nylim[1]

            if k in ['saturation_current_ref']:
                ylims = ylims * np.array([0.5, 1.5])
            else:
                ylims = ylims + 0.1 * np.array([-1, 1]) * (ylims[1] - ylims[0])

            plt.ylim(ylims)

            # calculate error 
            from pvpro.postprocess import calculate_error_synthetic
            from pvpro.postprocess import calculate_error_real

            Nrolling = 5
            error_df = np.NaN
            if cal_error_synthetic:
                error_df = calculate_error_synthetic(pfit,df,Nrolling)
                error_df.loc['diode_factor', 'corr_coef'] = 1
            
            if cal_error_real:
                error_df = calculate_error_real(pfit, compare)
                error_df.loc['diode_factor', 'corr_coef'] = 0

            if k not in ['residual' ] and yoy_result is not None and yoy_plot:
                t_smooth = np.linspace(x_show.min(),
                                       x_show.max(),
                                       20)
                t_mean = np.mean(x_show)
                plt.plot(t_smooth,
                         scale * pfit[k].median() * (1 + (t_smooth - t_mean) * (
                                 yoy_result[k]['percent_per_year'] * 1e-2)),'--',
                        linewidth = 2,
                         color='darkorange', 
                         label='YOY trend of PVPRO')
                hori = 'left'
                posi = [0.02,0.04]

                if cal_error_synthetic | cal_error_real:
                    plt.text(posi[0], posi[1], 'RMSE: {:.2f}%\
                                        \nCorr_coef: {:.2f}\
                                        \nRate: {:.2f}%/yr\
                                        '.\
                            format(
                                error_df['rms_rela'][k]*100,
                                error_df['corr_coef'][k],
                            yoy_result[k]['percent_per_year'],
                            ),
                            transform=plt.gca().transAxes,
                            backgroundcolor=[1, 1, 1, 0],
                            fontsize=9,
                            horizontalalignment = hori)
                else:
                    plt.text(posi[0], posi[1], 'Rate: {:.2f}%/yr'.\
                            format(
                            yoy_result[k]['percent_per_year']
                            ),
                            transform=plt.gca().transAxes,
                            backgroundcolor=[1, 1, 1, 0],
                            fontsize=9,
                            horizontalalignment = hori)
            if xticks:
                plt.xticks(np.arange(4), labels= xticks, fontsize=10, rotation=45)
            else:
                plt.xticks(fontsize=10, rotation=0) , 
                if n in [nrows*2-1, nrows*2]:   
                    plt.xlabel('Year', fontsize=10)

            plt.yticks(fontsize=10)
            plt.ylabel(ylabel[k], fontsize=10, fontweight='bold')

            if n == nrows*2-1:
                plt.legend(loc=legendloc, ncol = ncol, fontsize=10)

            n = n + 1
    return error_df

def plot_scatter(x, y, c, boolean_mask=None, figure_number=None,
                 vmin=0,
                 vmax=70,
                 plot_x_min=0,
                 plot_x_max=40,
                 plot_y_min=0,
                 plot_y_max=10,
                 figsize=(6.5, 3.5),
                 text_str='',
                 cbar=True,
                 cmap='jet',
                 ylabel='',
                 xlabel='',
                 clabel=''):
    """
    Make Vmp, Imp scatter plot.

    Parameters
    ----------
    p_plot
    figure_number
    iteration
    vmin
    vmax

    Returns
    -------

    """

    # if figure_number is not None:
    # Make figure for inverter on.
    # fig = plt.figure(figure_number, figsize=figsize)
    # plt.clf()
    # ax = plt.axes()

    # df = self.get_df_for_iteration(iteration,
    #                                use_clear_times=use_clear_times)
    if boolean_mask is None:
        boolean_mask = np.ones_like(x, dtype='bool')

    if len(x[boolean_mask]) > 0:

        # Make scatterplot
        h_sc = plt.scatter(x[boolean_mask], y[boolean_mask],
                           c=c[boolean_mask],
                           s=0.2,
                           cmap=cmap,
                           vmin=vmin,
                           vmax=vmax)

        if cbar:
            pcbar = plt.colorbar(h_sc)
            pcbar.set_label(clabel)

    plt.text(0.05, 0.95, text_str,
             horizontalalignment='left',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=8)

    plt.xlim([plot_x_min, plot_x_max])
    plt.ylim([plot_y_min, plot_y_max])
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel(xlabel, fontsize=9)
    plt.ylabel(ylabel, fontsize=9)

    return h_sc


def plot_Vmp_Imp_scatter(voltage, current, poa, temperature_cell,
                         operating_cls,
                         boolean_mask=None,
                         p_plot=None,
                         vmin=0,
                         vmax=70,
                         plot_imp_max=8,
                         plot_vmp_max=40,
                         figsize=(6.5, 3.5),
                         cbar=True,
                         text_str='',
                         ylabel='Current (A)',
                         xlabel='Voltage (V)'):
    """
    Make Vmp, Imp scatter plot.

    Parameters
    ----------
    p_plot
    figure_number
    iteration
    vmin
    vmax

    Returns
    -------

    """

    # if figure_number is not None:
    # Make figure for inverter on.
    # fig = plt.figure(figure_number, figsize=figsize)
    # plt.clf()
    # ax = plt.axes()

    if boolean_mask is None:
        boolean_mask = operating_cls == 0
    else:
        boolean_mask = np.logical_and(operating_cls == 0, boolean_mask)

    h_sc = plot_scatter(
        x=voltage,
        y=current,
        c=temperature_cell,
        boolean_mask=boolean_mask,
        vmin=vmin,
        vmax=vmax,
        plot_x_max=plot_vmp_max,
        plot_y_max=plot_imp_max,
        text_str=text_str,
        cbar=cbar,
        xlabel=xlabel,
        ylabel=ylabel
    )

    temp_limits = np.linspace(vmin, vmax, 8)

    if p_plot is not None:

        # Plot temperature scan
        temperature_smooth = np.linspace(vmin, vmax, 20)

        for effective_irradiance in [100, 1000]:
            out = pvlib_single_diode(
                effective_irradiance=np.array([effective_irradiance]),
                temperature_cell=temperature_smooth,
                **p_plot)

            voltage_plot = out['v_mp']
            current_plot = out['i_mp']
            plt.plot(voltage_plot, current_plot, 'k:')

            plt.text(voltage_plot[-1] - 0.5, current_plot[-1],
                     '{:.1g} sun'.format(effective_irradiance / 1000),
                     horizontalalignment='right',
                     verticalalignment='center',
                     fontsize=8)

        # Plot irradiance scan
        for j in np.flip(np.arange(len(temp_limits))):
            temp_curr = temp_limits[j]
            irrad_smooth = np.linspace(1, 1000, 500)

            out = pvlib_single_diode(
                effective_irradiance=irrad_smooth,
                temperature_cell=temp_curr + np.zeros_like(irrad_smooth),
                **p_plot)

            voltage_plot = out['v_mp']
            current_plot = out['i_mp']

            # find the right color to plot.
            norm_temp = (temp_curr - vmin) / (vmax - vmin)
            line_color = np.array(h_sc.cmap(norm_temp))

            line_color[3] = 0.3

            plt.plot(voltage_plot, current_plot,
                     label='Fit {:2.0f} C'.format(temp_curr),
                     color=line_color,
                     )


    return h_sc


def plot_poa_Imp_scatter(current, poa, temperature_cell,
                         operating_cls,
                         voltage=None,
                         boolean_mask=None,
                         vmin=0,
                         vmax=70,
                         plot_poa_max=1200,
                         plot_imp_max=10,
                         figsize=(6.5, 3.5),
                         cbar=True,
                         text_str='',
                         ylabel='Current (A)',
                         xlabel='POA (W/m2^2)'):
    """
    Make Vmp, Imp scatter plot.

    Parameters
    ----------
    p_plot
    figure_number
    iteration
    vmin
    vmax

    Returns
    -------

    """

    if boolean_mask is None:
        boolean_mask = operating_cls == 0
    else:
        boolean_mask = np.logical_and(operating_cls == 0, boolean_mask)

    h_sc = plot_scatter(
        x=poa,
        y=current,
        c=temperature_cell,
        boolean_mask=boolean_mask,
        vmin=vmin,
        vmax=vmax,
        plot_x_max=plot_poa_max,
        plot_y_max=plot_imp_max,
        text_str=text_str,
        cbar=cbar,
        xlabel=xlabel,
        ylabel=ylabel
    )

    return h_sc