import numpy as np
import pandas as pd
import warnings

from array import array

from tqdm import tqdm

import matplotlib.pyplot as plt
from pvpro.main import pvlib_single_diode
from rdtools.degradation import degradation_year_on_year


def analyze_yoy(pfit : 'dataframe'):
    out = {}

    for k in ['photocurrent_ref', 'saturation_current_ref',
              'resistance_series_ref', 'resistance_shunt_ref',
              'conductance_shunt_extra', 'diode_factor', 'i_sc_ref',
              'v_oc_ref', 'i_mp_ref', 'v_mp_ref', 'p_mp_ref', 'v_mp_ref_est',
              'i_mp_ref_est', 'p_mp_ref_est',
              'nNsVth_ref']:
        if k in pfit:
            Rd_pct, Rd_CI, calc_info = degradation_year_on_year(pd.Series(pfit[k]),
                                                                recenter=False)
            renorm = np.nanmedian(pfit[k])
            if renorm == 0:
                renorm = np.nan

            out[k] = {
                'change_per_year': Rd_pct * 1e-2,
                'percent_per_year': Rd_pct / renorm,
                'change_per_year_CI': np.array(Rd_CI) * 1e-2,
                'percent_per_year_CI': np.array(Rd_CI) / renorm,
                'calc_info': calc_info,
                'median': np.nanmedian(pfit[k])}

    return out

def calculate_error_real(pfit : 'dataframe', df_ref : 'dataframe', nrolling : int = 1):
    keys = ['diode_factor',
            'photocurrent_ref', 'saturation_current_ref',
            'resistance_series_ref',
            'resistance_shunt_ref',
            'i_sc_ref', 'v_oc_ref',
            'i_mp_ref', 'v_mp_ref', 'p_mp_ref']

    all_error_df = pd.DataFrame(index=keys, columns=['rms', 'rms_rela', 'corr_coef'])

    for key in keys:
        para_ref = df_ref[key].rolling(nrolling).mean()
        para_pvpro = pfit[key].rolling(nrolling).mean()

        mask = np.logical_and(~np.isnan(para_pvpro), ~np.isnan(para_ref))
        corrcoef = np.corrcoef(para_pvpro[mask], para_ref[mask])
        rela_rmse = np.sqrt(np.mean((para_pvpro[mask]-para_ref[mask]) ** 2))/np.mean(para_pvpro[mask])

        all_error_df['rms_rela'][key] = rela_rmse
        all_error_df['corr_coef'][key] = corrcoef[0,1]

    return all_error_df

def calculate_error_synthetic(pfit : 'dataframe', df : 'dataframe', zero_mean : bool =False):
    dft = pd.DataFrame()

    keys = ['diode_factor',
            'photocurrent_ref', 'saturation_current_ref',
            'resistance_series_ref',
            'conductance_shunt_extra', 'resistance_shunt_ref',
            'nNsVth_ref', 'i_sc_ref', 'v_oc_ref',
            'i_mp_ref', 'v_mp_ref', 'p_mp_ref']

    for k in range(len(pfit)):
        cax = np.logical_and(df.index >= pfit['t_start'].iloc[k],
                             df.index < pfit['t_end'].iloc[k])
        dfcurr_mean = df[cax][keys].mean()

        for key in dfcurr_mean.keys():
            dft.loc[pfit['t_start'].iloc[k], key] = dfcurr_mean[key]

    all_error_df = pd.DataFrame(index=keys, columns=['rms', 'rms_rela', 'corr_coef'])
    
    for k in keys:
       
        p1 = dft[k]
        p2 = pfit[k]
        mask = ~np.isnan(p1) & ~np.isnan(p2)# remove nan value for corrcoef calculation
        
        all_error_df.loc[k, 'rms'] = np.sqrt(np.mean((p1[mask]-p2[mask]) ** 2))
        all_error_df.loc[k,'rms_rela'] = np.sqrt(np.nanmean(((p1[mask]-p2[mask])/p1[mask]) ** 2))
        all_error_df.loc[k, 'corr_coef'] = np.corrcoef(p1[mask], 
                                                  p2[mask])[0,1]

    return all_error_df

"""
Functions to plot PVPRO results
"""
def plot_results_timeseries_error(pfit : 'dataframe', 
                            df : 'dataframe' = None, 
                            yoy_result : bool =None,
                            compare : bool =None,
                            compare_label : str ='True value',
                            compare_plot_style : str ='.',
                            extra_text : str ='',
                            nrows : int =5,
                            ncols : int =2,
                            wspace : float =0.4,
                            hspace : float =0.1,
                            keys_to_plot : bool =None,
                            yoy_plot : bool  = False,
                            linestyle : str = '.',
                            figsize : tuple = (8, 9),
                            legendloc : tuple = [0.3, -1.7],
                            ncol : int = 3,
                            cal_error_synthetic : bool = False,
                            cal_error_real : bool = False,
                            xticks : bool = None,
                            nylim : bool = None):
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

def plot_scatter(x : array, y : array, c : array, 
                 boolean_mask : bool =None, 
                 figure_number : int =None,
                 vmin : float =0,
                 vmax : float =70,
                 plot_x_min : float =0,
                 plot_x_max : float =40,
                 plot_y_min : float =0,
                 plot_y_max : float =10,
                 figsize : tuple =(6.5, 3.5),
                 text_str : str ='',
                 cbar : bool =True,
                 cmap : str ='jet',
                 ylabel : str ='',
                 xlabel : str ='',
                 clabel : str =''):
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

def plot_Vmp_Imp_scatter(voltage : array, current : array, poa : array, temperature_cell : array,
                         operating_cls : array,
                         boolean_mask : bool =None,
                         p_plot : 'dataframe' =None,
                         vmin : float =0,
                         vmax : float =70,
                         plot_imp_max : float =8,
                         plot_vmp_max : float =40,
                         figsize : tuple =(6.5, 3.5),
                         cbar : bool =True,
                         text_str : str ='',
                         ylabel : str ='Current (A)',
                         xlabel : str ='Voltage (V)'):
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

def plot_poa_Imp_scatter(current : array , poa : array , temperature_cell : array ,
                         operating_cls : array ,
                         voltage : array =None,
                         boolean_mask : array =None,
                         vmin : float =0,
                         vmax : float =70,
                         plot_poa_max : float =1200,
                         plot_imp_max : float =10,
                         figsize : tuple =(6.5, 3.5),
                         cbar : bool =True,
                         text_str : str ='',
                         ylabel : str ='Current (A)',
                         xlabel : str ='POA (W/m2^2)'):
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