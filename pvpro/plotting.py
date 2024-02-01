
import numpy as np
import pandas as pd
import warnings

import seaborn as sns
from array import array
import matplotlib.dates as mdates

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error
from pvpro.modeling import pvlib_single_diode, pv_system_single_diode_model, single_diode_predict
from pvpro.main import calculate_error_real, calculate_error_synthetic, calc_err
import matplotlib.dates as mdates
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.ticker import FuncFormatter


    
"""
Functions to plot pre-processing results
"""

def plot_operating_condition(df):

    """
    Plot heatmap of the operating conditions

    :param df: processed data containing the 'operating_cls'
    :return : matplotlib.figure.Figure

    """

    matplotlib.rcParams['font.family'] = 'Arial' 

    m, day_axis = make_2d(df, key = 'operating_cls')

    year_start = day_axis[0].year

    def format_fn(tick_val, tick_pos):
        return (tick_pos-1)*yeargap + year_start

    # function to calculate the position at a percentage in a range
    def get_position_at_percentage(ymax, ymin, percentage):
        yrange = ymax-ymin
        y = ymin + percentage*yrange
        return y

    _, ax = plt.subplots(figsize=[6,4])

    # Build colormap
    colors = sns.color_palette("Paired")[:5]
    n_bins = 5  # Discretizes the interpolation into bins
    cmap_name = 'my_map'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


    im = ax.imshow(m, aspect='auto', interpolation='none', cmap=cmap)

    length_year = day_axis[-1].year - day_axis[0].year
    label_rotation = 0
    if length_year > 4:
        label_rotation = 30

    length_year = df.index[-1].year - df.index[0].year
    yeargap = 1
    label_rotation = 0
    if length_year > 4:
        label_rotation = 45
        yeargap = 2

    ax.set_xticks(np.arange(len(day_axis)))
    ax.set_xticklabels(day_axis, rotation=label_rotation)
    ax.xaxis.set_major_locator(YearLocator(yeargap))
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))

    ymax,ymin = ax.get_ylim()
    yposi = get_position_at_percentage(ymax, ymin, np.array([0.13, 0.4, 0.72]))
    ax.set_yticks(yposi)
    ax.set_yticklabels(['(Sunrise)', 'Time of day', '(Sunset)'], rotation=90)
    ax.tick_params(labelsize=11)

    ax.set_title('Operating conditions of PV system', fontweight='bold', fontsize = 13)

    cbar = plt.colorbar(im)
    cbar.set_ticks([-1.6, -0.8, 0, 0.8, 1.6])  # Set specific ticks for the color bar
    cbar.set_ticklabels(['Anomaly','Inverter off','MPP', 'Open circuit', 'Clipped'], fontsize = 11)

    plt.gcf().set_dpi(120)

def make_2d(df, key="power_dc", trim_start=True, trim_end=True, return_day_axis=False):
    
    """
    This function constructs a 2D array (or matrix) from a time series signal with a standardized time axis. The data is
    chunked into days, and each consecutive day becomes a column of the matrix.

    :param df: A pandas data frame contained tabular data with a standardized time axis.
    :param key: The key corresponding to the column in the data frame contained the signal to make into a matrix
    :return: A 2D numpy array with shape (measurements per day, days in data set)

    """
    
    days = df.resample("D").first().index
    try:
        freq_delta_seconds = df.index.freq.delta.seconds

    except AttributeError:
        # No frequency defined for index. Attempt to infer
        freq_ns = np.median(df.index[1:] - df.index[:-1])
        freq_delta_seconds = int(freq_ns / np.timedelta64(1, "s"))
    
    n_steps = int(24 * 60 * 60 / freq_delta_seconds)
    df_resampled = df.asfreq('{}S'.format(freq_delta_seconds), method='ffill')
    
    if not trim_start:
        start = days[0].strftime("%Y-%m-%d")
    else:
        start = days[1].strftime("%Y-%m-%d")
    if not trim_end:
        end = days[-1].strftime("%Y-%m-%d")
    else:
        end = days[-2].strftime("%Y-%m-%d")

    D = np.copy(df_resampled[key].loc[start:end].values.reshape(n_steps, -1, order="F"))

    day_axis = pd.date_range(start=start, end=end, freq="1D")
    
    return D, day_axis

def plot_dc_power(df : pd.DataFrame):

    """
    Plot DC power over time

    :param df: processed data
    :return : matplotlib.figure.Figure
    
    """

    _, ax = plt.subplots(figsize=[6,4])

    ax.scatter(df.index, df['power_dc'], s =10, alpha = 0.5, color ='#FFA222', edgecolors='None')

    length_year = df.index[-1].year - df.index[0].year
    yeargap = 1
    label_rotation = 0
    if length_year > 4:
        label_rotation = 45
        yeargap = 2

    h_fmt = mdates.DateFormatter('%Y')
    xloc = mdates.YearLocator(yeargap)
    ax.xaxis.set_major_locator(xloc)
    ax.xaxis.set_major_formatter(h_fmt)
    ax.grid()

    ax.tick_params(axis='x', rotation=label_rotation)
    ax.tick_params(labelsize=11)
    ax.set_title('DC power of PV system', fontsize=13, fontweight = 'bold')
    ax.set_ylabel('DC power (W)', fontsize=12)

    plt.gcf().set_dpi(120)


"""
Functions to plot PVPRO results
"""

def plot_results_timeseries(pfit : pd.DataFrame, 
                            df : pd.DataFrame = None, 
                            yoy_result : dict = None,
                            compare : pd.DataFrame = None,
                            yoy_plot : bool = True,
                            legendloc : list[float] = [0.7, -0.4],
                            ncol : int = 3,
                            cal_error_synthetic : bool = False,
                            cal_error_real : bool = False,
                            show_CI : bool = False):

    """
    Plot trend of extracted parameters vs time.

    : param pfit: dataframe containing extracted SDM parameters
    : param df : 
    : param yoy_result : year-of-year (YOY) degradation trend
    : param compare : Reference data to compare
    : param yoy_plot : plot yoy results
    : param legendloc : location of legend
    : param ncol : number of colunms of legend
    : param cal_error_synthetic : calculate error when using synthetic data
    : param cal_error_real : calculate error when using field data
    : param show_CI : show confidence interval (CI) of YOY trend 

    : return : matplotlib.figure.Figure

    """
    
    matplotlib.rcParams['font.family'] = 'Arial' 
    
    ylabel = {'diode_factor': 'Diode factor',
            'photocurrent_ref': 'Iph_ref (A)',
            'saturation_current_ref': 'I0_ref (A)',
            'resistance_series_ref': 'Rs_ref (Ω)',
            'resistance_shunt_ref': 'Rsh_ref (Ω)',
            'i_sc_ref': 'Isc_ref (A)',
            'v_oc_ref': 'Voc_ref (V)',
            'i_mp_ref': 'Imp_ref (A)',
            'p_mp_ref': 'Pmp_ref (W)',
            'v_mp_ref': 'Vmp_ref (V)'
            }
    
    keys = ['Isc', 'Voc', 'Imp', 'Vmp', 'Pmp', 'Iph', 'I0', 'n', 'Rs', 'Rsh']

    keys_to_plot = [
            'i_sc_ref', 'v_oc_ref', 'i_mp_ref', 'v_mp_ref', 'p_mp_ref',
            'photocurrent_ref','saturation_current_ref', 'diode_factor',
        'resistance_series_ref', 'resistance_shunt_ref']

    warnings.filterwarnings("ignore")

    _, ax_all = plt.subplots(2, 5, figsize=[18,6])
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    for key_id in range(0, 10):

        para_show = keys[key_id]

        if key_id < 5:
            ax = ax_all[0,key_id]
        else:
            ax = ax_all[1,key_id-5]

        k = keys_to_plot[key_id]

        if k in pfit:

            x_show = pfit['t_years']
            ax.scatter(x_show, pfit[k], 20,
                    color='#00A0FF', alpha = 0.8, edgecolor = 'None',
                    label='PVPRO')

            ylims = np.array([np.nanmin(pfit[k]), np.nanmax(pfit[k])])

            if compare is not None:
                x_real_show = compare['t_years']
                if k in compare:
                    ax.plot(x_real_show, compare[k],
                            '.',
                            color = 'deepskyblue',
                            label = 'True value',
                            )

                    ylims[0] = np.min([ylims[0], np.nanmin(compare[k])])
                    ylims[1] = np.max([ylims[1], np.nanmax(compare[k])])

            if k in ['diode_factor']:
                ylims[0] = np.nanmin([0.9, ylims[0]])
                ylims[1] = np.nanmax([1.2, ylims[1]])

            if k in ['saturation_current_ref']:
                ylims = ylims * np.array([0.5, 1.5])
            else:
                ylims = ylims + 0.1 * np.array([-1, 1]) * (ylims[1] - ylims[0])

            ax.set_ylim(ylims)

            # calculate error 
            
            Nrolling = 1
            error_df = np.NaN
            if cal_error_synthetic:
                error_df = calculate_error_synthetic(pfit,df,Nrolling)

            if cal_error_real:
                error_df = calculate_error_real(pfit, compare)

            if (yoy_result is not None) and yoy_plot:
                t_smooth = np.linspace(x_show.min(), x_show.max(), 20)
                t_mean = np.mean(x_show)

                ax.plot(t_smooth,
                        pfit[k].median() * (1 + (t_smooth - t_mean) * (
                        yoy_result[k]['percent_per_year'] * 1e-2)),'--',
                        linewidth = 4,
                        color='#F8891D',
                        label = 'YOY trend')
                
                if show_CI & (k != 'diode_factor'):
                    ax.plot(t_smooth,
                            pfit[k].median() * (1 + (t_smooth - t_mean) * (
                            yoy_result['photocurrent_ref']['percent_per_year_CI'][0] * 1e-2)),'--',
                            linewidth = 4,
                            color='#FF973C', alpha=0.5)
                    ax.plot(t_smooth,
                            pfit[k].median() * (1 + (t_smooth - t_mean) * (
                            yoy_result['photocurrent_ref']['percent_per_year_CI'][1] * 1e-2)),'--',
                            linewidth = 4,
                            color='#FF973C', alpha=0.5, label = 'CI of YOY trend')
                
                hori = 'left'
                posi = [0.02,0.04]

                if cal_error_synthetic | cal_error_real:

                    # No calculation of diode factor
                    error_df.loc['diode_factor', 'corr_coef'] = 1

                    ax.text(posi[0], posi[1], 'RMSE: {:.2f}%\
                                        \nCorr_coef: {:.2f}\
                                        \nRate: {:.2f}%/yr\
                                        '.\
                            format(
                                error_df['rms_rela'][k]*100,
                                error_df['corr_coef'][k],
                            yoy_result[k]['percent_per_year']),
                            transform=ax.transAxes,
                            backgroundcolor=[1, 1, 1, 0],
                            fontsize=11,
                            horizontalalignment = hori)
                else:
                    ax.text(posi[0], posi[1], 'Rate: {:.2f}%/yr'.\
                            format(yoy_result[k]['percent_per_year']),
                            transform=ax.transAxes,
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                            fontsize=11,
                            fontweight='bold',
                            horizontalalignment = hori)
                    
            length_year = x_show.max() - x_show.min()
            label_rotation = 0
            if length_year > 4:
                label_rotation = 30
                    
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.tick_params(axis='y', labelsize=11)
            ax.tick_params(axis='x', rotation=label_rotation)
            ax.set_ylabel(ylabel[k], fontsize=12)
            ax.set_title('{}_ref'.format(para_show),weight='bold',fontsize = 13)
            ax.grid()

            if key_id == 6:
                ax.legend(loc=legendloc, ncol = ncol, fontsize=13)

    # plt.tight_layout()
    plt.gcf().set_dpi(150)


"""
Function to plot post-processed results
"""

def plot_post_processed_results(df_post, model = 'smooth_monotonic'):

    """
    Plot trends (decomposed periodic trend, degrdation trend) of post-processed  parameters vs time.

    : param df_post: dataframe of post-processed data
    : param model : model of analysis, default is 'smooth_monotonic'

    : return : matplotlib.figure.Figure

    """

    matplotlib.rcParams['font.family'] = 'Arial' 

    keys = ['Isc', 'Voc', 'Imp', 'Vmp', 'Pmp', 'Iph', 'I0', 'n', 'Rs', 'Rsh']
    keys_pvpro = ['i_sc_ref', 'v_oc_ref', 'i_mp_ref', 'v_mp_ref', 'p_mp_ref', 
                  'photocurrent_ref','saturation_current_ref',
                  'diode_factor', 'resistance_series_ref', 'resistance_shunt_ref']
    ylabels = {'diode_factor': 'n',
            'photocurrent_ref': 'Iph_ref (A)',
            'saturation_current_ref': 'I0_ref (A)',
            'resistance_series_ref': 'Rs_ref (Ω)',
            'resistance_shunt_ref': 'Rsh_ref (Ω)',
            'i_sc_ref': 'Isc_ref (A)',
            'v_oc_ref': 'Voc_ref (V)',
            'i_mp_ref': 'Imp_ref (A)',
            'p_mp_ref': 'Pmp_ref (W)',
            'v_mp_ref': 'Vmp_ref (V)'
            }

    _, ax_all = plt.subplots(2, 5, figsize=[18,6])
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    for key_id in range(0, 10):

        if key_id < 5:
            ax = ax_all[0,key_id]
        else:
            ax = ax_all[1,key_id-5]
        
        para_show = keys[key_id]
        para_pvpro = keys_pvpro[key_id]

        #### PVPRO ####
        x_pvpro = df_post.df.index
        y_pvpro = df_post.df[para_pvpro]

        ylim = [0,1000]

        yscale = 'linear'
        if para_show == 'Rs':
            ylim = [0,1]
        elif para_show == 'Iph':
            ylim = [0,10]
        elif para_show == 'Rsh':
            ylim = [200,800]
        elif para_show == 'n':
            ylim = [0,3]

        mask_y = (y_pvpro>ylim[0]) & (y_pvpro<ylim[1])
        
        ax.scatter(x_pvpro[mask_y],y_pvpro[mask_y].rolling(1).mean(), 30, label = 'PV-Pro',
                       color = '#00A0FF', alpha = 0.6, edgecolor = 'None')

        ### post-processed ###
        if para_show != 'n':
            key_post = para_pvpro + "_" + model
            if key_post in df_post.descaled_data:
                datashow = df_post.descaled_data[key_post]
                x = datashow.index
                ycom = datashow['composed_signal']
                ytrend = datashow['x5']*datashow['composed_signal'][0]
                yperi = datashow['x4']*datashow['x3'][0]

                Nrolling = 1

                ax.plot(x,ycom.rolling(Nrolling).mean(), '-', linewidth = 3, label = 'Composed',
                            color = '#0070C0')
                ax.plot(x,yperi.rolling(Nrolling).mean(), '-', linewidth = 3, label = 'Periodic',
                            color = '#FFC000')
                ax.plot(x,ytrend.rolling(Nrolling).mean(), '-', linewidth = 3, label = 'Degra trend',
                            color = '#F2529D')
    
        ax.set_yscale(yscale)
        ax.set_ylabel(ylabels[para_pvpro], fontsize = 13)

        length_year = x[-1].year- x[0].year
        label_rotation = 0
        yeargap = 1
        if length_year > 4:
            label_rotation = 60
            yeargap = 2
        
        date_form = DateFormatter("%Y")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(mdates.YearLocator(yeargap))

        ax.tick_params(labelsize=12)
        ax.tick_params(axis='x', rotation=label_rotation)

        if para_show in ['Iph', 'Imp']:
            ax.legend( markerscale=1, prop={'size': 13}, framealpha=0.5, loc = 3)
        ax.set_title('{}_ref'.format(para_show),weight='bold',fontsize = 15)
        ax.grid(True)
        plt.tight_layout()
        plt.gcf().set_dpi(150)


"""
Functions to plot irradiance-to-power conversion results

"""

def plot_predicted_ref_power(y_predicted : list, 
                             y_ref : list, 
                             nominal_power : float):

    """
    Plot predicted and reference power

    :param y_predicted: predicted power
    :param y_ref: reference power
    :param nominal_power: nominal power of the PV system
    
    """

    err = calc_err(y_predicted, y_ref, nominal_power)
    matplotlib.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(figsize = [7,4], dpi = 120)
    ax.plot(y_ref.index, y_ref/1000, '--', linewidth = 3, label = 'Ref')
    ax.plot(y_ref.index, y_predicted/1000, linewidth = 3, label = 'PV-Pro')
    ax.text(0.15, 0.7, 
            'nMAE: {:.2f}%\nnRMSE: {:.2f}%'.format(err['nMAE'], err['nRMSE']),
            transform=fig.transFigure, fontsize = 20, fontweight = 'bold')
    ax.set_ylabel('Power (kW)', fontsize = 18)
    ax.set_xlabel('Time', fontsize = 18)
    h_fmt = mdates.DateFormatter('%Hh')
    ax.xaxis.set_major_formatter(h_fmt)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.set_title('{}'.format(y_ref.index[0].date()), fontweight = 'bold', fontsize = 20)

    ax.legend(fontsize = 18)
    ax.grid(True)