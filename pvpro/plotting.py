
import numpy as np
import pandas as pd
import warnings

from pandas.core.frame import DataFrame

import seaborn as sns
from array import array

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error
from pvpro.main import pvlib_single_diode, pv_system_single_diode_model
from pvpro.main import calculate_error_real, calculate_error_synthetic
from pvpro.main import single_diode_predict

class PvProPlot:

    def __init__(self,
                 system_name : str ='Unknown'):
            self.name = system_name
    
    """
    Functions to plot pre-processing results
    """

    def plot_scatter(self, x : array, y : array, c : array, 
                    boolean_mask : bool =None, 
                    vmin : float =0,
                    vmax : float =70,
                    plot_x_min : float =0,
                    plot_x_max : float =40,
                    plot_y_min : float =0,
                    plot_y_max : float =10,
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

    def plot_Vmp_Imp_scatter_preprocess(self, voltage : pd.Series, 
                            current : pd.Series, 
                            temperature_cell : pd.Series,
                            operating_cls : pd.Series,
                            boolean_mask : np.ndarray =None,
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

        h_sc = self.plot_scatter(
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

    def plot_temperature_Vmp_scatter(self, df : pd.DataFrame,
                                    p_plot : bool =None,
                                    figure_number : bool =None,
                                    vmin : float =0,
                                    vmax : float =1200,
                                    plot_imp_max : float =8,
                                    plot_vmp_min : float =20,
                                    plot_vmp_max : float =45,
                                    plot_temperature_min : float =-10,
                                    plot_temperature_max : float =70,
                                    figsize : tuple =(6.5, 3.5),
                                    cmap : str ='viridis',
                                    cbar : bool =True):
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
        fig = plt.figure(figure_number, figsize=figsize)

        irrad_limits = np.linspace(vmin, vmax, 8)

        if len(df) > 0:
            inv_on_points = np.array(df['operating_cls'] == 0)

            vmp = np.array(
                df.loc[
                    inv_on_points, self.voltage_key]) / self.modules_per_string
            imp = np.array(
                df.loc[inv_on_points, self.current_key]) / self.parallel_strings

            # Make scatterplot
            h_sc = plt.scatter(df.loc[inv_on_points, 'temperature_cell'], vmp,
                                c=df.loc[inv_on_points, self.irradiance_poa_key],
                                s=0.2,
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax)

        if p_plot is not None:

            Ee_to_plot = [250, 500, 1000, 1200]
            temperature_smooth = np.linspace(plot_temperature_min,
                                                plot_temperature_max, 20)
            # Plot irradiance scan
            for j in np.flip(np.arange(len(Ee_to_plot))):
                effective_irradiance_curr = Ee_to_plot[j]

                voltage_plot, current_plot = self.single_diode_predict(
                    effective_irradiance=effective_irradiance_curr,
                    temperature_cell=temperature_smooth,
                    operating_cls=np.zeros_like(temperature_smooth),
                    params=p_plot)

                # find the right color to plot.
                norm_temp = (effective_irradiance_curr - vmin) / (vmax - vmin)
                line_color = np.array(h_sc.cmap(norm_temp))

                line_color[3] = 0.3

                plt.plot(temperature_smooth, voltage_plot,
                            # label='Fit {:2.0f} C'.format(temp_curr),
                            color=line_color,
                            )

        text_str = self.build_plot_text_str(df, p_plot=p_plot)

        plt.text(0.05, 0.95, text_str,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=plt.gca().transAxes,
                    fontsize=8)

        plt.xlim([plot_temperature_min, plot_temperature_max])
        plt.ylim([plot_vmp_min, plot_vmp_max])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        if cbar and len(df) > 0:
            pcbar = plt.colorbar(h_sc)
            pcbar.set_label('Irradiance (W/m^2)')

        plt.xlabel('Cell Temperature (C)', fontsize=9)
        plt.ylabel('Voltage (V)', fontsize=9)

        plt.show()

        return fig

    def plot_poa_Imp_scatter(self, current : pd.Series, 
                            poa : pd.Series, 
                            temperature_cell : pd.Series,
                            operating_cls : pd.Series,
                            boolean_mask : np.ndarray =None,
                            vmin : float =0,
                            vmax : float =70,
                            plot_poa_max : float =1200,
                            plot_imp_max : float =10,
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

        h_sc = self.plot_scatter(
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

    def plot_suns_voc_scatter(self, df : pd.DataFrame,
                                p_plot : dict,
                                figure_number : int =2,
                                vmin : float =0,
                                vmax : float =70,
                                plot_voc_max : float =45.,
                                cbar : bool =True):
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

        # Make figure for inverter on.

        temp_limits = np.linspace(vmin, vmax, 8)
        if len(df) > 0:
            inv_off_points = np.array(df['operating_cls'] == 1)

            voc = np.array(
                df.loc[
                    inv_off_points, self.voltage_key]) / self.modules_per_string
            irrad = np.array(df.loc[inv_off_points, self.irradiance_poa_key])

            h_sc = plt.scatter(voc, irrad,
                                c=df.loc[inv_off_points, 'temperature_cell'],
                                s=0.2,
                                cmap='jet',
                                vmin=0,
                                vmax=70)

        # Plot temperature scan
        temperature_smooth = np.linspace(0, 70, 20)

        # Plot irradiance scan
        irrad_smooth = np.linspace(1 ** 0.1, 1200 ** 0.1, 300) ** 10
        for j in np.flip(np.arange(len(temp_limits))):
            temp_curr = temp_limits[j]

            voltage_plot, current_plot = pv_system_single_diode_model(
                effective_irradiance=irrad_smooth,
                temperature_cell=temp_curr + np.zeros_like(irrad_smooth),
                operating_cls=np.zeros_like(irrad_smooth) + 1,
                cells_in_series=self.cells_in_series,
                alpha_isc=self.alpha_isc,
                resistance_shunt_ref=self.resistance_shunt_ref,
                diode_factor=p_plot['diode_factor'],
                photocurrent_ref=p_plot['photocurrent_ref'],
                saturation_current_ref=p_plot['saturation_current_ref'],
                resistance_series_ref=p_plot['resistance_series_ref'],
                conductance_shunt_extra=p_plot['conductance_shunt_extra']
            )

            norm_temp = (temp_curr - vmin) / (vmax - vmin)
            line_color = np.array(h_sc.cmap(norm_temp))
            line_color[3] = 0.3

            plt.plot(voltage_plot, irrad_smooth,
                        label='Fit {:2.0f} C'.format(temp_curr),
                        color=line_color
                        )
        text_str = self.build_plot_text_str(df, p_plot=p_plot)

        plt.text(0.05, 0.95, text_str,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=plt.gca().transAxes,
                    fontsize=8)

        plt.xlim([0, plot_voc_max])
        plt.yscale('log')
        plt.ylim([1, 1200])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        if cbar and len(df) > 0:
            pcbar = plt.colorbar(h_sc)
            pcbar.set_label('Cell Temperature (C)')

        plt.xlabel('Voc (V)', fontsize=9)
        plt.ylabel('POA (W/m^2)', fontsize=9)

    def plot_current_irradiance_clipped_scatter(self, df : pd.DataFrame,
                                                p_plot : dict,
                                                figure_number : int =1,
                                                vmin : float =0,
                                                vmax : float =70,
                                                plot_imp_max : float =8,
                                                cbar : bool =True):
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

        temp_limits = np.linspace(vmin, vmax, 8)
        if len(df) > 0:
            cax = np.array(df['operating_cls'] == 2)

            current = np.array(
                df.loc[cax, self.current_key]) / self.parallel_strings

            irrad = np.array(df.loc[cax, self.irradiance_poa_key])

            h_sc = plt.scatter(irrad, current,
                                c=df.loc[cax, 'temperature_cell'],
                                s=0.2,
                                cmap='jet',
                                vmin=0,
                                vmax=70)

        text_str = self.build_plot_text_str(df, p_plot=p_plot)

        plt.text(0.05, 0.95, text_str,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=plt.gca().transAxes,
                    fontsize=8)

        plt.ylim([0, plot_imp_max])
        plt.xlim([0, 1200])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        if cbar and len(df) > 0:
            pcbar = plt.colorbar(h_sc)
            pcbar.set_label('Cell Temperature (C)')

        plt.ylabel('Current (A)', fontsize=9)
        plt.xlabel('POA (W/m^2)', fontsize=9)

    def plot_current_irradiance_mpp_scatter(self, pvp,
                                            df : pd.Series,
                                            p_plot : dict =None,
                                            vmin : float =0,
                                            vmax : float =70,
                                            plot_imp_max : float =8,
                                            cbar : bool =True):
        # Make figure for inverter on.

        temp_limits = np.linspace(vmin, vmax, 8)

        if len(df) > 0:
            cax = np.array(df['operating_cls'] == 0)

            current = np.array(
                df.loc[cax, pvp.current_key]) / pvp.parallel_strings

            irrad = np.array(df.loc[cax, pvp.irradiance_poa_key])
            h_sc = plt.scatter(irrad, current,
                                c=df.loc[cax, 'temperature_cell'],
                                s=0.2,
                                cmap='jet',
                                vmin=0,
                                vmax=70)

        if p_plot is not None:
            # Plot irradiance scan
            for j in np.flip(np.arange(len(temp_limits))):
                temp_curr = temp_limits[j]
                irrad_smooth = np.linspace(1, 1200, 500)

                voltage_plot, current_plot = pv_system_single_diode_model(
                    effective_irradiance=irrad_smooth,
                    temperature_cell=temp_curr + np.zeros_like(irrad_smooth),
                    operating_cls=np.zeros_like(irrad_smooth) + 0,
                    cells_in_series=pvp.cells_in_series,
                    alpha_isc=pvp.alpha_isc,
                    resistance_shunt_ref=p_plot['resistance_shunt_ref'],
                    diode_factor=p_plot['diode_factor'],
                    photocurrent_ref=p_plot['photocurrent_ref'],
                    saturation_current_ref=p_plot['saturation_current_ref'],
                    resistance_series_ref=p_plot['resistance_series_ref'],
                    conductance_shunt_extra=p_plot['conductance_shunt_extra'],
                    band_gap_ref = pvp.Eg_ref,
                    dEgdT = pvp.dEgdT

                )

                norm_temp = (temp_curr - vmin) / (vmax - vmin)
                line_color = np.array(h_sc.cmap(norm_temp))

                line_color[3] = 0.3

                plt.plot(irrad_smooth, current_plot,
                            label='Fit {:2.0f} C'.format(temp_curr),
                            color=line_color,
                            # color='C' + str(j)
                            )

        text_str = pvp.build_plot_text_str(df, p_plot=p_plot)

        plt.text(0.05, 0.95, text_str,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=plt.gca().transAxes,
                    fontsize=8)

        plt.ylim([0, plot_imp_max])
        plt.xlim([0, 1200])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        if cbar and len(df) > 0:
            pcbar = plt.colorbar(h_sc)
            pcbar.set_label('Cell Temperature (C)')

        plt.ylabel('Current (A)', fontsize=9)
        plt.xlabel('POA (W/m^2)', fontsize=9)

    def plot_temperature_rise_irradiance_scatter(self, df : pd.DataFrame,
                                                p_plot : dict,
                                                vmin : float =0,
                                                vmax : float =70,
                                                cbar : bool =True):
            """


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

            temp_limits = np.linspace(vmin, vmax, 8)

            cax = np.array(df['operating_cls'] == 0)
            if len(df) > 0:
                irrad = np.array(df.loc[cax, self.irradiance_poa_key])
                Trise = np.array(df.loc[cax, self.temperature_module_key] - df.loc[
                    cax, self.temperature_ambient_key])

                h_sc = plt.scatter(irrad, Trise,
                                c=df.loc[cax, 'temperature_cell'],
                                s=0.2,
                                cmap='jet',
                                vmin=0,
                                vmax=70)

            text_str = self.build_plot_text_str(df, p_plot=p_plot)

            plt.text(0.05, 0.95, text_str,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=plt.gca().transAxes,
                    fontsize=8)

            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            if cbar and len(df) > 0:
                pcbar = plt.colorbar(h_sc)
                pcbar.set_label('Cell Temperature (C)')

            plt.ylabel('T module-T ambient (C)', fontsize=9)
            plt.xlabel('POA (W/m^2)', fontsize=9)

    def plot_operating_cls(self, extra_matrices, figsize : tuple =(12, 6)):

        if not 'operating_cls' in extra_matrices:
            raise Exception("""Must call 'run_preprocess_sdt' first to use 
            this visualization.""")
            
        # function to calculate the position at a percentage in a range
        def get_position_at_percentage(ymax, ymin, percentage):
            yrange = ymax-ymin
            y = ymin + percentage*yrange
            return y

        fig = plt.figure(figsize=figsize)

        # Build colormap
        colors = sns.color_palette("Paired")[:5]
        n_bins = 5  # Discretizes the interpolation into bins
        cmap_name = 'my_map'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        matrix_show = extra_matrices['operating_cls']
        fig, ax = plt.subplots(figsize = [8,5])
        plt.imshow(matrix_show, aspect='auto',
                    interpolation='none',
                    cmap=cmap,
                    vmin=-2.5, vmax=2.5)
        ax.set_xticks(np.array([0,1,2,3])*365)
        ax.set_xticklabels(['2015', '2016', '2017', '2018'])
        ax.set_xlabel('Year')
        ax.get_ylim()
        ymax,ymin = ax.get_ylim()
        yposi = get_position_at_percentage(ymax, ymin, np.array([0.13, 0.4, 0.72]))
        ax.set_yticks(yposi)
        ax.set_yticklabels(['(Sunrise)', 'Time of day', '(Sunset)'], rotation=90)
        ax.set_title('Identified PV array operating condition', weight='bold')
        cbar = plt.colorbar()
        cbar.ax.set_yticklabels(['','Anomaly','Inverter off','MPP', 'Open circuit', 'Clipped'])
        plt.rcParams['font.size'] = '13'
        plt.gcf().set_dpi(100)
        plt.show
        return fig


    """
    Functions to plot off-MPP detection results
    """

    def plot_Pmp_error_vs_time(self, pvp, boolean_mask : array, points_show : array= None, figsize : list =[4,3], 
                                    sys_name : str = None):

        """
        Plot Pmp error vs time, where the at-MPP and off-MPP points are highlighted
        
        """
        if points_show:
            points_show_bool = np.full(boolean_mask.sum(), False)
            points_show_bool[points_show] = True
            df=pvp.df[boolean_mask][points_show_bool]
        else:
            df=pvp.df[boolean_mask]

        p_plot=pvp.p0

        fig, ax = plt.subplots(figsize=figsize)

        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings

        # calculate error
        v_esti, i_esti = single_diode_predict(pvp,
            effective_irradiance=df[pvp.irradiance_poa_key][mask],
            temperature_cell=df[pvp.temperature_cell_key][mask],
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)
        rmse_vmp = mean_squared_error(v_esti, vmp)/37
        rmse_imp = mean_squared_error(i_esti, imp)/8.6

        # Pmp error
        pmp_error = abs(vmp*imp - v_esti*i_esti)
        vmp_error = abs(vmp-v_esti)
        imp_error = abs(imp-i_esti)

        # Plot At-MPP points
        ax.scatter(df.index[mask], pmp_error, s =1, color ='#8ACCF8', label = 'At-MPP')

        # detect off-mpp and calculate off-mpp percentage
        offmpp = pmp_error>np.nanmean(pmp_error)+np.std(pmp_error)
        offmpp_ratio = offmpp.sum()/pmp_error.size*100  
        plt.text(df.index[0], 240, 'Off-MPP ratio:\
                        \n{}%'.format(round(offmpp_ratio,2)),
                    fontsize=12)
        
        # Plot off-MPP points
        ax.scatter(df.index[mask][offmpp], pmp_error[offmpp], s =1, color ='#FFA222', label = 'Off-MPP')

        # plot mean Pmp error line
        ax.plot([df.index[mask][0], df.index[mask][-1]], [np.nanmean(pmp_error)]*2, 
                    '--', linewidth = 1, color='#0070C0', label = 'Mean Pmp error')

        import matplotlib.dates as mdates
        # h_fmt = mdates.DateFormatter('%y-%m')
        h_fmt = mdates.DateFormatter('%Y')
        xloc = mdates.YearLocator(1)
        ax.xaxis.set_major_locator(xloc)
        ax.xaxis.set_major_formatter(h_fmt)

        # fig.autofmt_xdate()
        plt.ylim([0, 300])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.ylabel('Pmp error (W)', fontsize=12, fontweight = 'bold')
        lgnd = plt.legend(loc = 1)
        lgnd.legendHandles[0]._sizes = [20]
        lgnd.legendHandles[1]._sizes = [20]
        plt.title(sys_name, fontsize=13, fontweight = 'bold')

        plt.gcf().set_dpi(150)
        plt.show()

    def plot_Vmp_Imp_scatters_Pmp_error(self, pvp, boolean_mask : array, points_show : array = None, figsize : list =[4,3], show_only_offmpp : bool = False, 
                                sys_name : str = None, date_show : str = None):

        """
        Plot relative error (RE) of Vmp vs RE of Imp as scatters.
        The color of scatters corresponds to the RE of Pmp.
        
        """
        if points_show:
            points_show_bool = np.full(boolean_mask.sum(), False)
            points_show_bool[points_show] = True
            df=pvp.df[boolean_mask][points_show_bool]
        else:
            df=pvp.df[boolean_mask]
        p_plot=pvp.p0
        
        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings

        # calculate error
        v_esti, i_esti = single_diode_predict(pvp,
            effective_irradiance=df[pvp.irradiance_poa_key][mask],
            temperature_cell=df[pvp.temperature_cell_key][mask],
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)
        rmse_vmp = mean_squared_error(v_esti, vmp)/37
        rmse_imp = mean_squared_error(i_esti, imp)/8.6

        # Pmp error
        pmp_error = abs(vmp*imp - v_esti*i_esti)
        vmp_error = abs(vmp-v_esti)
        imp_error = abs(imp-i_esti)

        # calculate off-mpp percentage
        msk = np.full(pmp_error.size, True)
        if show_only_offmpp:
            msk = (pmp_error>np.nanmean(pmp_error)+np.std(pmp_error) ) & (pmp_error<300)

        # plot
        fig, ax = plt.subplots(figsize=figsize)

        h_sc = plt.scatter(vmp_error[msk]/37*100, imp_error[msk]/8.6*100, cmap='jet',
                s=10,  alpha = 0.8, c=pmp_error[msk])
                            
        pcbar = plt.colorbar(h_sc)
        pcbar.set_label('Pmp error (W)', fontsize = 13)

        if not date_show:
            date_show = df.index[mask][0].strftime("%Y-%m-%d")

        text_show = sys_name + '\n' + date_show
        plt.text (35,85, text_show)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlim([0,80])
        plt.ylim([0,100])
        plt.xlabel('RE of Vmp (%)', fontsize=13)
        plt.ylabel('RE of Imp (%)', fontsize=13)
        plt.title('Distribution of off-MPP points', fontweight = 'bold', fontsize=13)
        plt.gcf().set_dpi(120)
        plt.show()

    def plot_Vmp_Tm_Imp_G_vs_time (self, pvp, boolean_mask : array, points_show : array = None, figsize : list =[5,6]):

        points_show_bool = np.full(boolean_mask.sum(), False)
        points_show_bool[points_show] = True
        df=pvp.df[boolean_mask][points_show_bool]
        p_plot = pvp.p0

        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings

        # calculate error
        v_esti, i_esti = single_diode_predict(pvp,
            effective_irradiance=df[pvp.irradiance_poa_key][mask],
            temperature_cell=df[pvp.temperature_cell_key][mask],
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)
        
        fig, ax = plt.subplots(2,1,figsize=figsize, sharex = True)

        """ plot Imp and G """

        ax11 = ax[0]
        ax12 = ax11.twinx()
        
        xtime = df.index[mask]

        ## plot G in right
        lns11 = ax11.fill_between(xtime, df[pvp.irradiance_poa_key][mask], 0, 
                        alpha=0.3, color='#FF95C2',
                        zorder = 2, label = 'G')
        ax11.yaxis.tick_right()
        ax11.yaxis.set_label_position("right")

        ## plot Imp in left
        lns12, = ax12.plot(xtime, imp, '-o', color = 'deepskyblue', zorder = 2.5, label = 'Measured Imp')
        lns13, = ax12.plot(xtime, i_esti, '--o',zorder = 3,label = 'Estimated Imp')
        ax12.yaxis.tick_left()
        ax12.yaxis.set_label_position("left")

        ax11.grid(linestyle = '--')
        ax11.tick_params(labelsize=13)
        ax12.tick_params(labelsize=13)
        ax12.set_ylabel('Imp (A)', fontsize=13, color = np.array([24,116,205])/256, fontweight = 'bold')
        ax11.set_ylabel('Irradiance (${W/m^2}$)', fontsize=13, color = '#C47398', fontweight = 'bold')
        ax11.set_title(' Vmp and Imp on {}'.format(xtime[0].strftime('%Y-%m-%d')), 
                    fontweight = 'bold', fontsize=13)

        # combine legends

        lns1 = (lns12, lns13, lns11)
        labs1 = [l.get_label() for l in lns1]
        ax12.legend(lns1, labs1, loc=1)

        """ plot Vmp and Tm """

        ax21 = ax[1]
        ax22 = ax21.twinx()
        
        xtime = df.index[mask]
        
        ## plot Tm in right
        lns21 = ax21.fill_between(xtime, df[pvp.temperature_cell_key][mask], 0, 
                        alpha=0.4, color='#FFC000', edgecolor = None,
                        zorder = 2, label = 'Tm')
        ax21.yaxis.tick_right()
        ax21.yaxis.set_label_position("right")

        ## plot Imp in left
        lns22, = ax22.plot(xtime, vmp, '-o',zorder = 2.5, label = 'Measured Vmp', color = '#92D050')
        lns23, = ax22.plot(xtime, v_esti, '--o', color= '#009847',zorder = 3,label = 'Estimated Vmp')
        ax22.yaxis.tick_left()
        ax22.yaxis.set_label_position("left")
        
        ax21.grid(linestyle = '--')
        ax21.tick_params(labelsize=13)
        ax22.tick_params(labelsize=13)
        ax21.set_xlabel('Time', fontsize=13)
        ax22.set_ylabel('Vmp (A)', fontsize=13, color = '#009847', fontweight = 'bold')
        ax21.set_ylabel('Tm (℃)', fontsize=13, color = '#D8A402', fontweight = 'bold')
        

        import matplotlib.dates as mdates
        hours = mdates.HourLocator(interval = 1)
        h_fmt = mdates.DateFormatter('%Hh')
        ax21.xaxis.set_major_locator(hours)
        ax21.xaxis.set_major_formatter(h_fmt)

        # combine legends
        lns2 = (lns22, lns23, lns21)
        labs2 = [l.get_label() for l in lns2]
        ax22.legend(lns2, labs2, loc=7)


        plt.gcf().set_dpi(120)
        plt.show()

    def plot_Vmp_vs_Tm_Imp_vs_G (self, pvp, boolean_mask : array, points_show : array = None, figsize : tuple =[4,6]):

        points_show_bool = np.full(boolean_mask.sum(), False)
        points_show_bool[points_show] = True
        df=pvp.df[boolean_mask][points_show_bool]
        p_plot = pvp.p0

        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings
        G = df[pvp.irradiance_poa_key][mask]
        Tm = df[pvp.temperature_cell_key][mask]

        # estimate
        v_esti, i_esti = single_diode_predict(pvp,
            effective_irradiance=G,
            temperature_cell=Tm,
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)

        # error
        pmp_error = abs(vmp*imp - v_esti*i_esti)
        RE_vmp = abs(vmp-v_esti)/37*100
        RE_imp = abs(imp-i_esti)/8.6*100

        fig, ax = plt.subplots(2,1,figsize=figsize)

        ax1 = ax[0]
        ax1.scatter(G, RE_imp)
        ax1.grid(linestyle = '--')
        ax1.tick_params(labelsize=13)
        ax1.tick_params(labelsize=13)
        ax1.set_ylabel('RE_Imp (%)', fontsize=13, fontweight = 'bold')
        ax1.set_xlabel('G (${W/m^2}$)', fontsize=13, fontweight = 'bold')
        ax1.set_title('RE_Imp vs G', fontweight = 'bold', fontsize=13)

        ax2 = ax[1]
        ax2.scatter(Tm, RE_vmp, color = '#009847')
        ax2.grid(linestyle = '--')
        ax2.tick_params(labelsize=13)
        ax2.tick_params(labelsize=13)
        ax2.set_ylabel('RE_Vmp (%)', fontsize=13, fontweight = 'bold')
        ax2.set_xlabel('Tm (℃)', fontsize=13, fontweight = 'bold')
        ax2.set_title(' RE_Vmp vs Tm', fontweight = 'bold', fontsize=13)

        plt.gcf().set_dpi(120)
        plt.tight_layout()
        plt.show()


    """
    Functions to plot PVPRO results
    """
    def plot_results_timeseries_error(self, pfit : DataFrame, 
                                df : pd.DataFrame = None, 
                                yoy_result : dict =None,
                                compare : DataFrame = None,
                                compare_label : str ='True value',
                                nrows : int =5,
                                ncols : int =2,
                                wspace : float =0.4,
                                hspace : float =0.1,
                                keys_to_plot : list = None,
                                yoy_plot : bool = False,
                                linestyle : str = '.',
                                figsize : tuple = (8, 9),
                                legendloc : list[float] = [0.3, -1.7],
                                ncol : int = 3,
                                cal_error_synthetic : bool = False,
                                cal_error_real : bool = False,
                                xticks : list = None,
                                nylim : bool = False ):
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
                
                Nrolling = 5
                error_df = np.NaN
                if cal_error_synthetic:
                    error_df = calculate_error_synthetic(pfit,df,Nrolling)
                    error_df.loc['diode_factor', 'corr_coef'] = 1
                
                if cal_error_real:
                    error_df = calculate_error_real(pfit, compare)
                    error_df.loc['diode_factor', 'corr_coef'] = 0

                if k not in ['residual'] and (yoy_result is not None) and yoy_plot:
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
                    plt.xticks(fontsize=10, rotation=0) 
                if n in [nrows*2-1, nrows*2]:   
                    plt.xlabel('Year', fontsize=10)

                plt.yticks(fontsize=10)
                plt.ylabel(ylabel[k], fontsize=10, fontweight='bold')

                if n == nrows*2-1:
                    plt.legend(loc=legendloc, ncol = ncol, fontsize=10)

                n = n + 1
        return error_df

    def plot_Vmp_Imp_scatter(self, pvp,
                            df : pd.DataFrame,
                            p_plot : dict,
                            vmin : float =0,
                            vmax : float =70,
                            plot_imp_max : float =8,
                            plot_vmp_max : float =40,
                            cbar : bool =True,
                            ylabel : str ='Current (A)',
                            xlabel : str ='Voltage (V)'):
        """
        Make Vmp, Imp scatter plot.

        """

        temp_limits = np.linspace(vmin, vmax, 8)

        if len(df) > 0:
            inv_on_points = np.array(df['operating_cls'] == 0)
            vmp = np.array(
                df.loc[
                    inv_on_points, pvp.voltage_key]) / pvp.modules_per_string
            imp = np.array(
                df.loc[inv_on_points, pvp.current_key]) / pvp.parallel_strings

            # Make scatterplot
            h_sc = plt.scatter(vmp, imp,
                                c=df.loc[inv_on_points, 'temperature_cell'],
                                s=0.2,
                                cmap='jet',
                                vmin=vmin,
                                vmax=vmax)

        if p_plot is not None:
            # Plot one sun
            one_sun_points = np.logical_and.reduce((
                df['operating_cls'] == 0,
                df[pvp.irradiance_poa_key] > 995,
                df[pvp.irradiance_poa_key] < 1005,
            ))
            if len(one_sun_points) > 0:
                # print('number one sun points: ', len(one_sun_points))
                plt.scatter(df.loc[
                                one_sun_points, pvp.voltage_key] / pvp.modules_per_string,
                            df.loc[
                                one_sun_points, pvp.current_key] / pvp.parallel_strings,
                            c=df.loc[one_sun_points, 'temperature_cell'],
                            edgecolors='k',
                            s=0.2)

            # Plot temperature scan
            temperature_smooth = np.linspace(vmin, vmax, 20)

            for effective_irradiance in [100, 1000]:
                voltage_plot, current_plot = single_diode_predict(pvp,
                    effective_irradiance=np.array([effective_irradiance]),
                    temperature_cell=temperature_smooth,
                    operating_cls=np.zeros_like(temperature_smooth),
                    params=p_plot)

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

                voltage_plot, current_plot = single_diode_predict(pvp,
                    effective_irradiance=irrad_smooth,
                    temperature_cell=temp_curr + np.zeros_like(irrad_smooth),
                    operating_cls=np.zeros_like(irrad_smooth),
                    params=p_plot)

                # find the right color to plot.
                norm_temp = (temp_curr - vmin) / (vmax - vmin)
                line_color = np.array(h_sc.cmap(norm_temp))

                line_color[3] = 0.3

                plt.plot(voltage_plot, current_plot,
                            label='Fit {:2.0f} C'.format(temp_curr),
                            color=line_color,
                            )
        text_str = pvp.build_plot_text_str(df, p_plot=p_plot)

        plt.text(0.05, 0.95, text_str,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=plt.gca().transAxes,
                    fontsize=8)

        plt.xlim([0, plot_vmp_max])
        plt.ylim([0, plot_imp_max])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        if cbar and len(df) > 0:
            pcbar = plt.colorbar(h_sc)
            pcbar.set_label('Cell Temperature (C)')

        plt.xlabel(xlabel, fontsize=9)
        plt.ylabel(ylabel, fontsize=9)