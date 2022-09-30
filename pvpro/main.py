from array import array
import pvlib
import numpy as np
import pandas as pd


import datetime
import os
import warnings
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
from pvpro.singlediode import estimate_Eg_dEgdT

from solardatatools import DataHandler
from pvlib.temperature import sapm_cell_from_module

from pvpro.singlediode import pv_system_single_diode_model
from pvpro.fit import production_data_curve_fit
from pvpro.classify import classify_operating_mode

from pvpro.estimate import estimate_singlediode_params, estimate_imp_ref, \
    estimate_vmp_ref

import shutil
from pvpro.preprocess import monotonic
from pvanalytics.features import clipping

from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pvpro.preprocess import find_linear_model_outliers_timeseries


class PvProHandler:
    """
    Class for running pvpro analysis.

    """

    def __init__(self,
                 df : 'dataframe',
                 system_name : str ='Unknown',
                 voltage_key : str =None,
                 current_key : str =None,
                 # power_key=None,
                 temperature_cell_key : str ='temperature_cell',
                 temperature_module_key : str =None,
                 temperature_ambient_key : str =None,
                 irradiance_poa_key : str =None,
                 modules_per_string : int =None,
                 parallel_strings : int =None,
                 alpha_isc : float =None,
                 resistance_shunt_ref : float =600,
                 # delta_T=3,
                 use_clear_times : bool =True,
                 cells_in_series : int =None,
                 technology : str =None,
                 ):

        # Initialize datahandler object.

        self.dh = DataHandler(df)

        self.df = df
        self.system_name = system_name
        self.cells_in_series = cells_in_series
        
        self.alpha_isc = alpha_isc # alpha_isc is in units of A/C.
        self.resistance_shunt_ref = resistance_shunt_ref
        self.voltage_key = voltage_key
        self.current_key = current_key
        self.temperature_cell_key = temperature_cell_key
        self.temperature_ambient_key = temperature_ambient_key
        self.irradiance_poa_key = irradiance_poa_key
        self.modules_per_string = modules_per_string
        self.parallel_strings = parallel_strings
        self.technology = technology

        self.p0 = dict(
            diode_factor=1.03,
            photocurrent_ref=4,
            saturation_current_ref=1e-11,
            resistance_series_ref=0.4,
            conductance_shunt_extra=0
        )

        # Eg and dEgdT based on PV technology
        self.Eg_ref, self.dEgdT = estimate_Eg_dEgdT(self.technology)

        # self.solver = solver

    @property
    def df(self):
        """
        Store dataframe inside the DataHandler.

        Returns
        -------
        df : dataframe
            Time-series data
        """
        return self.dh.data_frame_raw

    @df.setter
    def df(self, value):
        """
        Set Dataframe by setting the version inside datahandler.

        Parameters
        ----------
        value : dataframe
            Time-series data

        Returns
        -------

        """
        self.dh.data_frame_raw = value


    def info(self):
        """
        Print info about the class.

        Returns
        -------

        """
        keys = ['system_name', 'delta_T',
                'cells_in_series',
                'alpha_isc', 'voltage_key', 'current_key',
                'temperature_module_key',
                'irradiance_poa_key', 'modules_per_string', 'parallel_strings']

        info_display = {}
        for k in keys:
            info_display[k] = self.__dict__[k]

        print(pd.Series(info_display))
        return info_display

    def quick_parameter_extraction(self,
                                   freq : str ='W',
                                   max_iter : int =np.inf,
                                   verbose : bool =False,
                                   figure : bool =True,
                                   optimize_Rs_Io : bool=True):
        """
        Run quick parameter estimation over data.

        Parameters
        ----------
        freq : str

            pandas frequency for sectioning data.

        max_iter : numeric

            Maximum number of iterations to run the algorithm.

        verbose : bool

            Whether to print verbose output.

        figure : bool

            Whether to draw and export figures on last iteration.

        optimize_Rs_Io

            Whether to run series resistance and saturation current
            optimization.

        Returns
        -------

        out : dataframe

            Dataframe of best-fit parameters

        """
        start_time = time.time()
        out = {}

        time_bounds = pd.date_range(self.df.index[0].date(), self.df.index[-1],
                                    freq=freq)

        time_center = time_bounds[:-1] + (
                self.df.index[1] - self.df.index[0]) / 2

        df = self.df[self.df['operating_cls'] == 0]

        ret_list = []

        num_iter = int(np.min([max_iter, (len(time_bounds) - 1)]))

        for k in tqdm(range(num_iter)):

            # This is a slow way to do this.
            cax = np.logical_and.reduce((
                df.index >= time_bounds[k],
                df.index < time_bounds[k + 1],
            ))

            draw_figure = figure and k == (len(time_bounds) - 2)
            if draw_figure:
                print('drawing figures on last iteration.')

            ret, opt_result = estimate_singlediode_params(
                poa=df.loc[cax, self.irradiance_poa_key],
                temperature_cell=df.loc[cax, self.temperature_cell_key],
                vmp=df.loc[cax, self.voltage_key] / self.modules_per_string,
                imp=df.loc[cax, self.current_key] / self.parallel_strings,
                cells_in_series=self.cells_in_series,
                # delta_T=self.delta_T,
                figure=draw_figure,
                verbose=verbose,
                optimize_Rs_Io=optimize_Rs_Io
            )

            ret_list.append(ret)

        out['p'] = pd.DataFrame(ret_list, index=time_center[:num_iter])
        out['p']['t_years'] = np.array(
            [t.year + (t.dayofyear - 1) / 365.25 for t in time_center])

        out['time'] = time_center[:num_iter]

        print('Elapsed time: {:.0f} s'.format((time.time() - start_time)))
        return out

    def execute(self,
                iteration : str ='all',
                boolean_mask : bool =None,
                days_per_run : float =365.25,
                iterations_per_year : int =10,
                start_point_method : str ='fixed',
                use_mpp_points : bool =True,
                use_voc_points : bool =True,
                use_clip_points : bool =True,
                diode_factor : bool =None,
                photocurrent_ref : bool =None,
                saturation_current_ref : bool =None,
                resistance_series_ref : bool =None,
                resistance_shunt_ref : bool =None,
                conductance_shunt_extra : float =0,
                verbose : bool =False,
                method : str ='minimize',
                solver : str ='L-BFGS-B',
                save_figs : bool =True,
                save_figs_directory : str ='figures',
                plot_imp_max : float =8,
                plot_vmp_max : float =40,
                fit_params : bool =None,
                lower_bounds : bool =None,
                upper_bounds : bool =None,
                singlediode_method : str ='fast',
                saturation_current_multistart : bool =None,
                technology : bool = None 
                ):
        """
        Main PVPRO Method.


        Parameters
        ----------
        iteration
        boolean_mask


        days_per_run : float

            Number of days passed to fit in each iteration.

        time_step_between_iterations_days
        start_point_method
        use_mpp_points
        use_voc_points
        use_clip_points
        verbose
        method
        solver
        save_figs
        save_figs_directory
        plot_imp_max
        plot_vmp_max
        fit_params
        lower_bounds
        upper_bounds

        Returns
        -------

        """

        start_time = time.time()

        # Make directories for exporting figures.
        if save_figs:
            try:
                shutil.rmtree(save_figs_directory)
            except OSError as e:
                pass
                # print('Cannot remove directory {}'.format(save_figs_directory))

            os.mkdir(save_figs_directory)

            export_folders = [
                os.path.join(save_figs_directory, 'Vmp_Imp'),
                os.path.join(save_figs_directory, 'suns_Voc'),
                os.path.join(save_figs_directory, 'clipped'),
                os.path.join(save_figs_directory, 'poa_Imp'),
            ]
            for folder in export_folders:
                if not os.path.exists(folder):
                    os.mkdir(folder)

        q = 1.602e-19
        kB = 1.381e-23

        # Calculate iteration start days
        dataset_length_days = (self.df.index[-1] - self.df.index[0]).days
        iteration_start_days = np.arange(0, dataset_length_days - days_per_run,
                                         365.25 / iterations_per_year)

        # Fit params taken from p0
        if fit_params == None:
            fit_params = ['photocurrent_ref', 'saturation_current_ref',
                          'resistance_series_ref', 
                         'conductance_shunt_extra', 
                         'resistance_shunt_ref', 
                          'diode_factor']
            # self.fit_params = fit_params

        if saturation_current_multistart is None:
            saturation_current_multistart = [0.2, 0.5, 1, 2, 5]

        # Calculate time midway through each iteration.
        self.time = []
        for d in iteration_start_days:
            self.time.append(self.df.index[0] +
                             datetime.timedelta(
                                 int(d + 0.5 * days_per_run)))

        # Bounds
        lower_bounds = lower_bounds or dict(
            diode_factor=0.5,
            photocurrent_ref=0.01,
            saturation_current_ref=1e-13,
            resistance_series_ref=0,
            conductance_shunt_extra=0
        )

        upper_bounds = upper_bounds or dict(
            diode_factor=2,
            photocurrent_ref=20,
            saturation_current_ref=1e-5,
            resistance_series_ref=1,
            conductance_shunt_extra=10
        )

        # Initialize pfit dataframe for saving fit values.
        pfit = pd.DataFrame()

        # p0 contains the start point for each fit.
        p0 = pd.DataFrame(index=range(len(iteration_start_days)),
                          columns=fit_params)

        # for d in range(len(self.iteration_start_days)):
        fit_result = []

        if iteration == 'all':
            # print('Executing fit on all start days')
            iteration = np.arange(len(iteration_start_days))

        n = 0
        k_last_iteration = 0
        for k in tqdm(iteration):

            # Get df for current iteration.
            if boolean_mask is None:
                boolean_mask = np.ones(len(self.df), dtype=np.bool)
            else:
                boolean_mask = np.array(boolean_mask)

            if not len(self.df) == len(boolean_mask):
                raise ValueError(
                    'Boolean mask has length {}, it must have same length as self.df: {}'.format(
                        len(boolean_mask), len(self.df)))

            pfit.loc[k, 't_start'] = self.df.index[0] + datetime.timedelta(
                iteration_start_days[k])
            pfit.loc[k, 't_end'] = self.df.index[0] + datetime.timedelta(
                iteration_start_days[k] + days_per_run)

            idx = np.logical_and.reduce(
                (
                    self.df.index >= pfit.loc[k, 't_start'],
                    self.df.index < pfit.loc[k, 't_end'],
                    boolean_mask
                ))

            # Get a section of df for this iteration.
            df = self.df[idx]

            if len(df) > 10:

                # Can update p0 each iteration in future.
                if start_point_method.lower() == 'fixed':
                    for key in fit_params:
                        p0.loc[k, key] = self.p0[key]
                elif start_point_method.lower() == 'last':
                    if n == 0:
                        for key in fit_params:
                            p0.loc[k, key] = self.p0[key]
                    else:
                        for key in fit_params:
                            p0.loc[k, key] = p0.loc[k_last_iteration, key]
                else:
                    raise ValueError(
                        'start_point_method must be "fixed" or "last"')

                # Do quick parameters estimation on this iteration.from

                # Do pvpro fit on this iteration.
                if verbose:
                    print('p0 for fit:', p0.loc[k])
                    print('voltage for fit: ',
                          df[self.voltage_key] / self.modules_per_string)
                    print('current for fit: ',
                          df[self.current_key] / self.parallel_strings)

                fit_result_iter = production_data_curve_fit(
                    temperature_cell=np.array(df['temperature_cell']),
                    effective_irradiance=np.array(df[self.irradiance_poa_key]),
                    operating_cls=np.array(df['operating_cls']),
                    voltage=np.array(
                        df[self.voltage_key]) / self.modules_per_string,
                    current=np.array(
                        df[self.current_key]) / self.parallel_strings,
                    cells_in_series=self.cells_in_series,
                    alpha_isc=self.alpha_isc,
                    resistance_shunt_ref=resistance_shunt_ref,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    p0=p0.loc[k, fit_params],
                    diode_factor=diode_factor,
                    photocurrent_ref=photocurrent_ref,
                    saturation_current_ref=saturation_current_ref,
                    resistance_series_ref=resistance_series_ref,
                    conductance_shunt_extra=conductance_shunt_extra,
                    verbose=verbose,
                    solver=solver,
                    method=method,
                    use_mpp_points=use_mpp_points,
                    use_voc_points=use_voc_points,
                    use_clip_points=use_clip_points,
                    singlediode_method=singlediode_method,
                    saturation_current_multistart=saturation_current_multistart,
                    band_gap_ref = self.Eg_ref,
                    dEgdT = self.dEgdT
                )
                pfit_iter = fit_result_iter['p']
                pfixed_iter = fit_result_iter['fixed_params']
                residual = fit_result_iter['residual']

                # Estimate parameters quickly.
                cax = df['operating_cls'] == 0
                if np.sum(cax) > 10:

                    est, est_results = estimate_singlediode_params(
                        poa=df.loc[cax, self.irradiance_poa_key],
                        temperature_cell=df.loc[cax, 'temperature_cell'],
                        imp=df.loc[cax, self.current_key] / self.parallel_strings,
                        vmp=df.loc[cax, self.voltage_key] / self.modules_per_string,
                        vmp_model='sandia1',
                        imp_model='sandia',
                        cells_in_series = self.cells_in_series,
                        technology = technology
                        )

                    for key in est.keys():
                        pfit.loc[k, key + '_est'] = est[key]

                for p in pfit_iter: pfit.loc[k, p] = pfit_iter[p]

                for p in pfixed_iter: pfit.loc[k, p] = pfixed_iter[p]

                pfit.loc[k, 'residual'] = residual

                fit_result.append(fit_result_iter)

                if verbose:
                    print('\n--')
                    print('Fit Results:')
                    print('Final residual: {:.4f}'.format(residual))
                    print('\nStartpoint:')
                    print(p0.loc[k, fit_params])
                    print('\nBest fit:')
                    print(pfit.loc[k, fit_params])

                if save_figs:
                    plt.figure(10, figsize=(6.5, 3.5))
                    plt.clf()

                    self.plot_Vmp_Imp_scatter(df=df,
                                              p_plot=pfit_iter,
                                              figure_number=100,
                                              plot_imp_max=plot_imp_max,
                                              plot_vmp_max=plot_vmp_max,
                                              )
                    vmp_imp_fig_name = os.path.join(save_figs_directory,
                                                    'Vmp_Imp',
                                                    '{}_Vmp-Imp_{}.png'.format(
                                                        self.system_name, k))

                    if verbose:
                        print('Exporting: {}'.format(vmp_imp_fig_name))
                    plt.savefig(vmp_imp_fig_name,
                                dpi=350, )
                    plt.figure(11, figsize=(6.5, 3.5))
                    plt.clf()
                    self.plot_suns_voc_scatter(df=df,
                                               p_plot=pfit_iter,
                                               figure_number=101,
                                               plot_voc_max=plot_vmp_max * 1.1)
                    plt.savefig(os.path.join(save_figs_directory,
                                             'suns_Voc',
                                             '{}_suns-Voc_{}.png'.format(
                                                 self.system_name, k)),
                                dpi=350, )
                    plt.figure(12, figsize=(6.5, 3.5))
                    plt.clf()
                    self.plot_current_irradiance_clipped_scatter(
                        df=df,
                        p_plot=pfit_iter,
                        figure_number=103,
                        plot_imp_max=plot_imp_max)
                    plt.savefig(os.path.join(save_figs_directory,
                                             'clipped',
                                             '{}_clipped_{}.png'.format(
                                                 self.system_name, k)),
                                dpi=350, )
                    plt.figure(13, figsize=(6.5, 3.5))
                    plt.clf()
                    self.plot_current_irradiance_mpp_scatter(df=df,
                                                             p_plot=pfit_iter,
                                                             figure_number=104,
                                                             plot_imp_max=plot_imp_max
                                                             )
                    plt.savefig(os.path.join(save_figs_directory,
                                             'poa_Imp',
                                             '{}_poa-Imp_{}.png'.format(
                                                 self.system_name, k)),
                                dpi=350, )

                n = n + 1
                k_last_iteration = k
                # except:
                #     print('** Error with this iteration.')

                # Calculate other parameters vs. time.
                pfit.loc[k, 'nNsVth_ref'] = pfit.loc[
                                                k, 'diode_factor'] * self.cells_in_series * kB / q * (
                                                    25 + 273.15)

                out = pvlib.pvsystem.singlediode(
                    photocurrent=pfit.loc[k, 'photocurrent_ref'],
                    saturation_current=pfit.loc[k, 'saturation_current_ref'],
                    resistance_series=pfit.loc[k, 'resistance_series_ref'],
                    resistance_shunt = pfit.loc[k, 'resistance_shunt_ref'], # Get Rsh by estimation
                    nNsVth=pfit.loc[k, 'nNsVth_ref'])

                for p in out.keys():
                    pfit.loc[k, p + '_ref'] = out[p]

        pfit['t_mean'] = pfit['t_start'] + datetime.timedelta(
            days=days_per_run / 2)

        pfit['t_years'] = np.array(
            [t.year + (t.dayofyear - 1) / 365.25 for t in pfit['t_mean']])
        pfit.index = pfit['t_start']

        self.result = dict(
            p=pfit,
            p0=p0,
            fit_result=fit_result,
            execution_time_seconds=time.time() - start_time
        )

        print(
            'Elapsed time: {:.2f} min'.format((time.time() - start_time) / 60))

    def estimate_p0(self,
                    verbose : bool =False,
                    boolean_mask : bool =None,
                    technology : bool =None):
        """
        Make a rough estimate of the startpoint for fitting the single diode
        model.

        Returns
        -------

        """
        # print(technology)
        if boolean_mask is None:
            self.p0, result = estimate_singlediode_params(
                poa=self.df[self.irradiance_poa_key],
                temperature_cell=self.df[self.temperature_cell_key],
                vmp=self.df[self.voltage_key] / self.modules_per_string,
                imp=self.df[self.current_key] / self.parallel_strings,
                cells_in_series=self.cells_in_series,
                # delta_T=self.delta_T,
                technology=technology,
                verbose=verbose
            )
        else:
            self.p0, result = estimate_singlediode_params(
                poa=self.df.loc[boolean_mask, self.irradiance_poa_key],
                temperature_cell=self.df.loc[
                    boolean_mask, self.temperature_cell_key],
                vmp=self.df.loc[
                        boolean_mask, self.voltage_key] / self.modules_per_string,
                imp=self.df.loc[
                        boolean_mask, self.current_key] / self.parallel_strings,
                cells_in_series=self.cells_in_series,
                # delta_T=self.delta_T,
                technology=technology,
                verbose=verbose
            )

    def single_diode_predict(self,
                             effective_irradiance : array,
                             temperature_cell : array,
                             operating_cls : array,
                             params : dict,
                             ):

        voltage, current = pv_system_single_diode_model(
            effective_irradiance=effective_irradiance,
            temperature_cell=temperature_cell,
            operating_cls=operating_cls,
            cells_in_series=self.cells_in_series,
            alpha_isc=self.alpha_isc,
            resistance_shunt_ref=params['resistance_shunt_ref'],
            diode_factor=params['diode_factor'],
            photocurrent_ref=params['photocurrent_ref'],
            saturation_current_ref=params['saturation_current_ref'],
            resistance_series_ref=params['resistance_series_ref'],
            conductance_shunt_extra=params['conductance_shunt_extra'],
            band_gap_ref = self.Eg_ref,
            dEgdT = self.dEgdT
        )

        return voltage, current

    def build_plot_text_str(self, df : 'dataframe', p_plot : bool =None):

        if len(df) > 0:
            dates_str = 'Dates: {} to {}\n'.format(
                df.index[0].strftime("%m/%d/%Y"),
                df.index[-1].strftime("%m/%d/%Y"))
        else:
            dates_str = 'Dates: None\n'

        system_info_str = 'System: {}\n'.format(self.system_name) + \
                          dates_str + \
                          'Current: {}\n'.format(self.current_key) + \
                          'Voltage: {}\n'.format(self.voltage_key) + \
                          'Irradiance: {}\n'.format(self.irradiance_poa_key)

        if p_plot is not None:
            text_str = system_info_str + \
                       'n_diode: {:1.2f} \n'.format(p_plot['diode_factor']) + \
                       'reference_photocurrent: {:1.2f} A\n'.format(
                           p_plot['photocurrent_ref']) + \
                       'saturation_current_ref: {:1.2f} nA\n'.format(
                           p_plot['saturation_current_ref'] * 1e9) + \
                       'resistance_series: {:1.2f} Ohm\n'.format(
                           p_plot['resistance_series_ref']) + \
                        'resistance_shunt: {:1.2f} Ω\n\n'.format(
                           p_plot['resistance_shunt_ref'])
                    #    'conductance shunt: {:1.4f} 1/Ω\n\n'.format(
                    #        p_plot['conductance_shunt_extra'])
        else:
            text_str = system_info_str

        return text_str

    def plot_Vmp_Imp_scatter(self,
                             df : 'dataframe',
                             p_plot : bool =None,
                             figure_number : bool =None,
                             vmin : float =0,
                             vmax : float =70,
                             plot_imp_max : float =8,
                             plot_vmp_max : float =40,
                             figsize : tuple =(6.5, 3.5),
                             cbar : bool =True,
                             ylabel : str ='Current (A)',
                             xlabel : str ='Voltage (V)'):
        """
        Make Vmp, Imp scatter plot.

        Parameters
        ----------
        p_plot
        figure_number

        vmin
        vmax

        Returns
        -------

        """

        temp_limits = np.linspace(vmin, vmax, 8)

        if len(df) > 0:
            inv_on_points = np.array(df['operating_cls'] == 0)
            vmp = np.array(
                df.loc[
                    inv_on_points, self.voltage_key]) / self.modules_per_string
            imp = np.array(
                df.loc[inv_on_points, self.current_key]) / self.parallel_strings

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
                df[self.irradiance_poa_key] > 995,
                df[self.irradiance_poa_key] < 1005,
            ))
            if len(one_sun_points) > 0:
                # print('number one sun points: ', len(one_sun_points))
                plt.scatter(df.loc[
                                one_sun_points, self.voltage_key] / self.modules_per_string,
                            df.loc[
                                one_sun_points, self.current_key] / self.parallel_strings,
                            c=df.loc[one_sun_points, 'temperature_cell'],
                            edgecolors='k',
                            s=0.2)

            # Plot temperature scan
            temperature_smooth = np.linspace(vmin, vmax, 20)

            for effective_irradiance in [100, 1000]:
                voltage_plot, current_plot = self.single_diode_predict(
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

                voltage_plot, current_plot = self.single_diode_predict(
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
        text_str = self.build_plot_text_str(df, p_plot=p_plot)

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

    def plot_temperature_Vmp_scatter(self,
                                     df : 'dataframe',
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

    def plot_suns_voc_scatter(self,
                              df : 'dataframe',
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

    def plot_current_irradiance_clipped_scatter(self,
                                                df : 'dataframe',
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

    def plot_current_irradiance_mpp_scatter(self,
                                            df : 'dataframe',
                                            p_plot : bool =None,
                                            figure_number : int =3,
                                            vmin : float =0,
                                            vmax : float =70,
                                            plot_imp_max : float =8,
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

        # Make figure for inverter on.

        temp_limits = np.linspace(vmin, vmax, 8)

        if len(df) > 0:
            cax = np.array(df['operating_cls'] == 0)

            current = np.array(
                df.loc[cax, self.current_key]) / self.parallel_strings

            irrad = np.array(df.loc[cax, self.irradiance_poa_key])
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
                    cells_in_series=self.cells_in_series,
                    alpha_isc=self.alpha_isc,
                    resistance_shunt_ref=p_plot['resistance_shunt_ref'],
                    diode_factor=p_plot['diode_factor'],
                    photocurrent_ref=p_plot['photocurrent_ref'],
                    saturation_current_ref=p_plot['saturation_current_ref'],
                    resistance_series_ref=p_plot['resistance_series_ref'],
                    conductance_shunt_extra=p_plot['conductance_shunt_extra'],
                    band_gap_ref = self.Eg_ref,
                    dEgdT = self.dEgdT

                )

                norm_temp = (temp_curr - vmin) / (vmax - vmin)
                line_color = np.array(h_sc.cmap(norm_temp))

                line_color[3] = 0.3

                plt.plot(irrad_smooth, current_plot,
                         label='Fit {:2.0f} C'.format(temp_curr),
                         color=line_color,
                         # color='C' + str(j)
                         )

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


    def plot_temperature_rise_irradiance_scatter(self,
                                                 df : 'dataframe',
                                                 p_plot : dict,
                                                 figure_number : int =1,
                                                 vmin : float =0,
                                                 vmax : float =70,
                                                 plot_imp_max : float =8,
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
