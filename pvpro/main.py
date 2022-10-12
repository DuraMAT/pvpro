from array import array
import pvlib
import numpy as np
import pandas as pd


import datetime
import os
import time
import scipy
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from tqdm import tqdm
from functools import partial


from matplotlib.colors import LinearSegmentedColormap

from solardatatools import DataHandler
from pvlib.singlediode import _lambertw_i_from_v, _lambertw_v_from_i, \
    bishop88_mpp, bishop88_v_from_i
from pvlib.pvsystem import calcparams_desoto, singlediode

from pvpro.estimate import estimate_singlediode_params

from pvanalytics.features import clipping
from sklearn.linear_model import LinearRegression

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
        self.Eg_ref, self.dEgdT = self.estimate_Eg_dEgdT(self.technology)

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

                fit_result_iter = self.production_data_curve_fit(
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

    def _x_to_p(self, x : array, key : str):
        """
        Change from numerical fit value (x) to physical parameter (p). This
        transformation improves the numerical performance of the fitting algorithm.

        Parameters
        ----------
        x : ndarray

            Value of parameter for numerical fitting.

        key : str

            parameter to change, can be 'diode_factor', 'photocurrent_ref',
            'saturation_current_ref', 'resistance_series_ref',
            'resistance_shunt_ref', 'conductance_shunt_extra'


        Returns
        -------
        p : ndarray

            Value of physical parameter.


        """
        if key == 'saturation_current_ref':
            return np.exp(x-23)
        
        elif key == 'saturation_current':
            return np.exp(x-23)
        elif key == 'resistance_shunt_ref':
            return np.exp(2 * (x - 1))
        elif key == 'resistance_shunt':
            return np.exp(2 * (x - 1))
        elif key == 'alpha_isc':
            return x * 1e-3
        elif key == 'resistance_series_ref':
            return x/2.2
        elif key == 'resistance_series':
            return x/2.2
        else:
            return x

    def _p_to_x(self, p : array, key : str):
        """
        Change from physical parameter (p) to numerical fit value (x). This
        transformation improves the numerical performance of the fitting algorithm.

        Parameters
        ----------
        p : ndarray

            Value of physical parameter.

        key : str

            parameter to change, can be 'diode_factor', 'photocurrent_ref',
            'saturation_current_ref', 'resistance_series_ref',
            'resistance_shunt_ref', 'conductance_shunt_extra'

        Returns
        -------
        x : ndarray

            Value of parameter for numerical fitting.
        """
        if key == 'saturation_current_ref':
            return np.log(p) + 23
        elif key == 'resistance_shunt_ref':
            return np.log(p) / 2 + 1
        elif key == 'resistance_shunt':
            return np.log(p) / 2 + 1
        elif key == 'resistance_series_ref':
            return p*2.2
        elif key == 'resistance_series':
            return p*2.2
        elif key == 'saturation_current':
            return np.log(p)+23
        elif key == 'alpha_isc':
            return p * 1e3
        else:
            return p

    def _pvpro_L2_loss(self, x : array, sdm : 'function', voltage : array, current : array, voltage_scale : float, current_scale : float,
                    weights : float, fit_params : list):
        voltage_fit, current_fit = sdm(
            **{param: self._x_to_p(x[n], param) for n, param in
            zip(range(len(x)), fit_params)}
        )

        # Note that summing first and then calling nanmean is slightly faster.
        return np.nanmean(((voltage_fit - voltage) * weights / voltage_scale) ** 2 + \
                        ((current_fit - current) * weights / current_scale) ** 2)

    def production_data_curve_fit(self,
        temperature_cell : array,
        effective_irradiance : array,
        operating_cls : array,
        voltage : array,
        current : array,
        cells_in_series : int =60,
        band_gap_ref : float = None,
        dEgdT : float = None,
        p0 : dict =None,
        lower_bounds : float =None,
        upper_bounds : float =None,
        alpha_isc : float =None,
        diode_factor : array =None,
        photocurrent_ref : array =None,
        saturation_current_ref : array =None,
        resistance_series_ref : array =None,
        resistance_shunt_ref : array =None,
        conductance_shunt_extra : array =None,
        verbose : bool =False,
        solver : str ='L-BFGS-B',
        singlediode_method : str ='fast',
        method : str ='minimize',
        use_mpp_points : bool =True,
        use_voc_points : bool =True,
        use_clip_points : bool =True,
        # fit_params=None,
        saturation_current_multistart : array =None,
        brute_number_grid_points : int =2
        ):
        """
        Use curve fitting to find best-fit single-diode-model paramters given the
        operating data.

        Data to be fit is supplied in 1xN vectors as the inputs temperature_cell,
        effective_irradiance, operation_cls, voltage and current. Each of these
        inputs must have the same length.

        The inputs lower_bound, upper_bound, and p0 are all dictionaries
        with the same keys.

        Parameters
        ----------
        temperature_cell : ndarray

            Temperature of PV cell in C.

        effective_irradiance : ndarray

            Effective irradiance reaching module in W/m2.

        operating_cls : ndarray

            Integer describing the type of operating point. See
            classify_operating_mode for a description of the operating points.

        voltage : ndarray

            DC voltage of the PV module. To get from a string voltage to the
            effective module voltage, divide by the string size.

        current : ndarray

            DC current of the PV module. To get from an array current to the
            effective module current, divide by the number of strings in parallel.

        lower_bounds : dict

            dictionary of the lower bound for each fit parameter.

        upper_bounds : dict

            dictionary of the upper bound for each fit parameter.

        alpha_isc : numeric

            Temperature coefficient of short circuit current in A/C. If not
            provided then alpha_isc is a fit paramaeter. Typically it is not
            suggested to make alpha_isc a fit parameter since it only slightly
            affects module operating voltage/current.

        diode_factor : numeric or None (default).

            diode ideality factor. If not provided, then diode_factor is a fit
            parameter.

        photocurrent_ref : numeric or None (default)

            Reference photocurrent in A. If not provided, then photocurrent_ref
            is a fit parameter.

        saturation_current_ref : numeric or None (default)

            Reference saturation current in A. If not provided or if set to None,
            then saturation_current_ref is a fit parameter.

        resistance_series_ref : numeric or None (default)

            Reference series resistance in Ohms. If not provided or if set to None,
            then resistance_series_ref is a fit parameter.

        resistance_shunt_ref : numeric or None (default)

            Reference shunt resistance in Ohms. If not provided or if set to None,
            then resistance_shunt_ref is a fit parameter.

        p0 : dict

            Dictionary providing startpoint.

        cells_in_series : int

            Number of cells in series in the module.

        band_gap_ref : float

            Band gap at reference conditions in eV.

        verbose : bool

            Display fit output.

        Returns
        -------

        """
        voltage = np.array(voltage)
        current = np.array(current)
        effective_irradiance = np.array(effective_irradiance)
        temperature_cell = np.array(temperature_cell)
        operating_cls = np.array(operating_cls)

        # Check all the same length:
        if len(np.unique([len(voltage), len(current), len(effective_irradiance),
                        len(temperature_cell), len(operating_cls)])) > 1:
            raise Exception("Length of inputs 'voltage', 'current', 'effective_irradiance', 'temperature_cell', 'operating_cls' must all be the same." )

        if p0 is None:
            p0 = dict(
                diode_factor=1.0,
                photocurrent_ref=6,
                saturation_current_ref=1e-9,
                resistance_series_ref=0.2,
                conductance_shunt_extra=0.001,
                alpha_isc=0.002
            )

        # if fit_params is None:

        fit_params = []
        if diode_factor is None:
            fit_params.append('diode_factor')
        if photocurrent_ref is None:
            fit_params.append('photocurrent_ref')
        if saturation_current_ref is None:
            fit_params.append('saturation_current_ref')
        if resistance_series_ref is None:
            fit_params.append('resistance_series_ref')
        if conductance_shunt_extra is None:
            fit_params.append('conductance_shunt_extra')
        if resistance_shunt_ref is None:
            fit_params.append('resistance_shunt_ref')
        if alpha_isc is None:
            fit_params.append('alpha_isc')
        if band_gap_ref is None:
            fit_params.append('band_gap_ref')

        if lower_bounds is None:
            lower_bounds = dict(
                diode_factor=0.5,
                photocurrent_ref=0.01,
                saturation_current_ref=1e-13,
                resistance_series_ref=0,
                conductance_shunt_extra=0,
                alpha_isc=0,
                band_gap_ref=0.5,
            )

        if upper_bounds is None:
            upper_bounds = dict(
                diode_factor=2,
                photocurrent_ref=20,
                saturation_current_ref=1e-5,
                resistance_series_ref=1,
                conductance_shunt_extra=10,
                alpha_isc=1,
                band_gap_ref=2.0,
            )

        if saturation_current_multistart is None:
            saturation_current_multistart = [0.2, 0.5, 1, 2, 5]

        # Only keep points belonging to certain operating classes
        cls_keepers = np.logical_or.reduce(
            (use_mpp_points * (operating_cls == 0),
            use_voc_points * (operating_cls == 1),
            use_clip_points * (operating_cls == 2),
            ))

        keepers = np.logical_and.reduce((cls_keepers,
                                        np.isfinite(voltage),
                                        np.isfinite(current),
                                        np.isfinite(effective_irradiance),
                                        np.isfinite(temperature_cell)
                                        ))
        effective_irradiance = effective_irradiance[keepers]
        temperature_cell = temperature_cell[keepers]
        operating_cls = operating_cls[keepers]
        voltage = voltage[keepers]
        current = current[keepers]

        weights = np.zeros_like(operating_cls)
        weights[operating_cls == 0] = 1
        weights[operating_cls == 1] = 1
        weights[operating_cls == 2] = 1

        if verbose:
            print('Total number points: {}'.format(len(operating_cls)))
            print('Number MPP points: {}'.format(np.sum(operating_cls == 0)))
            print('Number Voc points: {}'.format(np.sum(operating_cls == 1)))
            print('Number Clipped points: {}'.format(np.sum(operating_cls == 2)))

        fixed_params = {}
        # Set kwargs
        if not diode_factor == None:
            fixed_params['diode_factor'] = diode_factor
        if not photocurrent_ref == None:
            fixed_params['photocurrent_ref'] = photocurrent_ref
        if not saturation_current_ref == None:
            fixed_params['saturation_current_ref'] = saturation_current_ref
        if not resistance_series_ref == None:
            fixed_params['resistance_series_ref'] = resistance_series_ref
        if not resistance_shunt_ref == None:
            fixed_params['resistance_shunt_ref'] = resistance_shunt_ref
        if not alpha_isc == None:
            fixed_params['alpha_isc'] = alpha_isc
        if not conductance_shunt_extra == None:
            fixed_params['conductance_shunt_extra'] = conductance_shunt_extra
        if not band_gap_ref == None:
            fixed_params['band_gap_ref'] = band_gap_ref
        if not dEgdT == None:
            fixed_params['dEgdT'] = dEgdT


        # If no points left after removing unused classes, function returns.
        if len(effective_irradiance) == 0 or len(
                operating_cls) == 0 or len(voltage) == 0 or len(current) == 0:

            if verbose:
                print('No valid values received.')

            return {'p': {key: np.nan for key in fit_params},
                    'fixed_params': fixed_params,
                    'residual': np.nan,
                    'p0': p0,
                    }

        model_kwargs = dict(
            effective_irradiance=effective_irradiance,
            temperature_cell=temperature_cell,
            operating_cls=operating_cls,
            cells_in_series=cells_in_series,
            voltage_operation=voltage,
            current_operation=current,
            singlediode_method=singlediode_method,
        )

        # Set scale factor for current and voltage:
        current_median = np.median(current)
        voltage_median = np.median(voltage)


        sdm = partial(self.pv_system_single_diode_model, **model_kwargs, **fixed_params)

        residual = partial(self._pvpro_L2_loss,
                        sdm=sdm,
                        voltage=voltage,
                        current=current,
                        voltage_scale=voltage_median,
                        current_scale=current_median,
                        weights=weights,
                        fit_params=fit_params)

        if method == 'minimize':
            if solver.lower() == 'nelder-mead':
                bounds = None
            elif solver.upper() == 'L-BFGS-B':
                bounds = scipy.optimize.Bounds(
                    [self._p_to_x(lower_bounds[k], k) for k in fit_params],
                    [self._p_to_x(upper_bounds[k], k) for k in fit_params])
            else:
                raise Exception('solver must be "Nelder-Mead" or "L-BFGS-B"')

            if verbose:
                print('--')
                print('p0:')
                for k in fit_params:
                    print(k, p0[k])
                print('--')

            if verbose:
                print('Performing minimization... ')

            if 'saturation_current_ref' in fit_params:
                saturation_current_ref_start = p0['saturation_current_ref']
                loss = np.inf
                for Io_multiplier in saturation_current_multistart:

                    p0[
                        'saturation_current_ref'] = saturation_current_ref_start * Io_multiplier
                    # Get numerical fit start point.
                    x0 = [self._p_to_x(p0[k], k) for k in fit_params]

                    res_curr = minimize(residual,
                                        x0=x0,
                                        bounds=bounds,
                                        method=solver,
                                        options=dict(
                                            # maxiter=500,
                                            disp=False,
                                            # ftol=1e-9,
                                        ),
                                        )
                    if verbose:
                        print('Saturation current multiplier: ', Io_multiplier)
                        print('loss:', res_curr['fun'])

                    if res_curr['fun'] < loss:
                        if verbose:
                            print('New best value found, setting')
                        res = res_curr
                        loss = res_curr['fun']
            else:
                # Get numerical fit start point.
                x0 = [self._p_to_x(p0[k], k) for k in fit_params]

                res = minimize(residual,
                            x0=x0,
                            bounds=bounds,
                            method=solver,
                            options=dict(
                                #    maxiter=100,
                                disp=False,
                                #    ftol=0.001,
                            ),
                            )

            if verbose:
                print(res)
            n = 0
            p_fit = {}
            for param in fit_params:
                p_fit[param] = self._x_to_p(res.x[n], param)
                n = n + 1

            out = {'p': p_fit,
                'fixed_params': fixed_params,
                'residual': res['fun'],
                'x0': x0,
                'p0': p0,
                }
            for k in res:
                out[k] = res[k]

            return out

    """
    Functions about single diode modelling
    """
    def estimate_Eg_dEgdT(self, technology : str):
        allEg = {'multi-Si': 1.121,
                            'mono-Si': 1.121,
                            'GaAs': 1.424,
                            'CIGS': 1.15, 
                            'CdTe':  1.475}

        alldEgdT = {'multi-Si': -0.0002677,
                            'mono-Si': -0.0002677,
                            'GaAs': -0.000433,
                            'CIGS': -0.00001, 
                            'CdTe': -0.0003}

        return allEg[technology], alldEgdT[technology]

    def calcparams_pvpro(self, effective_irradiance : array, temperature_cell : array,
                        alpha_isc : array, nNsVth_ref : array, photocurrent_ref : array,
                        saturation_current_ref : array,
                        resistance_shunt_ref : array, resistance_series_ref : array,
                        conductance_shunt_extra : array,
                        band_gap_ref : float =None, dEgdT : float =None,
                        irradiance_ref : float =1000, temperature_ref : float =25):
        """
        Similar to pvlib calcparams_desoto, except an extra shunt conductance is
        added.

        Returns
        -------

        """

        iph, io, rs, rsh, nNsVth = calcparams_desoto(
            effective_irradiance,
            temperature_cell,
            alpha_sc=alpha_isc,
            a_ref=nNsVth_ref,
            I_L_ref=photocurrent_ref,
            I_o_ref=saturation_current_ref,
            R_sh_ref=resistance_shunt_ref,
            R_s=resistance_series_ref,
            EgRef=band_gap_ref,
            dEgdT=dEgdT,
            irrad_ref=irradiance_ref,
            temp_ref=temperature_ref
        )
        

        return iph, io, rs, rsh, nNsVth

    def singlediode_fast(self, photocurrent : array, 
                        saturation_current : array, 
                        resistance_series : array,
                        resistance_shunt : array, nNsVth : array, calculate_voc : bool =False):
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth)  # collect args

        i_mp, v_mp, p_mp = bishop88_mpp(
            *args, method='newton'
        )
        if calculate_voc:
            v_oc = bishop88_v_from_i(
                0.0, *args, method='newton'
            )

            return {'v_mp': v_mp,
                'i_mp': i_mp,
                'v_oc': v_oc}
        else:
            return {'v_mp': v_mp,
                'i_mp': i_mp}

    def pvlib_single_diode(self,
            effective_irradiance : float,
            temperature_cell : float,
            resistance_shunt_ref : array,
            resistance_series_ref : array,
            diode_factor : array,
            cells_in_series : array,
            alpha_isc : array,
            photocurrent_ref : array,
            saturation_current_ref : array,
            conductance_shunt_extra : array =0,
            irradiance_ref : float =1000,
            temperature_ref : float =25,
            ivcurve_pnts : bool =None,
            output_all_params : bool =False,
            singlediode_method : str ='fast',
            calculate_voc : bool =False,
            technology : str = None,
            band_gap_ref : float = None,
            dEgdT : float = None
    ):
        """
        Find points of interest on the IV curve given module parameters and
        operating conditions.

        method 'newton' is about twice as fast as method 'lambertw'

        Parameters
        ----------
        effective_irradiance : numeric
            effective irradiance in W/m^2

        temp_cell : numeric
            Cell temperature in C

        resistance_shunt : numeric

        resistance_series : numeric

        diode_ideality_factor : numeric

        number_cells_in_series : numeric

        alpha_isc :
            in amps/c

        reference_photocurrent : numeric
            photocurrent at standard test conditions, in A.

        reference_saturation_current : numeric

        reference_Eg : numeric
            band gap in eV.

        reference_irradiance : numeric
            reference irradiance in W/m^2. Default is 1000.

        reference_temperature : numeric
            reference temperature in C. Default is 25.

        technology : string
            PV technology

        verbose : bool
            Whether to print information.

        Returns
        -------
        OrderedDict or DataFrame

            The returned dict-like object always contains the keys/columns:

                * i_sc - short circuit current in amperes.
                * v_oc - open circuit voltage in volts.
                * i_mp - current at maximum power point in amperes.
                * v_mp - voltage at maximum power point in volts.
                * p_mp - power at maximum power point in watts.
                * i_x - current, in amperes, at ``v = 0.5*v_oc``.
                * i_xx - current, in amperes, at ``V = 0.5*(v_oc+v_mp)``.

            If ivcurve_pnts is greater than 0, the output dictionary will also
            include the keys:

                * i - IV curve current in amperes.
                * v - IV curve voltage in volts.

            The output will be an OrderedDict if photocurrent is a scalar,
            array, or ivcurve_pnts is not None.

            The output will be a DataFrame if photocurrent is a Series and
            ivcurve_pnts is None.

        """
        if band_gap_ref is None:
            band_gap_ref, dEgdT = self.estimate_Eg_dEgdT(technology)

        kB = 1.381e-23
        q = 1.602e-19
        nNsVth_ref = diode_factor * cells_in_series * kB / q * (
                273.15 + temperature_ref)

        iph, io, rs, rsh, nNsVth = self.calcparams_pvpro( effective_irradiance,
                                                    temperature_cell,
                                                    alpha_isc,
                                                    nNsVth_ref,
                                                    photocurrent_ref,
                                                    saturation_current_ref,
                                                    resistance_shunt_ref,
                                                    resistance_series_ref,
                                                    conductance_shunt_extra,
                                                    band_gap_ref=band_gap_ref, dEgdT=dEgdT,
                                                    irradiance_ref=irradiance_ref,
                                                    temperature_ref=temperature_ref)
        if len(iph)>1: 
            iph[iph <= 0] = 0 

        if singlediode_method == 'fast':
            out = self.singlediode_fast(iph,
                                io,
                                rs,
                                rsh,
                                nNsVth,
                                calculate_voc=calculate_voc
                                )

        elif singlediode_method in ['lambertw', 'brentq', 'newton']:
            out = singlediode(iph,
                            io,
                            rs,
                            rsh,
                            nNsVth,
                            method=singlediode_method,
                            ivcurve_pnts=ivcurve_pnts,
                            )
        else:
            raise Exception(
                'Method must be "fast","lambertw", "brentq", or "newton"')
        # out = rename(out)

        if output_all_params:

            params = {'photocurrent': iph,
                    'saturation_current': io,
                    'resistance_series': rs,
                    'resistace_shunt': rsh,
                    'nNsVth': nNsVth}

            for p in params:
                out[p] = params[p]

        return out

    def singlediode_v_from_i(self, 
            current : array,
            effective_irradiance : array,
            temperature_cell : array,
            resistance_shunt_ref : array,
            resistance_series_ref : array,
            diode_factor : array,
            cells_in_series : int,
            alpha_isc : float,
            photocurrent_ref : array,
            saturation_current_ref : array,
            band_gap_ref : float =None, 
            dEgdT : float=None,
            reference_irradiance : float=1000,
            reference_temperature : float=25,
            method : str ='newton',
            verbose : bool =False,
        ):
        """
        Calculate voltage at a particular point on the IV curve.

        """
        kB = 1.381e-23
        q = 1.602e-19

        iph, io, rs, rsh, nNsVth = calcparams_desoto(
            effective_irradiance,
            temperature_cell,
            alpha_sc=alpha_isc,
            a_ref=diode_factor * cells_in_series * kB / q * (
                    273.15 + reference_temperature),
            I_L_ref=photocurrent_ref,
            I_o_ref=saturation_current_ref,
            R_sh_ref=resistance_shunt_ref,
            R_s=resistance_series_ref,
            EgRef=band_gap_ref,
            dEgdT=dEgdT,
        )
        voltage = _lambertw_v_from_i(rsh, rs, nNsVth, current, io, iph)

        return voltage

    def singlediode_i_from_v(self,
            voltage : array,
            effective_irradiance : array,
            temperature_cell : array,
            resistance_shunt_ref : array,
            resistance_series_ref : array,
            diode_factor : array,
            cells_in_series : int,
            alpha_isc : float,
            photocurrent_ref : array,
            saturation_current_ref : array,
            band_gap_ref  : float =None,
            dEgdT : float =None,
            reference_irradiance : float =1000,
            reference_temperature : float =25,
            method : str ='newton',
            verbose : bool =False,
        ):
        """
        Calculate current at a particular voltage on the IV curve.

        """
        kB = 1.381e-23
        q = 1.602e-19

        iph, io, rs, rsh, nNsVth = calcparams_desoto(
            effective_irradiance,
            temperature_cell,
            alpha_sc=alpha_isc,
            a_ref=diode_factor * cells_in_series * kB / q * (
                    273.15 + reference_temperature),
            I_L_ref=photocurrent_ref,
            I_o_ref=saturation_current_ref,
            R_sh_ref=resistance_shunt_ref,
            R_s=resistance_series_ref,
            EgRef=band_gap_ref,
            dEgdT=dEgdT,
        )
        current = _lambertw_i_from_v(rsh, rs, nNsVth, voltage, io, iph)

        return current

    def pv_system_single_diode_model(self,
            effective_irradiance : array,
            temperature_cell : array,
            operating_cls : array,
            diode_factor : array,
            photocurrent_ref : array,
            saturation_current_ref : array,
            resistance_series_ref : array,
            conductance_shunt_extra : array,
            resistance_shunt_ref : array,
            cells_in_series : int,
            alpha_isc : array,
            voltage_operation : array = None,
            current_operation : array = None,
            technology : str = None,
            band_gap_ref : float = None,
            dEgdT : float = None,
            singlediode_method : str ='fast',
            **kwargs
    ):
        """
        Function for returning the dc operating (current and voltage) point given
        single diode model parameters and the operating_cls.

        If the operating class is open-circuit, or maximum-power-point then this
        function is a simple call to pvlib_single_diode.

        If the operating class is 'clipped', then a more complicated algorithm is
        used to find the "closest" point on the I,V curve to the
        current_operation, voltage_operation input point. For numerical
        efficiency, the point chosen is actually not closest, but triangulated
        based on the current_operation, voltage_operation input and
        horizontally/vertically extrapolated poitns on the IV curve.

        """
        if type(voltage_operation) == type(None):
            voltage_operation = np.zeros_like(effective_irradiance)
        if type(current_operation) == type(None):
            current_operation = np.zeros_like(effective_irradiance)

        out = self.pvlib_single_diode(
            effective_irradiance,
            temperature_cell,
            resistance_shunt_ref,
            resistance_series_ref,
            diode_factor,
            cells_in_series,
            alpha_isc,
            photocurrent_ref,
            saturation_current_ref,
            conductance_shunt_extra=conductance_shunt_extra,
            singlediode_method=singlediode_method,
            technology=technology,
            band_gap_ref=band_gap_ref,
            dEgdT=dEgdT,
            calculate_voc=np.any(operating_cls==1))

        # First, set all points to
        voltage_fit = out['v_mp']
        current_fit = out['i_mp']

        #         Array of classifications of each time stamp.
        #         0: System at maximum power point.
        #         1: System at open circuit conditions.
        #         2: Clip

        # If cls is 2, then system is clipped, need to find closest iv curve point.
        if np.any(operating_cls == 2):
            cax = operating_cls == 2
            voltage_operation[cax][voltage_operation[cax] > out['v_oc'][cax]] = \
                out['v_oc'][cax]
            current_operation[cax][current_operation[cax] > out['i_sc'][cax]] = \
                out['i_sc'][cax]

            current_closest = self.singlediode_i_from_v(voltage=voltage_operation[cax],
                                                effective_irradiance=
                                                effective_irradiance[cax],
                                                temperature_cell=
                                                temperature_cell[cax],
                                                resistance_shunt_ref=resistance_shunt_ref,
                                                resistance_series_ref=resistance_series_ref,
                                                diode_factor=diode_factor,
                                                cells_in_series=cells_in_series,
                                                alpha_isc=alpha_isc,
                                                photocurrent_ref=photocurrent_ref,
                                                saturation_current_ref=saturation_current_ref,
                                                band_gap_ref=band_gap_ref,
                                                dEgdT=dEgdT,
                                                )

            voltage_closest = self.singlediode_v_from_i(
                current=current_operation[cax],
                effective_irradiance=effective_irradiance[cax],
                temperature_cell=temperature_cell[cax],
                resistance_shunt_ref=resistance_shunt_ref,
                resistance_series_ref=resistance_series_ref,
                diode_factor=diode_factor,
                cells_in_series=cells_in_series,
                alpha_isc=alpha_isc,
                photocurrent_ref=photocurrent_ref,
                saturation_current_ref=saturation_current_ref,
                band_gap_ref=band_gap_ref,
                dEgdT=dEgdT,
            )

            voltage_closest[voltage_closest < 0] = 0


            voltage_fit[cax] = 0.5 * (voltage_operation[cax] + voltage_closest)
            current_fit[cax] = 0.5 * (current_operation[cax] + current_closest)

        # If cls is 1, then system is at open-circuit voltage.
        cls1 = operating_cls == 1
        if np.sum(cls1) > 0:
            voltage_fit[operating_cls == 1] = out['v_oc'][operating_cls == 1]
            current_fit[operating_cls == 1] = 0

        return voltage_fit, current_fit



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
