from array import array
import pvlib
import numpy as np
import pandas as pd


import datetime
import os
import time
import scipy
import shutil
import string
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from tqdm import tqdm
from functools import partial
from scipy.special import lambertw
from numpy.linalg import pinv
from xmlrpc.client import boolean
from sklearn.linear_model import HuberRegressor
from matplotlib.colors import LinearSegmentedColormap
from solardatatools import DataHandler
from pvlib.pvsystem import calcparams_desoto, singlediode
from pvlib.singlediode import _lambertw_i_from_v, _lambertw_v_from_i, \
    bishop88_mpp, bishop88_v_from_i


"""
Class for running pvpro analysis.
"""
class PvProHandler:
    
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

        pvesti = EstimateInitial()

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

                    est, est_results = pvesti.estimate_singlediode_params(
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


        sdm = partial(pv_system_single_diode_model, **model_kwargs, **fixed_params)

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

    def estimate_p0(self,
                    verbose : bool =False,
                    boolean_mask : bool =None,
                    technology : bool =None):
        """
        Make a rough estimate of the startpoint for fitting the single diodemodel.
        """
        pvesti = EstimateInitial()

        if boolean_mask is None:
            self.p0, result = pvesti.estimate_singlediode_params(
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
            self.p0, result = pvesti.estimate_singlediode_params(
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

    def analyze_yoy(self, pfit : 'dataframe'):
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

    def calculate_error_real(self, pfit : 'dataframe', df_ref : 'dataframe', nrolling : int = 1):
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

    def calculate_error_synthetic(self, pfit : 'dataframe', df : 'dataframe', nrolling : int = 1):
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
        Functions of plotting
        """


"""
Class for estimation of initial paramters
"""
class EstimateInitial:

    def estimate_imp_ref(self, poa : array,
                     temperature_cell : array,
                     imp : array,
                     poa_lower_limit : float =200,
                     irradiance_ref : float =1000,
                     temperature_ref : float =25,
                     figure : bool =False,
                     figure_number : int =20 ,
                     model : string ='sandia',
                     verbose : bool =False,
                     solver : string ='huber',
                     epsilon : float =1.5,
                     ):
        """
        Estimate imp_ref and beta_imp using operation data.

        Model forms taken from Ref. [1]

        [1] D.L. King, W.E. Boyson, J.A. Kratochvill. Photovoltaic Array
        Performance Model. SAND2004-3535.

        Parameters
        ----------
        poa : array
            Plane-of-array irradiance in W/m2

        temperature_cell : array
            cell temperature in C.

        imp : array
            DC current at max power.

        irradiance_ref : float
            Reference irradiance, typically 1000 W/m^2

        temperature_ref : float
            Reference temperature, typically 25 C

        figure : bool
            Whether to plot a figure

        figure_number : int
            Figure number for plotting

        model : str

            Model to solve. Options are:

            'temperature' - Model is Imp = I_mp_ref * E/E_ref *(1 + alpha_imp * (T-T_ref))

            'sandia'. Model is Imp = I_mp_ref * (c0*E/E_ref + c1* (E/E_ref)^2) *(1 + alpha_imp * (T-T_ref))

        verbose : bool
            Verbose output

        Returns
        -------
        dict containing
            i_mp_ref

            alpha_imp

            i_mp_model

        """

        cax = np.logical_and.reduce((
            poa > poa_lower_limit,
            poa < 1100,
            np.isfinite(poa),
            np.isfinite(temperature_cell),
            np.isfinite(imp)
            # imp > np.nanmax(imp) * 0.5
        ))

        if np.sum(cax) < 2:
            return np.nan, np.nan

        temperature_cell = np.array(temperature_cell[cax])
        imp = np.array(imp[cax])
        poa = np.array(poa[cax])

        # kB = 1.381e-23
        # q = 1.602e-19
        # Vth = kB * (temperature_cell + 273.15) / q

        # avoid problem with integer input
        Ee = np.array(poa, dtype='float64') / irradiance_ref

        dT = temperature_cell - temperature_ref

        if model.lower() == 'sandia':

            X = np.zeros(shape=(len(temperature_cell), 4))
            X[:, 0] = Ee
            X[:, 1] = dT * Ee
            X[:, 2] = Ee - 1
            X[:, 3] = dT * (Ee - 1)

            if solver.lower() == 'huber':
                huber = HuberRegressor(epsilon=epsilon,
                                    fit_intercept=False)
                huber.fit(X, imp)
                coeff = huber.coef_
            elif solver.lower() == 'pinv':
                coeff = np.dot(pinv(X), imp)

            imp_ref = coeff[0]
            alpha_imp = coeff[1]
            imp_irrad_coeff_1 = coeff[2]
            imp_irrad_coeff_2 = coeff[3]

            # coeff_irrad_1 = coeff[2]
            # coeff_irrad_2 = coeff[3]

            def imp_model(temperature, irradiance):
                Ee = irradiance / irradiance_ref
                return Ee * (imp_ref + alpha_imp * (temperature - temperature_ref) + \
                            imp_irrad_coeff_1 * (Ee - 1) + \
                            imp_irrad_coeff_2 * (temperature - temperature_ref) * (
                                    Ee - 1))

            out = {'i_mp_ref': imp_ref,
                'alpha_imp': alpha_imp,
                'i_mp_model': imp_model,
                }

        elif model.lower() == 'temperature':
            X = np.zeros(shape=(len(temperature_cell), 2))
            X[:, 0] = Ee
            X[:, 1] = dT * Ee

            coeff = np.dot(pinv(X), imp)

            imp_ref = coeff[0]
            alpha_imp = coeff[1]

            def imp_model(temperature, irradiance):
                Ee = irradiance / irradiance_ref
                return Ee * (imp_ref + alpha_imp * (temperature - temperature_ref))

            out = {'i_mp_ref': imp_ref,
                'alpha_imp': alpha_imp,
                'i_mp_model': imp_model,
                }
        else:
            raise Exception(
                'Vmp model not recognized, valid options are "sandia" and "temperature"')

        if figure:
            plt.figure(figure_number)
            plt.clf()

            vmin = 10
            vmax = 80
            h_sc = plt.scatter(poa, imp,
                            c=temperature_cell,
                            s=0.2,
                            vmin=vmin,
                            cmap='jet',
                            vmax=vmax)

            x_smooth = np.linspace(0, 1000, 2)

            for temperature_plot in [25, 50]:
                norm_temp = (temperature_plot - vmin) / (vmax - vmin)
                line_color = np.array(h_sc.cmap(norm_temp))

                line_color[3] = 0.9
                plt.plot(x_smooth, imp_model(temperature_plot, x_smooth),
                        color=line_color)
                if temperature_plot == 25:
                    plt.plot(irradiance_ref, imp_ref, '.',
                            color=line_color)
                    plt.text(irradiance_ref, imp_ref,
                            'imp_ref: {:.2f} V  '.format(imp_ref),
                            horizontalalignment='right')

            plt.xlabel('POA (W/m2)')
            plt.ylabel('Imp (A)')
            plt.xlim([0, 1000])
            plt.ylim([0, imp_ref * 1.1])
            pcbar = plt.colorbar()
            pcbar.set_label('Cell temperature (C)')

            plt.show()

        return out

    def estimate_vmp_ref(self, poa : array,
                        temperature_cell : array,
                        vmp : array,
                        irradiance_ref : float =1000,
                        temperature_ref : float =25,
                        figure : bool =False,
                        figure_number : int =21,
                        model: string ='sandia1',
                        solver: string ='huber',
                        epsilon : float =2.5
                        ):
        """
        Estimate vmp_ref using operation data. Function works for any size of
        power block.

        Model forms taken from Ref. [1]

        [1] D.L. King, W.E. Boyson, J.A. Kratochvill. Photovoltaic Array
        Performance Model. SAND2004-3535.


        Parameters
        ----------
        poa : array
            Plane-of-array irradiance in W/m2

        temperature_cell : array
            cell temperature in C.

        vmp : array
            DC voltage at max power, in V.

        irradiance_ref : float
            Reference irradiance, typically 1000 W/m^2

        temperature_ref : float
            Reference temperature, typically 25 C

        figure : bool
            Whether to plot a figure

        figure_number : int
            Figure number for plotting

        model : str

            Model to solve. Options are:

            'sandia'. Model is Vmp = Vmp_ref + beta_vmp*(T-T_ref) + \
            c0*delta*log(E/E_ref) + c1 * (delta * log(E/E_ref))^2

            where delta = (temperature_cell + 273.15)

            'temperature' - Model is Vmp = Vmp_ref + beta_vmp*(T-T_ref)


        verbose : bool
            Verbose output

        Returns
        -------
        dict containing
            v_mp_ref

            beta_vmp

            v_mp_model

        """

        cax = np.logical_and.reduce((
            poa > 200,
            poa < 1100,
            np.isfinite(poa),
            np.isfinite(temperature_cell),
            np.isfinite(vmp)
            # imp > np.nanmax(imp) * 0.5
        ))

        if np.sum(cax) < 2:
            return np.nan, np.nan

        temperature_cell = np.array(temperature_cell[cax])
        vmp = np.array(vmp[cax])
        poa = np.array(poa[cax])

        delta = (temperature_cell + 273.15)

        # avoid problem with integer input
        Ee = np.array(poa, dtype='float64') / irradiance_ref

        # set up masking for 0, positive, and nan inputs
        Ee_gt_0 = np.full_like(Ee, False, dtype='bool')
        Ee_eq_0 = np.full_like(Ee, False, dtype='bool')
        notnan = ~np.isnan(Ee)
        np.greater(Ee, 0, where=notnan, out=Ee_gt_0)
        np.equal(Ee, 0, where=notnan, out=Ee_eq_0)

        # avoid repeated computation
        logEe = np.full_like(Ee, np.nan)
        np.log(Ee, where=Ee_gt_0, out=logEe)
        logEe = np.where(Ee_eq_0, -np.inf, logEe)

        if model.lower() == 'sandia':
            X = np.zeros(shape=(len(temperature_cell), 4))
            X[:, 0] = 1
            X[:, 1] = temperature_cell - temperature_ref
            X[:, 2] = delta * logEe
            X[:, 3] = (delta * logEe) ** 2

            if solver.lower() == 'huber':
                huber = HuberRegressor(epsilon=epsilon,
                                    fit_intercept=False)
                huber.fit(X, vmp)
                coeff = huber.coef_
            elif solver.lower() == 'pinv':
                coeff = np.dot(pinv(X), vmp)

            vmp_ref = coeff[0]
            beta_vmp = coeff[1]
            coeff_irrad_1 = coeff[2]
            coeff_irrad_2 = coeff[3]

            def vmp_model(temperature, irradiance):
                return vmp_ref + beta_vmp * (temperature - temperature_ref) + \
                    coeff_irrad_1 * (temperature + 273.15) * np.log(
                    irradiance / irradiance_ref) + \
                    coeff_irrad_2 * ((temperature + 273.15) * np.log(
                    irradiance / irradiance_ref)) ** 2

            out = {'v_mp_ref': vmp_ref,
                'beta_vmp': beta_vmp,
                'coeff_irrad_1': coeff_irrad_1,
                'coeff_irrad_2': coeff_irrad_2,
                'vmp_model': vmp_model}

        if model.lower() == 'sandia1':
            X = np.zeros(shape=(len(temperature_cell), 3))
            X[:, 0] = 1
            X[:, 1] = temperature_cell - temperature_ref
            X[:, 2] = delta * logEe

            if solver.lower() == 'huber':
                huber = HuberRegressor(epsilon=epsilon,
                                    fit_intercept=False)
                huber.fit(X, vmp)
                coeff = huber.coef_
            elif solver.lower() == 'pinv':
                coeff = np.dot(pinv(X), vmp)

            vmp_ref = coeff[0]
            beta_vmp = coeff[1]
            coeff_irrad_1 = coeff[2]

            def vmp_model(temperature, irradiance):
                return vmp_ref + beta_vmp * (temperature - temperature_ref) + \
                    coeff_irrad_1 * (temperature + 273.15) * np.log(
                    irradiance / irradiance_ref)

            out = {'v_mp_ref': vmp_ref,
                'beta_vmp': beta_vmp,
                'coeff_irrad_1': coeff_irrad_1,
                'vmp_model': vmp_model}

        elif model.lower() == 'temperature':
            X = np.zeros(shape=(len(temperature_cell), 2))
            X[:, 0] = 1
            X[:, 1] = temperature_cell - temperature_ref

            if solver.lower() == 'huber':
                huber = HuberRegressor(epsilon=epsilon,
                                    fit_intercept=False)
                huber.fit(X, vmp)
                coeff = huber.coef_
            elif solver.lower() == 'pinv':
                coeff = np.dot(pinv(X), vmp)

            vmp_ref = coeff[0]
            beta_vmp = coeff[1]

            def vmp_model(temperature, irradiance):
                return vmp_ref + beta_vmp * (temperature - temperature_ref)

            out = {'v_mp_ref': vmp_ref,
                'beta_vmp': beta_vmp,
                'vmp_model': vmp_model}
        else:
            raise Exception(
                'Vmp model not recognized, valid options are "sandia" and "temperature"')

        if figure:
            plt.figure(figure_number)
            plt.clf()

            vmin = 0
            vmax = 1100
            h_sc = plt.scatter(temperature_cell, vmp,
                            c=poa,
                            s=0.2,
                            vmin=vmin,
                            cmap='jet',
                            vmax=vmax)

            x_smooth = np.linspace(temperature_cell.min(), temperature_cell.max(),
                                2)

            for poa_plot in [250, 500, 750, 1000]:
                norm_poa = (poa_plot - vmin) / (vmax - vmin)
                line_color = np.array(h_sc.cmap(norm_poa))

                line_color[3] = 0.9
                plt.plot(x_smooth, vmp_model(x_smooth, poa_plot),
                        color=line_color)
                if poa_plot == 1000:
                    plt.plot(temperature_ref, vmp_ref, '.',
                            color=line_color)
                    plt.text(temperature_ref, vmp_ref,
                            '  vmp_ref: {:.2f} V'.format(vmp_ref),
                            horizontalalignment='left')

            plt.xlabel('Cell temperature (C)')
            plt.ylabel('Vmp (V)')
            pcbar = plt.colorbar()
            pcbar.set_label('POA (W/m2)')

            plt.show()

        return out

    def estimate_diode_factor(self, vmp_ref : array, beta_vmp : float, imp_ref : array,
                            alpha_isc_norm  : float =0,
                            resistance_series : float =0.35,
                            cells_in_series : int =60,
                            temperature_ref : float =25,
                            technology : string =None):

        # Thermal temperature
        k = 1.381e-23
        q = 1.602e-19
        Vth = k * (temperature_ref + 273.15) / q

        # Rough estimate: beta_voc is typically similar to beta_vmp
        beta_voc = self.estimate_beta_voc(beta_vmp, technology=technology)

        # Rough estimate, voc_ref is a little larger than vmp_ref.
        voc_ref = self.estimate_voc_ref(vmp_ref,technology=technology)

        beta_voc_norm = beta_voc / voc_ref

        delta_0 = (1 - 298.15 * beta_voc_norm) / (
                50.05 - 298.15 * alpha_isc_norm)

        w0 = lambertw(np.exp(1 / delta_0 + 1))

        nNsVth = (vmp_ref + resistance_series * imp_ref) / (w0 - 1)

        diode_factor = nNsVth / (cells_in_series * Vth)

        return diode_factor.real

    def estimate_photocurrent_ref_simple(self, imp_ref: array, technology : str ='mono-Si'):
        photocurrent_imp_ratio = {'multi-Si': 1.0746167586063207,
                                'mono-Si': 1.0723051517913444,
                                'thin-film': 1.1813401654607967,
                                'cigs': 1.1706462692015707,
                                'cdte': 1.1015249105470803}

        photocurrent_ref = imp_ref * photocurrent_imp_ratio[technology]

        return photocurrent_ref

    def estimate_saturation_current(self, isc : array, voc : array, nNsVth : array):
        """
            .. [2] John A Dufﬁe, William A Beckman, "Solar Engineering of Thermal
        Processes", Wiley, 2013

        Parameters
        ----------
        isc
        voc
        nNsVth

        Returns
        -------

        """
        return isc * np.exp(-voc / nNsVth)

    def estimate_cells_in_series(self, voc_ref : array, technology : str ='mono-Si'):
        """

        Note: Could improve this by using the fact that most modules have one of
        a few different numbers of cells in series. This will only work well if
        single module voc_ref is given.

        Parameters
        ----------
        voc_ref
        technology

        Returns
        -------

        """

        voc_cell = {'thin-film': 0.7477344670083659,
                    'multi-Si': 0.6207941068112764,
                    'cigs': 0.4972842261904762,
                    'mono-Si': 0.6327717834732666,
                    'cdte': 0.8227840909090908}

        return int(voc_ref / voc_cell[technology])

    def estimate_voc_ref(self, vmp_ref : array, technology : str =None):
        voc_vmp_ratio = {'thin-film': 1.3069503474012514,
                        'multi-Si': 1.2365223483476515,
                        'cigs': 1.2583291018540534,
                        'mono-Si': 1.230866745147029,
                        'cdte': 1.2188176469944012}
        voc_ref = vmp_ref * voc_vmp_ratio[technology]
        

        return voc_ref

    def estimate_beta_voc(self, beta_vmp : float, technology : str ='mono-Si'):
        beta_voc_to_beta_vmp_ratio = {'thin-film': 0.9594252453485964,
                                    'multi-Si': 0.9782579114165342,
                                    'cigs': 0.9757373267198366,
                                    'mono-Si': 0.9768254239046427,
                                    'cdte': 0.9797816054754396}
        beta_voc = beta_vmp * beta_voc_to_beta_vmp_ratio[technology]
        return beta_voc

    def estimate_alpha_isc(self, isc : array, technology : str):
        alpha_isc_to_isc_ratio = {'multi-Si': 0.0005864523754010862,
                                'mono-Si': 0.0005022410194560715,
                                'thin-film': 0.00039741211251133725,
                                'cigs': -8.422066533574996e-05,
                                'cdte': 0.0005573603056215652}

        alpha_isc = isc * alpha_isc_to_isc_ratio[technology]
        return alpha_isc

    def estimate_isc_ref(self, imp_ref : array, technology: str):
        isc_to_imp_ratio = {'multi-Si': 1.0699135787527263,
                            'mono-Si': 1.0671785412770871,
                            'thin-film': 1.158663685900219,
                            'cigs': 1.1566217151572733, 
                            'cdte': 1.0962996330688608}

        isc_ref = imp_ref * isc_to_imp_ratio[technology]

        return isc_ref

    def estimate_resistance_series_simple(self, vmp : array, imp : array,
                                        saturation_current : array,
                                        photocurrent : array,
                                        nNsVth : array):
        Rs = (nNsVth * np.log1p(
            (photocurrent - imp) / saturation_current) - vmp) / imp
        return Rs

    def estimate_singlediode_params(self, poa: array,
                                    temperature_cell : array,
                                    vmp : array,
                                    imp : array,
                                    band_gap_ref : float =1.121,
                                    dEgdT : float =-0.0002677,
                                    alpha_isc : float =None,
                                    cells_in_series : int =None,
                                    technology : string =None,
                                    convergence_test : float =0.0001,
                                    temperature_ref : float =25,
                                    irradiance_ref : float =1000,
                                    resistance_series_ref : float =None,
                                    resistance_shunt_ref : float =600,
                                    figure : bool =False,
                                    figure_number_start : int =20,
                                    imp_model : string ='sandia',
                                    vmp_model : string ='sandia1',
                                    verbose : bool =False,
                                    max_iter : int =10,
                                    optimize_Rs_Io : bool =False,
                                    ):
        """

        Estimate the Desoto single diode model parameters for a PV module.

        Input values for a whole string can be provided, but the best results
        will occur if vmp, imp are given for a single module. This means that the
        user should divide vmp by the number of modules in series and imp by the
        number of strings in parallel before entering into this function.

        cells_in_series is an optional input, but the algorithm will perform
        better if this parameter is provided. cells_in_series should be the total
        number of cells in series, so be sure to multiply by the number of
        modules in series if a string vmp is used.

        alpha_isc is also an optional input, similarly, the algorithm will
        perform better with this provided.

        Parameters
        ----------
        poa
        temperature_cell
        vmp
        imp
        resistance_series_ref
        cells_in_series
        figure

        Returns
        -------

        """
        if verbose:
            print('--\nEstimate singlediode model parameters')

        start_time = time.time()
        if poa.size == 0:
            return dict(
                diode_factor=np.nan,
                photocurrent_ref=np.nan,
                saturation_current_ref=np.nan,
                resistance_series_ref=np.nan,
                conductance_shunt_extra=np.nan,
                v_oc_ref=np.nan,
                i_mp_ref=np.nan,
                i_sc_ref=np.nan,
                v_mp_ref=np.nan,
                p_mp_ref=np.nan,
                alpha_isc=np.nan,
                alpha_imp=np.nan,
                beta_vmp=np.nan,
                beta_voc=np.nan,
                gamma_pmp=np.nan,
                cells_in_series=np.nan,
                nNsVth_ref=np.nan,
            )

        poa = np.array(poa)
        # temperature_module = np.array(temperature_module)
        temperature_cell = np.array(temperature_cell)
        vmp = np.array(vmp)
        imp = np.array(imp)
        figure_number = figure_number_start

        out = self.estimate_imp_ref(poa=poa,
                            temperature_cell=temperature_cell,
                            imp=imp,
                            temperature_ref=temperature_ref,
                            irradiance_ref=irradiance_ref,
                            model=imp_model,
                            figure=figure,
                            figure_number=figure_number,
                            verbose=verbose
                            )
        figure_number += 1

        imp_ref = out['i_mp_ref']
        alpha_imp = out['alpha_imp']

        if verbose:
            print('imp ref: {}'.format(imp_ref))

        out = self.estimate_vmp_ref(
            poa=poa,
            temperature_cell=temperature_cell,
            vmp=vmp,
            temperature_ref=temperature_ref,
            irradiance_ref=irradiance_ref,
            figure=figure,
            figure_number=figure_number,
            model=vmp_model)
        figure_number += 1

        vmp_ref = out['v_mp_ref']
        beta_vmp = out['beta_vmp']

        pmp_ref = vmp_ref * imp_ref
        voc_ref = self.estimate_voc_ref(vmp_ref, technology=technology)
        if cells_in_series == None:
            cells_in_series = self.estimate_cells_in_series(voc_ref,
                                                    technology=technology)

        diode_factor = self.estimate_diode_factor(vmp_ref=vmp_ref,
                                            beta_vmp=beta_vmp,
                                            imp_ref=imp_ref,
                                            technology=technology,
                                            cells_in_series=cells_in_series)

        kB = 1.381e-23
        q = 1.602e-19

        nNsVth_ref = diode_factor * cells_in_series * kB * (
                temperature_ref + 273.15) / q

        beta_voc = self.estimate_beta_voc(beta_vmp, technology=technology)

        photocurrent_ref = self.estimate_photocurrent_ref_simple(imp_ref,
                                                            technology=technology)

        isc_ref = self.estimate_isc_ref(imp_ref, technology=technology)
        if verbose:
            print('Simple isc_ref estimate: {}'.format(isc_ref))

        saturation_current_ref = self.estimate_saturation_current(isc=isc_ref,
                                                            voc=voc_ref,
                                                            nNsVth=nNsVth_ref,
                                                            )
        if verbose:
            print('Simple saturation current ref estimate: {}'.format(
                saturation_current_ref))

        if alpha_isc == None:
            alpha_isc = self.estimate_alpha_isc(isc_ref, technology=technology)

        kB = 1.381e-23
        q = 1.602e-19
        Tref = temperature_ref + 273.15

        nNsVth_ref = diode_factor * cells_in_series * kB * Tref / q

        if resistance_series_ref == None:
            resistance_series_ref = self.estimate_resistance_series_simple(vmp_ref,
                                                                    imp_ref,
                                                                    saturation_current_ref,
                                                                    photocurrent_ref,
                                                                    nNsVth=nNsVth_ref)
        if verbose:
            print(
                'resistance_series_ref estimate: {}'.format(resistance_series_ref))

        results = {}
        num_iter = max_iter
    

        # Output parameters
        params = dict(
            diode_factor=diode_factor,
            photocurrent_ref=photocurrent_ref,
            saturation_current_ref=saturation_current_ref,
            resistance_series_ref=resistance_series_ref,
            resistance_shunt_ref=resistance_shunt_ref,
            conductance_shunt_extra=0.000,
            v_oc_ref=voc_ref,
            i_mp_ref=imp_ref,
            i_sc_ref=isc_ref,
            v_mp_ref=vmp_ref,
            p_mp_ref=pmp_ref,
            alpha_isc=alpha_isc,
            alpha_imp=alpha_imp,
            beta_vmp=beta_vmp,
            beta_voc=beta_voc,
            cells_in_series=cells_in_series,
            nNsVth_ref=nNsVth_ref,
        )

        if verbose:
            print('Elapsed time: {:.2f}'.format(time() - start_time))

        return params, results


"""
Functions about single diode modelling
"""

def estimate_Eg_dEgdT(technology : str):
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

def calcparams_pvpro(effective_irradiance : array, temperature_cell : array,
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

def singlediode_fast(photocurrent : array, 
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

def pvlib_single_diode(
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
        band_gap_ref, dEgdT = estimate_Eg_dEgdT(technology)

    kB = 1.381e-23
    q = 1.602e-19
    nNsVth_ref = diode_factor * cells_in_series * kB / q * (
            273.15 + temperature_ref)

    iph, io, rs, rsh, nNsVth = calcparams_pvpro( effective_irradiance,
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
        out = singlediode_fast(iph,
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

def singlediode_v_from_i(
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

def singlediode_i_from_v(
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

def pv_system_single_diode_model(
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

    out = pvlib_single_diode(
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

        current_closest = singlediode_i_from_v(voltage=voltage_operation[cax],
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

        voltage_closest = singlediode_v_from_i(
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
