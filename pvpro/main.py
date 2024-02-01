from array import array
from typing import Union

import pvlib
import numpy as np
import pandas as pd
import time
import scipy
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.optimize import minimize
from tqdm import tqdm
from functools import partial
from scipy.special import lambertw
from numpy.linalg import pinv
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error

from solardatatools import DataHandler
from rdtools.degradation import degradation_year_on_year
from pvpro.modeling import estimate_Eg_dEgdT, calcparams_pvpro, single_diode_predict
from pvpro.modeling import singlediode_fast, pvlib_single_diode, pv_system_single_diode_model


"""
Class for running pvpro analysis.
"""
class PvProHandler:
    
    def __init__(self,
                 df : pd.DataFrame,
                 system_name : str ='Unknown',
                 voltage_key : str =None,
                 current_key : str =None,
                 temperature_cell_key : str ='temperature_cell',
                 temperature_ambient_key : str =None,
                 irradiance_poa_key : str =None,
                 modules_per_string : int =None,
                 parallel_strings : int =None,
                 alpha_isc : float =None,
                 resistance_shunt_ref : float = 600,
                 cells_in_series : int =None,
                 technology : str =None,
                 days_per_run: int = 14,
                 disable_tqdm: bool = False,
                 include_operating_cls: bool = True
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
        self.days_per_run = days_per_run
        self.disable_tqdm = disable_tqdm
        self.include_operating_cls = include_operating_cls

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
                boolean_mask : list[bool] = None,
                days_per_run : float = 14,
                iterations_per_year : int = 26,
                start_point_method : str ='fixed',
                use_mpp_points : bool =True,
                use_voc_points : bool =True,
                use_clip_points : bool =True,
                diode_factor : float =None,
                photocurrent_ref : float =None,
                saturation_current_ref : float =None,
                resistance_series_ref : float =None,
                resistance_shunt_ref : float =None,
                conductance_shunt_extra : float =0,
                verbose : bool =False,
                method : str ='minimize',
                solver : str ='L-BFGS-B',
                fit_params : list[str] = None,
                lower_bounds : dict = None,
                upper_bounds : dict = None,
                singlediode_method : str ='fast',
                saturation_current_multistart : list[float] = None
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

        q = 1.602e-19
        kB = 1.381e-23

        # Calculate iteration start days
        dataset_length_days = (self.df.index[-1] - self.df.index[0]).days + 1
        iteration_start_days = np.arange(0, dataset_length_days - days_per_run + 1,
                                         365.25 / iterations_per_year)

        # Fit params taken from p0
        if fit_params == None:
            fit_params = ['photocurrent_ref', 'saturation_current_ref',
                          'resistance_series_ref', 
                         'conductance_shunt_extra', 
                         'resistance_shunt_ref', 
                          'diode_factor']

        if saturation_current_multistart is None:
            saturation_current_multistart = [0.2, 0.5, 1, 2, 5]

        # Calculate time midway through each iteration.
        self.time = []
        for d in iteration_start_days:
            self.time.append(self.df.index[0] +
                             timedelta(int(d + 0.5 * days_per_run)))

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
        for k in tqdm(iteration, disable=self.disable_tqdm):

            # Get df for current iteration.
            if boolean_mask is None:
                boolean_mask = np.ones(len(self.df), dtype=np.bool)
            else:
                boolean_mask = np.array(boolean_mask)

            if not len(self.df) == len(boolean_mask):
                raise ValueError(
                    'Boolean mask has length {}, it must have same length as self.df: {}'.format(
                        len(boolean_mask), len(self.df)))

            pfit.loc[k, 't_start'] = self.df.index[0] + timedelta(
                iteration_start_days[k])
            pfit.loc[k, 't_end'] = self.df.index[0] + timedelta(
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
                    raise ValueError('start_point_method must be "fixed" or "last"')

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

                n = n + 1
                k_last_iteration = k

                # Calculate other parameters vs. time.
                pfit.loc[k, 'nNsVth_ref'] = pfit.loc[k, 'diode_factor'] * self.cells_in_series * kB / q * (25 + 273.15)

                out = pvlib.pvsystem.singlediode(
                    photocurrent=pfit.loc[k, 'photocurrent_ref'],
                    saturation_current=pfit.loc[k, 'saturation_current_ref'],
                    resistance_series=pfit.loc[k, 'resistance_series_ref'],
                    resistance_shunt = pfit.loc[k, 'resistance_shunt_ref'], 
                    nNsVth=pfit.loc[k, 'nNsVth_ref'])

                for p in out.keys():
                    pfit.loc[k, p + '_ref'] = out[p]

        pfit['t_mean'] = pfit['t_start'] + timedelta(
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

        print('Elapsed time: {:.2f} min'.format((time.time() - start_time) / 60))

    def _x_to_p(self, x : pd.Series, key : str):
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

    def _p_to_x(self, p : pd.Series, key : str):
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

    def _pvpro_L2_loss(self, x : pd.Series, sdm : 'function', voltage : pd.Series, current : pd.Series, voltage_scale : float, current_scale : float,
                    weights : float, fit_params : list):
        voltage_fit, current_fit = sdm(
            **{param: self._x_to_p(x[n], param) for n, param in
            zip(range(len(x)), fit_params)}
        )

        # Note that summing first and then calling nanmean is slightly faster.
        return np.nanmean(((voltage_fit - voltage) * weights / voltage_scale) ** 2 + \
                        ((current_fit - current) * weights / current_scale) ** 2)

    def production_data_curve_fit(self,
        temperature_cell : pd.Series,
        effective_irradiance : pd.Series,
        operating_cls : pd.Series,
        voltage : pd.Series,
        current : pd.Series,
        cells_in_series : int =60,
        band_gap_ref : float = None,
        dEgdT : float = None,
        p0 : dict =None,
        lower_bounds : float =None,
        upper_bounds : float =None,
        alpha_isc : float =None,
        diode_factor : pd.Series =None,
        photocurrent_ref : pd.Series =None,
        saturation_current_ref : pd.Series =None,
        resistance_series_ref : pd.Series =None,
        resistance_shunt_ref : pd.Series =None,
        conductance_shunt_extra : pd.Series =None,
        verbose : bool =False,
        solver : str ='L-BFGS-B',
        singlediode_method : str ='fast',
        method : str ='minimize',
        use_mpp_points : bool =True,
        use_voc_points : bool =True,
        use_clip_points : bool =True,
        # fit_params=None,
        saturation_current_multistart : pd.Series =None
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

                    p0['saturation_current_ref'] = saturation_current_ref_start * Io_multiplier
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

            n = 0
            try:
                p_fit = {}
                for param in fit_params:
                    p_fit[param] = self._x_to_p(res.x[n], param)
                    n = n + 1

                # remove the Gsh_extra
                p_fit['resistance_shunt_ref'] = 1/(1/p_fit['resistance_shunt_ref']-1e-5)

                out = {'p': p_fit,
                    'fixed_params': fixed_params,
                    'residual': res['fun'],
                    'x0': x0,
                    'p0': p0,
                    }
                for k in res:
                    out[k] = res[k]
            except:
                out = {'p': {},
                    'fixed_params': fixed_params,
                    'residual': np.nan,
                    'x0': x0,
                    'p0': p0,
                    }

            return out

    def estimate_p0(self,
                    verbose : bool =False,
                    boolean_mask : bool =None,
                    technology : str =None):
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

    def build_plot_text_str(self, df : pd.DataFrame, p_plot : dict):

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

    def remove_outliers(self, pfit : pd.DataFrame, nstd: int = 3):

        """
        Remove outliers whose difference to the mean value > nstd * std value

        """

        for k in ['photocurrent_ref', 'saturation_current_ref',
                'resistance_series_ref', 'resistance_shunt_ref', 'i_sc_ref',
                'v_oc_ref', 'i_mp_ref', 'v_mp_ref', 'p_mp_ref']:
            
            res = pfit[k]
            pfit[k] = res[np.abs(res-np.nanmean(res))< nstd * np.nanstd(res)]

        return pfit

    def analyze_yoy(self, pfit : pd.DataFrame):

        """
        Analzye the year of year (YOY) trend of parameters using Rdtools

        :param pfit: dataframe of the data
        :return out: dataframe containing YOY results

        """
        out = {}

        for k in ['photocurrent_ref', 'saturation_current_ref',
                'resistance_series_ref', 'resistance_shunt_ref','diode_factor', 'i_sc_ref',
                'v_oc_ref', 'i_mp_ref', 'v_mp_ref', 'p_mp_ref']:
                   
            if k in pfit:

                res = pfit[k]
                Rd_pct, Rd_CI, calc_info = degradation_year_on_year(pd.Series(res[~np.isnan(res)]),
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

    def run_pipeline(self):
        """
        Run pipeline of parameter extraction

        :param remove_outliers: remove outliers
        :return pfit: dataframe containing extracted SDM parameters
        
        """

        df = self.df
        if self.include_operating_cls:
            boolean_mask = np.logical_and.reduce((
                                       np.logical_not(df['current_irradiance_outliers']),
                                       np.logical_not(df['voltage_temperature_outliers']),
                                       df[self.irradiance_poa_key]>100,
                                       df['operating_cls']==0
                                    ))

        else:
            boolean_mask = np.logical_and.reduce((np.logical_not(df['current_irradiance_outliers']),
                                        np.logical_not(df['voltage_temperature_outliers']),
                                        df[self.irradiance_poa_key]>100
                                        ))
        
        # Estimate initial parameters
        self.estimate_p0(boolean_mask=boolean_mask, technology = self.technology)
        
        # Diode factor is set constant
        self.p0['diode_factor'] = 1

        hyperparams = {
            'use_voc_points': False,
            'use_mpp_points': True,
            'use_clip_points': False,
            'method': 'minimize',
            'solver': 'L-BFGS-B',
            'days_per_run': self.days_per_run,
            'iterations_per_year': int(365/self.days_per_run),
            'start_point_method': 'last',
            'saturation_current_multistart': [1],
            'verbose': False,
            'diode_factor': self.p0['diode_factor']
        }

        fit_params = ['photocurrent_ref', 'saturation_current_ref','resistance_series_ref', 
              'conductance_shunt_extra', 'resistance_shunt_ref', 'diode_factor']
        
        lower_bounds = dict(
                    diode_factor=0.1,
                    photocurrent_ref=0.01,
                    saturation_current_ref=5e-12,
                    resistance_series_ref=0.1,
                    conductance_shunt_extra=0,
                    resistance_shunt_ref=100
                )

        upper_bounds = dict(
            diode_factor=1.5,
            photocurrent_ref=20,
            saturation_current_ref=1e-6,
            resistance_series_ref=2,
            conductance_shunt_extra=0,
            resistance_shunt_ref=1000
        )

        self.execute(iteration='all', fit_params= fit_params,
                        lower_bounds=lower_bounds,
                        upper_bounds=upper_bounds,
                        **hyperparams,
                        boolean_mask = boolean_mask)
        
        pfit = self.result['p']

        return pfit

    def system_modelling(self, df : pd.DataFrame, pfit : dict, inx : list[bool] = None):

        """
        Model the PV sytem using the SDM parameters extracted by PV-Pro

        :param df: dataframe of the data 
        :param pfit: dict containing the SDM parameters
        :param inx: index of dataframe to model

        :return system_power: modeled power using the given SDM parameters and weather data

        """

        if inx is None:
            inx = [True]*len(df)

        out_test = pvlib_single_diode(
            effective_irradiance = df[self.irradiance_poa_key][inx],
            temperature_cell = df['temperature_cell'][inx],
            resistance_shunt_ref = pfit['resistance_shunt_ref'],
            resistance_series_ref = pfit['resistance_series_ref'],
            diode_factor = pfit['diode_factor'],
            cells_in_series = self.cells_in_series,
            alpha_isc = self.alpha_isc,
            photocurrent_ref = pfit['photocurrent_ref'],
            saturation_current_ref = pfit['saturation_current_ref'],
            band_gap_ref=1.121,
            dEgdT=-0.0002677,
            conductance_shunt_extra=pfit['conductance_shunt_extra'],
            irradiance_ref=1000,
            temperature_ref=25,
            ivcurve_pnts=None,
            output_all_params=False,
            singlediode_method='fast',
            calculate_voc=True)
        
        modules_per_string = self.modules_per_string
        parallel_strings = self.parallel_strings

        system_power = out_test['v_mp']* out_test['i_mp']*modules_per_string*parallel_strings

        return system_power

    """
    off-MPP functions
    """
    def detect_off_MPP(self, boolean_mask : array = None):

        """
        detect off-MPP based on Pmp error

        return
        ------
        off-MPP bool array
        
        """
        if boolean_mask is not None:
            df=self.df[boolean_mask]
        else:
            df=self.df

        p_plot=self.p0

        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, self.voltage_key]) / self.modules_per_string
        imp = np.array(df.loc[mask, self.current_key]) / self.parallel_strings

        # calculate error
        v_esti, i_esti = single_diode_predict(self,
            effective_irradiance=df[self.irradiance_poa_key][mask],
            temperature_cell=df[self.temperature_cell_key][mask],
            operating_cls=np.zeros_like(df[self.irradiance_poa_key][mask]),
            params=p_plot)
        rmse_vmp = mean_squared_error(v_esti, vmp)/37
        rmse_imp = mean_squared_error(i_esti, imp)/8.6

        # Pmp error
        pmp_error = abs(vmp*imp - v_esti*i_esti)
        vmp_error = abs(vmp-v_esti)
        imp_error = abs(imp-i_esti)

        # detect off-mpp and calculate off-mpp percentage
        offmpp = pmp_error>np.nanmean(pmp_error)+np.std(pmp_error)
        offmpp_ratio = offmpp.sum()/pmp_error.size*100  

        return offmpp

    def deconvolve_Pmp_error_on_V_I (self,  boolean_mask : array, points_show : array = None, figsize : list =[4.5,3], 
                                sys_name : str = None, date_text : str = None):

        p_plot=self.p0

        points_show_bool = np.full(boolean_mask.sum(), False)
        points_show_bool[points_show] = True
        df=self.df[boolean_mask][points_show_bool]
        
        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, self.voltage_key]) / self.modules_per_string
        imp = np.array(df.loc[mask, self.current_key]) / self.parallel_strings
        G = df[self.irradiance_poa_key][mask]
        Tm = df[self.temperature_cell_key][mask]

        # estimate
        v_esti, i_esti = single_diode_predict(self,
            effective_irradiance=G,
            temperature_cell=Tm,
            operating_cls=np.zeros_like(df[self.irradiance_poa_key][mask]),
            params=p_plot)

        # error
        pmp_error = abs(vmp*imp - v_esti*i_esti) 
        vmp_error = abs(vmp-v_esti)
        imp_error = abs(imp-i_esti)
        pmp_error = pmp_error + vmp_error*imp_error

        # contribution
        con_V = vmp_error*i_esti/pmp_error*100
        con_I = imp_error*v_esti/pmp_error*100

        fig, ax = plt.subplots(figsize=figsize)
        xtime = df.index[mask]

        # ax.fill_between(xtime, np.ones_like(con_V)*100, 0, alpha=1, color='#0070C0', edgecolor = 'white', linewidth = 2, label = 'Error of Vmp')
        ax.fill_between(xtime, con_I+con_V, 0, alpha=1, color='#0070C0', edgecolor = 'white', linewidth = 2, label = 'Error of Vmp')
        ax.fill_between(xtime, con_I, 0, alpha=1, color='#92D050', edgecolor = 'white', linewidth = 2, label = 'Error of Imp')

        
        hours = mdates.HourLocator(interval = 1)
        h_fmt = mdates.DateFormatter('%Hh')
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)

        # text
        if not date_text:
            datetext = df.index[mask][0].strftime("%Y-%m-%d")
        text_show = sys_name + '\n' + datetext
        ax.text(xtime[1], 10, text_show, fontsize=10)

        ax.tick_params(labelsize=10)
        ax.tick_params(labelsize=10)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Deconvolution\n of Pmp error (%)', fontsize=10, fontweight = 'bold')
        plt.ylim([0,100])
        plt.legend(loc=7, fontsize=10)
        plt.gcf().set_dpi(120)
        plt.show()

    def deconvolve_Pmp_error_on_V_I_line (self,  boolean_mask : array, points_show : array = None, figsize : list =[3,3], 
                                sys_name : str = None, date_text : str = None, title : str = None):

        p_plot=self.p0

        points_show_bool = np.full(boolean_mask.sum(), False)
        points_show_bool[points_show] = True
        df=self.df[boolean_mask][points_show_bool]
        
        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, self.voltage_key]) / self.modules_per_string
        imp = np.array(df.loc[mask, self.current_key]) / self.parallel_strings
        G = df[self.irradiance_poa_key][mask]
        Tm = df[self.temperature_cell_key][mask]

        # estimate
        v_esti, i_esti = single_diode_predict(self,
            effective_irradiance=G,
            temperature_cell=Tm,
            operating_cls=np.zeros_like(df[self.irradiance_poa_key][mask]),
            params=p_plot)

        # error
        pmp_error = abs(vmp*imp - v_esti*i_esti) 
        vmp_error = abs(vmp-v_esti)
        imp_error = abs(imp-i_esti)
        pmp_error = pmp_error #+ vmp_error*imp_error

        # contribution
        con_V = vmp_error*i_esti/pmp_error*100
        con_I = imp_error*v_esti/pmp_error*100

        fig, ax = plt.subplots(figsize=figsize)
        xtime = df.index[mask]

        # ax.fill_between(xtime, np.ones_like(con_V)*100, 0, alpha=1, color='#0070C0', edgecolor = 'white', linewidth = 2, label = 'Error of Vmp')
       
        ax.plot(xtime, con_V, '-o',color='#92D050',  linewidth = 2, label = 'Contribution of Vmp')
        ax.plot(xtime, con_I, '-o', color='#00B0F0', linewidth = 2, label = 'Contribution of Imp')
        ax.plot([xtime[0], xtime[-1]], [np.mean(con_V),np.mean(con_V)], '--', color='#00A44B', linewidth = 2)
        ax.plot([xtime[0], xtime[-1]], [np.mean(con_I),np.mean(con_I)], '--', color='#0070C0', linewidth = 2)
        ax.text(xtime[0], np.mean(con_V)+2, 'Mean: {:.2f}%'.format(np.mean(con_V)), fontsize=10)
        ax.text(xtime[0], np.mean(con_I)+2, 'Mean: {:.2f}%'.format(np.mean(con_I)), fontsize=10)
        
        hours = mdates.HourLocator(interval = 1)
        h_fmt = mdates.DateFormatter('%Hh')
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)

        # text
        if not date_text:
            datetext = df.index[mask][0].strftime("%Y-%m-%d")
        text_show = sys_name + '\n' + datetext

        ax.set_title(title, fontweight = 'bold', fontsize = 11)
        ax.text(xtime[0], 6, text_show, fontsize=11)
        ax.grid()

        ax.tick_params(labelsize=11)
        ax.tick_params(labelsize=11)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Contribution\n to Pmp error (%)', fontsize=11, fontweight = 'bold')
        plt.ylim([0,100])
        plt.legend(loc=7, fontsize=11)
        plt.gcf().set_dpi(120)
        plt.show()
        return fig


"""
Class for estimation of initial paramters
"""
class EstimateInitial:

    def __init__(self,
                 system_name : str ='Unknown'):
        self.name = system_name
        
    def estimate_imp_ref(self, poa : pd.Series,
                     temperature_cell : pd.Series,
                     imp : pd.Series,
                     poa_lower_limit : float =200,
                     irradiance_ref : float =1000,
                     temperature_ref : float =25,
                     figure : bool =False,
                     figure_number : int =20 ,
                     model : str ='sandia',
                     solver : str ='huber',
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

    def estimate_vmp_ref(self, poa : pd.Series,
                        temperature_cell : pd.Series,
                        vmp : pd.Series,
                        irradiance_ref : float =1000,
                        temperature_ref : float =25,
                        figure : bool =False,
                        figure_number : int =21,
                        model: str ='sandia1',
                        solver: str ='huber',
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

    def estimate_diode_factor(self, vmp_ref : pd.Series, beta_vmp : float, imp_ref : pd.Series,
                            alpha_isc_norm  : float =0,
                            resistance_series : float =0.35,
                            cells_in_series : int =60,
                            temperature_ref : float =25,
                            technology : str =None):

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

    def estimate_photocurrent_ref_simple(self, imp_ref: pd.Series, technology : str ='mono-Si'):
        photocurrent_imp_ratio = {'multi-Si': 1.0746167586063207,
                                'mono-Si': 1.0723051517913444,
                                'thin-film': 1.1813401654607967,
                                'CIGS': 1.1706462692015707,
                                'cdte': 1.1015249105470803}

        photocurrent_ref = imp_ref * photocurrent_imp_ratio[technology]

        return photocurrent_ref

    def estimate_saturation_current(self, isc : pd.Series, voc : pd.Series, nNsVth : pd.Series):
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

    def estimate_cells_in_series(self, voc_ref : pd.Series, technology : str ='mono-Si'):
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
                    'CIGS': 0.4972842261904762,
                    'mono-Si': 0.6327717834732666,
                    'cdte': 0.8227840909090908}

        return int(voc_ref / voc_cell[technology])

    def estimate_voc_ref(self, vmp_ref : pd.Series, technology : str):
        voc_vmp_ratio = {'thin-film': 1.3069503474012514,
                        'multi-Si': 1.2365223483476515,
                        'CIGS': 1.2583291018540534,
                        'mono-Si': 1.230866745147029,
                        'cdte': 1.2188176469944012}
        voc_ref = vmp_ref * voc_vmp_ratio[technology]
        

        return voc_ref

    def estimate_beta_voc(self, beta_vmp : float, technology : str ='mono-Si'):
        beta_voc_to_beta_vmp_ratio = {'thin-film': 0.9594252453485964,
                                    'multi-Si': 0.9782579114165342,
                                    'CIGS': 0.9757373267198366,
                                    'mono-Si': 0.9768254239046427,
                                    'cdte': 0.9797816054754396}
        beta_voc = beta_vmp * beta_voc_to_beta_vmp_ratio[technology]
        return beta_voc

    def estimate_alpha_isc(self, isc : pd.Series, technology : str):
        alpha_isc_to_isc_ratio = {'multi-Si': 0.0005864523754010862,
                                'mono-Si': 0.0005022410194560715,
                                'thin-film': 0.00039741211251133725,
                                'CIGS': -8.422066533574996e-05,
                                'cdte': 0.0005573603056215652}

        alpha_isc = isc * alpha_isc_to_isc_ratio[technology]
        return alpha_isc

    def estimate_isc_ref(self, imp_ref : pd.Series, technology: str):
        isc_to_imp_ratio = {'multi-Si': 1.0699135787527263,
                            'mono-Si': 1.0671785412770871,
                            'thin-film': 1.158663685900219,
                            'CIGS': 1.1566217151572733, 
                            'cdte': 1.0962996330688608}

        isc_ref = imp_ref * isc_to_imp_ratio[technology]

        return isc_ref

    def estimate_resistance_series_simple(self, vmp : pd.Series, imp : pd.Series,
                                        saturation_current : pd.Series,
                                        photocurrent : pd.Series,
                                        nNsVth : pd.Series):
        Rs = (nNsVth * np.log1p(
            (photocurrent - imp) / saturation_current) - vmp) / imp
        return Rs

    def estimate_singlediode_params(self, poa: pd.Series,
                                    temperature_cell : pd.Series,
                                    vmp : pd.Series,
                                    imp : pd.Series,
                                    band_gap_ref : float =1.121,
                                    dEgdT : float =-0.0002677,
                                    alpha_isc : float =None,
                                    cells_in_series : int =None,
                                    technology : str =None,
                                    convergence_test : float =0.0001,
                                    temperature_ref : float =25,
                                    irradiance_ref : float =1000,
                                    resistance_series_ref : float =None,
                                    resistance_shunt_ref : float =600,
                                    figure : bool =False,
                                    figure_number_start : int =20,
                                    imp_model : str ='sandia',
                                    vmp_model : str ='sandia1',
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
                            figure_number=figure_number
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
Functions about error calculation
"""

def calculate_error_real(pfit : pd.DataFrame, df_ref : pd.DataFrame, nrolling : int = 1):
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

def calculate_error_synthetic(pfit : pd.DataFrame, df : pd.DataFrame, nrolling : int = 1):
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
        mask = ~np.isnan(p1) & ~np.isnan(p2)
        
        all_error_df.loc[k, 'rms'] = np.sqrt(np.mean((p1[mask]-p2[mask]) ** 2))
        all_error_df.loc[k,'rms_rela'] = np.sqrt(np.nanmean(((p1[mask]-p2[mask])/p1[mask]) ** 2))
        all_error_df.loc[k, 'corr_coef'] = np.corrcoef(p1[mask], 
                                                p2[mask])[0,1]

    return all_error_df

"""
Functions for irradiance-to-power conversion of power prediction
"""

def str2date(date_string: str):
    """
    Convert string to datetime format

    :param date_string: string to convert
    :return datetime_object
    
    """
    date_format = "%Y-%m-%d"
    if isinstance(date_string, datetime):
        datetime_object = date_string
    else:
        datetime_object = datetime.strptime(date_string, date_format)

    return datetime_object

def get_train_test_index(df, test_start_date : str, train_days : int, test_days : int = 1):
    
    """
    Get index of training data and test data for the power prediction

    :param df: dataframe containing the data
    :param test_start_date: start date of the test
    :param train_days: length of historical data to extract SDM parameters
    :param test_days: length of test data (default 1 day)

    :return inx_train: index of training data
    :return inx_test: index of test data
    
    """

    test_start_date = str2date(test_start_date)
    train_start_date = test_start_date - timedelta(days=int(train_days))
    test_end_date = test_start_date + timedelta(days=int(test_days))

    inx_train = (df.index >= train_start_date) & (df.index < test_start_date)
    inx_test = (df.index >= test_start_date) & (df.index < test_end_date)

    return inx_train, inx_test

def calc_err(y_pred: list, y_ref : pd.DataFrame, nominal_power : float):

    """
    Calculate several error metrics between predicted and reference power

    :param y_pred: predicted power
    :param y_ref: reference power
    :param nominal_power: nominal power of the PV system
    :return err_dict: dict of errors

    """

    mean_ref = np.nanmean(y_ref)
    mae = np.nanmean(np.abs(y_pred - y_ref))
    mbe = np.nanmean(y_pred - y_ref)
    rmse = np.sqrt(np.nanmean(np.abs(y_pred - y_ref)**2))

    nmae_av = mae/mean_ref*100
    nmbe_av = mbe/mean_ref*100
    nrmse_av = rmse/mean_ref*100

    nmae = mae/nominal_power*100
    nmbe = mbe/nominal_power*100
    nrmse = rmse/nominal_power*100

    r2 = 1- np.sum((y_pred - y_ref)**2)/np.sum((y_ref-mean_ref)**2)

    err_dict = {'nMAE': nmae, 'nRMSE': nrmse, 'nMBE': nmbe, 
                'nMAEav': nmae_av, 'nRMSEav': nrmse_av, 'nMBEav': nmbe_av, 
                'R2': r2}

    return err_dict