import pvlib
import numpy as np
import pandas as pd
# import pytz
from collections import OrderedDict
# from functools import partial
import scipy
import datetime
import os
import warnings
import time

from pvlib.singlediode import _lambertw_i_from_v, _lambertw_v_from_i
from pvlib.pvsystem import calcparams_desoto

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.optimize import basinhopping

from pvpro.estimate import estimate_imp_ref, estimate_singlediode_params
from pvpro.singlediode import pvlib_single_diode, pv_system_single_diode_model, singlediode_closest_point

def production_data_curve_fit(
        temperature_cell,
        effective_irradiance,
        operating_cls,
        voltage,
        current,
        lower_bounds,
        upper_bounds,
        alpha_isc=None,
        diode_factor=None,
        photocurrent_ref=None,
        saturation_current_ref=None,
        resistance_series_ref=None,
        resistance_shunt_ref=None,
        conductance_shunt_extra=None,
        p0=None,
        cells_in_series=72,
        band_gap_ref=1.121,
        verbose=False,
        solver='nelder-mead',
        singlediode_method='newton',
        method='minimize',
        use_mpp_points=True,
        use_voc_points=True,
        use_clip_points=True,
        brute_number_grid_points=2,
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

    if type(p0) == type(None):
        p0 = dict(
            diode_factor=1.0,
            photocurrent_ref=8,
            saturation_current_ref=10,
            resistance_series_ref=10,
            resistance_shunt_ref=10
        )

    #         0: System at maximum power point.
    #         1: System at open circuit conditions.
    #         2: Low irradiance nighttime.
    #         3: Clipped/curtailed operation. Not necessarily at mpp.

    if not use_mpp_points:
        cax = operating_cls != 0
        effective_irradiance = effective_irradiance[cax]
        temperature_cell = temperature_cell[cax]
        operating_cls = operating_cls[cax]
        voltage = voltage[cax]
        current = current[cax]

    if not use_voc_points:
        cax = operating_cls != 1
        effective_irradiance = effective_irradiance[cax]
        temperature_cell = temperature_cell[cax]
        operating_cls = operating_cls[cax]
        voltage = voltage[cax]
        current = current[cax]

    if not use_clip_points:
        cax = operating_cls != 3
        effective_irradiance = effective_irradiance[cax]
        temperature_cell = temperature_cell[cax]
        operating_cls = operating_cls[cax]
        voltage = voltage[cax]
        current = current[cax]

    weights = np.zeros_like(operating_cls)
    weights[operating_cls == 0] = 1
    weights[operating_cls == 1] = 1
    weights[operating_cls == 3] = 1

    if verbose:
        print('Total points: {}'.format(len(operating_cls)))
        print('MPP points: {}'.format(np.sum(operating_cls == 0)))
        print('Voc points: {}'.format(np.sum(operating_cls == 1)))
        print('Clipped points: {}'.format(np.sum(operating_cls == 3)))

    if len(effective_irradiance) == 0 or len(effective_irradiance) == 0 or len(
            operating_cls) == 0 or len(voltage) == 0 or len(current) == 0:
        p = dict(
            diode_factor=np.nan,
            photocurrent_ref=np.nan,
            saturation_current_ref=np.nan,
            resistance_series_ref=np.nan,
            conductance_shunt_extra=np.nan
        )
        print('No valid values received.')
        return p, np.nan, -1

    model_kwargs = dict(
        effective_irradiance=effective_irradiance,
        temperature_cell=temperature_cell,
        operating_cls=operating_cls,
        cells_in_series=cells_in_series,
        band_gap_ref=band_gap_ref,
        voltage_operation=voltage,
        current_operation=current,
        method=singlediode_method,
    )

    if not diode_factor == None:
        model_kwargs['diode_factor'] = diode_factor
    if not photocurrent_ref == None:
        model_kwargs['photocurrent_ref'] = photocurrent_ref
    if not saturation_current_ref == None:
        model_kwargs['saturation_current_ref'] = saturation_current_ref
    if not resistance_series_ref == None:
        model_kwargs['resistance_series_ref'] = resistance_series_ref
    if not resistance_shunt_ref == None:
        model_kwargs['resistance_shunt_ref'] = resistance_shunt_ref
    if not alpha_isc == None:
        model_kwargs['alpha_isc'] = alpha_isc
    if not conductance_shunt_extra == None:
        model_kwargs['conductance_shunt_extra'] = conductance_shunt_extra

    # Functions for translating from optimization quantity (x) to physical parameter (p)
    def x_to_p(x, key):
        """
        Change from numerical fit value (x) to physical parameter (p).

        Parameters
        ----------
        x
        key

        Returns
        -------

        """
        if key == 'diode_factor':
            return x
        elif key == 'photocurrent_ref':
            return x
        elif key == 'saturation_current_ref':
            return np.exp(x - 21)
        elif key == 'resistance_series_ref':
            return x
        elif key == 'resistance_shunt_ref':
            return np.exp(2 * (x - 1))
        elif key == 'conductance_shunt_extra':
            return x

    def p_to_x(p, key):
        if key == 'diode_factor':
            return p
        elif key == 'photocurrent_ref':
            return p
        elif key == 'saturation_current_ref':
            return np.log(p) + 21
        elif key == 'resistance_series_ref':
            return p
        elif key == 'resistance_shunt_ref':
            return np.log(p) / 2 + 1
        elif key == 'conductance_shunt_extra':
            return p

    # note that this will be the order of parameters in the model function.
    fit_params = p0.keys()

    def model(x):
        p = model_kwargs.copy()
        n = 0
        for param in fit_params:
            p[param] = x_to_p(x[n], param)
            n = n + 1
        voltage_fit, current_fit = pv_system_single_diode_model(**p)

        # For clipped points, need to calculate

        return voltage_fit, current_fit

    def residual(x):
        voltage_fit, current_fit = model(x)
        return np.nanmean((np.abs(voltage_fit - voltage) * weights) ** 2 + \
                          (np.abs(current_fit - current) * weights) ** 2)

    # print(signature(model))

    # print('Starting residual: ',
    #       residual([p_to_x(p0[k], k) for k in fit_params]))

    if method == 'minimize':
        bounds = scipy.optimize.Bounds(
            [p_to_x(lower_bounds[k], k) for k in fit_params],
            [p_to_x(upper_bounds[k], k) for k in fit_params])

        x0 = [p_to_x(p0[k], k) for k in fit_params]

        # print('p0: ', p0)
        # print('bounds:', bounds)

        # print('Method: {}'.format(solver))

        res = scipy.optimize.minimize(residual,
                                      x0=x0,
                                      bounds=bounds,
                                      method=solver,
                                      options=dict(
                                          # maxiter=100,
                                          disp=verbose,
                                          # ftol=0.001,
                                      ),
                                      )

        # print(res)
        n = 0
        p_fit = {}
        for param in fit_params:
            p_fit[param] = x_to_p(res.x[n], param)
            n = n + 1

        # print('Best fit parameters (with scale included):')
        # for p in x_fit:
        #     print('{}: {}'.format(p, x_fit[p]))
        # print('Final Residual: {}'.format(res['fun']))
        return p_fit, res['fun'], res
    elif method == 'basinhopping':
        # lower_bounds_x = [p_to_x(lower_bounds[k], k) for k in fit_params]
        # upper_bounds_x = [p_to_x(upper_bounds[k], k) for k in fit_params]
        x0 = [p_to_x(p0[k], k) for k in fit_params]

        res = basinhopping(residual,
                           x0=x0,
                           niter=100,
                           T=0.2,
                           stepsize=0.1)
        n = 0
        p_fit = {}
        for param in fit_params:
            p_fit[param] = x_to_p(res.x[n], param)
            n = n + 1

        # print('Best fit parameters (with scale included):')
        # for p in x_fit:
        #     print('{}: {}'.format(p, x_fit[p]))
        # print('Final Residual: {}'.format(res['fun']))
        return p_fit, res['fun'], res
