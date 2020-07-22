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

from solardatatools import DataHandler
from scipy.optimize import basinhopping
from pvlib.temperature import sapm_cell_from_module

def pvlib_single_diode(
        effective_irradiance,
        temperature_cell,
        resistance_shunt_ref,
        resistance_series_ref,
        diode_factor,
        cells_in_series,
        alpha_isc,
        photocurrent_ref,
        saturation_current_ref,
        Eg_ref=1.121,
        dEgdT=-0.0002677,
        conductance_shunt_extra=0,
        reference_irradiance=1000,
        reference_temperature=25,
        method='newton',
        verbose=False,
        ivcurve_pnts=None,
        output_all_params=False
):
    """
    Find points of interest on the IV curve given module parameters and
    operating conditions.

    method 'newton is about twice as fast as method 'lambertw

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

    kB = 1.381e-23
    q = 1.602e-19

    # photocurrent = effective_irradiance/reference_irradiance* \
    #                (reference_photocurrent + alpha_isc*(temp_cell - reference_temperature))

    # Eg = reference_Eg*(1 - 0.0002677*(temperature_cell - reference_temperature))

    # saturation_current = reference_saturation_current*\
    #                      (temperature_cell+217.15)**3/(reference_temperature+217.15)**3*\
    #                      np.exp( q/kB*(reference_Eg/(reference_temperature+273.15)  - Eg/(temp_cell+273.15) ))
    #

    # nNsVth = diode_ideality_factor*cells_in_series*kB*(temp_cell+273.15)/q

    # a_ref = diode_factor*cells_in_series*kB/q*(273.15 + temperature_cell)

    p = {}

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
        EgRef=Eg_ref,
        dEgdT=dEgdT,
        irrad_ref=reference_irradiance,
        temp_ref=reference_temperature
    )

    rsh = 1 / (1 / rsh + conductance_shunt_extra)

    # if verbose:
    #     print('Photocurrent')
    #     print(photocurrent)
    #     print('Eg')
    #     print(Eg)
    #     print('Saturation current')
    #     print(saturation_current)
    #     print('nNsVth')
    #     print(nNsVth)

    out = pvlib.pvsystem.singlediode(iph,
                                     io,
                                     rs,
                                     rsh,
                                     nNsVth,
                                     method=method,
                                     ivcurve_pnts=ivcurve_pnts,
                                     )

    if output_all_params:

        params = {'photocurrent': iph,
                  'saturation_current': io,
                  'resistance_series': rs,
                  'resistace_shunt': rsh,
                  'nNsVth': nNsVth}

        for p in params:
            out[p] = params[p]

    # out = pvlib.singlediode(photocurrent, saturation_current,
    #                   resistance_series,
    #                   resistance_shunt, nNsVth,
    #                   method=method,
    #                   calculate_all=calculate_all)

    return out


def singlediode_closest_point(
        voltage,
        current,
        effective_irradiance,
        temperature_cell,
        resistance_shunt_ref,
        resistance_series_ref,
        diode_factor,
        cells_in_series,
        alpha_isc,
        photocurrent_ref,
        saturation_current_ref,
        Eg_ref=1.121,
        dEgdT=-0.0002677,
        reference_irradiance=1000,
        reference_temperature=25,
        method='newton',
        verbose=False,
        ivcurve_pnts=1000):
    out = pvlib_single_diode(
        effective_irradiance=effective_irradiance,
        temperature_cell=temperature_cell,
        cells_in_series=cells_in_series,
        alpha_isc=alpha_isc,
        ivcurve_pnts=ivcurve_pnts,
        resistance_series_ref=resistance_series_ref,
        resistance_shunt_ref=resistance_shunt_ref,
        diode_factor=diode_factor,
        photocurrent_ref=photocurrent_ref,
        saturation_current_ref=saturation_current_ref,
        Eg_ref=Eg_ref,
        dEgdT=dEgdT,
        reference_irradiance=reference_irradiance,
        reference_temperature=reference_temperature,
        calculate_all=True,
        method=method,
        verbose=verbose,
    )

    # point_error[k] = np.sqrt(np.min(
    #     (out['v'] - dfc['v_operation'][k]) ** 2 / out['v_oc'] ** 2 + (
    #             out['i'] - dfc['i_operation'][k]) ** 2 / out['i_sc'] ** 2))

    distance_to_curve_square = (out['v'] - voltage) ** 2 / out['v_oc'] ** 2 + \
                               (out['i'] - current) ** 2 / out['i_sc'] ** 2

    closest_distance_idx = np.argmin(distance_to_curve_square)

    return {
        'v_closest': out['v'][closest_distance_idx],
        'i_closest': out['i'][closest_distance_idx],
        'p_closest': out['i'][closest_distance_idx] * out['v'][
            closest_distance_idx],
        'v_target': voltage,
        'i_target': current,
        'p_target': current * voltage,
        'distance': np.sqrt(distance_to_curve_square[closest_distance_idx]),
        'v': out['v'],
        'i': out['i'],
    }


def singlediode_v_from_i(
        current,
        effective_irradiance,
        temperature_cell,
        resistance_shunt_ref,
        resistance_series_ref,
        diode_factor,
        cells_in_series,
        alpha_isc,
        photocurrent_ref,
        saturation_current_ref,
        Eg_ref=1.121,
        dEgdT=-0.0002677,
        reference_irradiance=1000,
        reference_temperature=25,
        method='newton',
        verbose=False,
):
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
        EgRef=Eg_ref,
        dEgdT=dEgdT,
    )
    voltage = _lambertw_v_from_i(rsh, rs, nNsVth, current, io, iph)

    return voltage


def singlediode_i_from_v(
        voltage,
        effective_irradiance,
        temperature_cell,
        resistance_shunt_ref,
        resistance_series_ref,
        diode_factor,
        cells_in_series,
        alpha_isc,
        photocurrent_ref,
        saturation_current_ref,
        Eg_ref=1.121,
        dEgdT=-0.0002677,
        reference_irradiance=1000,
        reference_temperature=25,
        method='newton',
        verbose=False,
):
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
        EgRef=Eg_ref,
        dEgdT=dEgdT,
    )
    current = _lambertw_i_from_v(rsh, rs, nNsVth, voltage, io, iph)

    return current


def classify_operating_mode(voltage, current,
                            power_clip=np.inf,
                            method='fraction'):
    """

    Parameters
    ----------
    voltage
    current
    method

    Returns
    -------
    operating_cls : array

        Array of classifications of each time stamp.
        -1: Unclassified
        0: System at maximum power point.
        1: System at open circuit conditions.
        2: Low irradiance nighttime.
        3: Clipped/curtailed operation. Not necessarily at mpp.

    """

    cls = np.zeros(np.shape(voltage)) - 1

    # Inverter on
    cls[np.logical_and(
        voltage > voltage.max() * 0.01,
        current > current.max() * 0.01,
    )] = 0

    # Nighttime, low voltage and irradiance.
    cls[np.logical_and(current < current.max() * 0.01,
                       voltage < voltage.max() * 0.01)] = 2

    # Inverter off
    cls[np.logical_and(current < current.max() * 0.01,
                       voltage > voltage.max() * 0.01)] = 1

    # Clipped data. Easy algorithm.
    cls[current * voltage > power_clip] = 3

    return cls


def pv_system_single_diode_model(
        effective_irradiance,
        temperature_cell,
        operating_cls,
        diode_factor,
        photocurrent_ref,
        saturation_current_ref,
        resistance_series_ref,
        conductance_shunt_extra,
        resistance_shunt_ref,
        cells_in_series,
        alpha_isc,
        voltage_operation=None,
        current_operation=None,
        band_gap_ref=1.121,
        dEgdT=-0.0002677,
        method='lambertw',
        **kwargs
):
    """

    Fit function, reorganize Vmp, Imp outputs of size (1*N,) into a size (
    N*2,) vector. This is important because this is how
    scipy.optimize.curve_fit works.

    Parameters
    ----------
    X :

        irrad_poa, temp_cell, operating_cls

    resistance_shunt
    resistance_series
    diode_ideality_factor

    reference_photocurrent
    reference_saturation_current

    Returns
    -------

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
        alpha_isc,  # note alpha_isc is fixed.
        photocurrent_ref,
        saturation_current_ref,
        conductance_shunt_extra=conductance_shunt_extra,
        method=method,
        Eg_ref=band_gap_ref,
        dEgdT=dEgdT)

    # First, set all points to
    voltage_fit = out['v_mp']
    current_fit = out['i_mp']

    #         Array of classifications of each time stamp.
    #         0: System at maximum power point.
    #         1: System at open circuit conditions.
    #         2: Low irradiance nighttime.

    # If cls is 3, then system is clipped, need to find closest iv curve point.

    if np.any(operating_cls == 3):
        cax = operating_cls == 3
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
                                               Eg_ref=band_gap_ref,
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
            Eg_ref=band_gap_ref,
            dEgdT=dEgdT,
        )

        voltage_closest[voltage_closest < 0] = 0

        # TODO: Compare perpindicular distance to midpoint distance.

        #
        # delta_V = voltage_operation[cax] - voltage_closest
        # delta_I = current_operation[cax] - current_closest
        #
        # voltage_fit[cax] = voltage_closest + delta_V/(1 + (delta_I/delta_V)**2)
        # current_fit[cax] = current_closest + delta_I/(1 + (delta_I/delta_V)**2)

        voltage_fit[cax] = 0.5 * (voltage_operation[cax] + voltage_closest)
        current_fit[cax] = 0.5 * (current_operation[cax] + current_closest)




        # print('Clipped points fit:')
        # print(pd.DataFrame({'Current Fit': current_fit[cax],
        #               'Current Op': current_operation[cax],
        #               'Voltage Fit': voltage_fit[cax],
        #               'Voltage Op': voltage_operation[cax]}))

    # for k in range(len(effective_irradiance)):
    #     if operating_cls[k]==3:
    #
    #         out_iv = singlediode_closest_point(
    #                           voltage=voltage_operation[k],
    #                           current=current_operation[k],
    #                           effective_irradiance=effective_irradiance[k],
    #                           temperature_cell=temperature_cell[k],
    #                           resistance_shunt_ref=resistance_shunt_ref,
    #                           resistance_series_ref=resistance_series_ref,
    #                           diode_factor=diode_factor,
    #                           cells_in_series=cells_in_series,
    #                           alpha_isc=alpha_isc,
    #                           photocurrent_ref=photocurrent_ref,
    #                           saturation_current_ref=saturation_current_ref,
    #                           Eg_ref=band_gap_ref,
    #                           dEgdT=dEgdT,
    #                           ivcurve_pnts=10000,
    #                           method=method
    #                           )
    #         voltage_fit[k] = out_iv['v_closest']
    #         current_fit[k] = out_iv['i_closest']
    #

    # If cls is 1, then system is at open-circuit voltage.
    voltage_fit[operating_cls == 1] = out['v_oc'][operating_cls == 1]
    current_fit[operating_cls == 1] = 0

    return voltage_fit, current_fit


def pv_system_single_diode_model_old(
        effective_irradiance,
        temperature_cell,
        operating_cls,
        diode_factor,
        photocurrent_ref,
        saturation_current_ref,
        resistance_series_ref,
        resistance_shunt_ref,
        cells_in_series,
        alpha_isc,
        band_gap_ref=1.121
):
    """

    Fit function, reorganize Vmp, Imp outputs of size (1*N,) into a size (
    N*2,) vector. This is important because this is how
    scipy.optimize.curve_fit works.

    Parameters
    ----------
    X :

        irrad_poa, temp_cell, operating_cls

    resistance_shunt
    resistance_series
    diode_ideality_factor

    reference_photocurrent
    reference_saturation_current

    Returns
    -------

    """

    out = pvlib_single_diode(
        effective_irradiance,
        temperature_cell,
        resistance_shunt_ref,
        resistance_series_ref,
        diode_factor,
        cells_in_series,
        alpha_isc,  # note alpha_isc is fixed.
        photocurrent_ref,
        saturation_current_ref,
        reference_Eg=band_gap_ref)

    # First, set all points to
    voltage_fit = out['v_mp']
    current_fit = out['i_mp']

    # If cls is 1, then system is at open-circuit voltage.
    voltage_fit[operating_cls == 1] = out['v_oc'][operating_cls == 1]
    current_fit[operating_cls == 1] = 0

    return np.ravel((voltage_fit, current_fit))


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


def estimate_imp_ref(irradiance_poa,
                     temperature_cell,
                     imp,
                     figure=False,
                     figure_number=11,
                     ):
    """
    Estimate imp_ref using operation data. Note that typically imp for an
    array would be divided by parallel_strings before calling this function.

    Parameters
    ----------
    irradiance_poa
    temperature_cell
    imp
    makefigure

    Returns
    -------

    """

    cax = np.logical_and.reduce((
        irradiance_poa > np.nanpercentile(irradiance_poa, 90),
        irradiance_poa < 1100,
    ))

    x = temperature_cell[cax]
    y = imp[cax] / irradiance_poa[cax] * 1000

    # # cax = np.logical_and(y > np.nanpercentile(y,80), y < np.nanmean(y) * 1.5)
    # cax = y > np.nanpercentile(y,50)
    # x = x[cax]
    # y = y[cax]

    imp_fit = np.polyfit(x, y, 1)
    imp_ref_estimate = np.polyval(imp_fit, 25)

    if figure:
        plt.figure(figure_number)
        plt.clf()

        x_smooth = np.linspace(x.min(), x.max(), 5)
        plt.hist2d(x, y, bins=(100, 100))
        plt.plot(x_smooth, np.polyval(imp_fit, x_smooth), 'r')
        plt.xlabel('Cell temperature (C)')
        plt.ylabel('Imp (A)')
        plt.show()

    return imp_ref_estimate


def estimate_vmp_ref(irradiance_poa,
                     temperature_cell,
                     vmp,
                     figure=False,
                     figure_number=12,
                     ):
    """
    Estimate imp_ref using operation data. Note that typically imp for an
    array would be divided by parallel_strings before calling this function.

    Parameters
    ----------
    irradiance_poa
    temperature_cell
    imp
    makefigure

    Returns
    -------

    """

    cax = np.logical_and.reduce((
        irradiance_poa > np.nanpercentile(irradiance_poa, 90),
        irradiance_poa < 1100,
    ))

    x = temperature_cell[cax]
    y = vmp[cax]

    # # cax = np.logical_and(y > np.nanpercentile(y,80), y < np.nanmean(y) * 1.5)
    # cax = y > np.nanpercentile(y,50)
    # x = x[cax]
    # y = y[cax]

    imp_fit = np.polyfit(x, y, 1)
    imp_ref_estimate = np.polyval(imp_fit, 25)

    if figure:
        plt.figure(figure_number)
        plt.clf()

        x_smooth = np.linspace(x.min(), x.max(), 5)
        plt.hist2d(x, y, bins=(100, 100))
        plt.plot(x_smooth, np.polyval(imp_fit, x_smooth), 'r')
        plt.xlabel('Cell temperature (C)')
        plt.ylabel('Vmp (V)')
        plt.show()

    return imp_ref_estimate


def estimate_photocurrent_ref(imp_ref_estimate):
    return imp_ref_estimate / 0.934


def estimate_saturation_current_ref(imp_ref,
                                    photocurrent_ref,
                                    vmp_ref=None,
                                    resistance_series_ref=0.4,
                                    cells_in_series=None):
    kB = 1.381e-23
    q = 1.602e-19
    T = 25 + 273.15
    Vth = kB * T / q
    diode_factor = 1.1
    # voc_cell = 0.6

    # If cells in series is not provided, then use a standard value.
    if cells_in_series == None or vmp_ref == None:
        vmp_ref = 0.6
        cells_in_series = 1

    saturation_current_ref_estimate = (photocurrent_ref - imp_ref) / np.exp(
        (vmp_ref + imp_ref * resistance_series_ref) / (
                cells_in_series * diode_factor * Vth))

    return saturation_current_ref_estimate


def estimate_sdm_params(irradiance_poa,
                        temperature_cell,
                        vmp,
                        imp,
                        resistance_series_ref=0.4,
                        cells_in_series=None,
                        figure=False):
    """

    Estimate the Desoto single diode model parameters.

    Parameters
    ----------
    irradiance_poa
    temperature_cell
    vmp
    imp
    resistance_series_ref
    cells_in_series
    figure

    Returns
    -------

    """
    imp_ref = estimate_imp_ref(irradiance_poa, temperature_cell, imp,
                               figure=figure)
    vmp_ref = estimate_vmp_ref(irradiance_poa, temperature_cell, vmp,
                               figure=figure)
    photocurrent_ref = estimate_photocurrent_ref(imp_ref)

    saturation_current_ref = estimate_saturation_current_ref(
        imp_ref=imp_ref, photocurrent_ref=photocurrent_ref, vmp_ref=vmp_ref,
        resistance_series_ref=resistance_series_ref,
        cells_in_series=cells_in_series)

    desoto_params = dict(
        diode_factor=1.15,
        photocurrent_ref=photocurrent_ref,
        saturation_current_ref=saturation_current_ref,
        resistance_series_ref=resistance_series_ref,
        conductance_shunt_extra=0.001
    )

    return desoto_params


def estimate_desoto(irradiance_poa,
                    temperature_cell,
                    vmp,
                    imp,
                    resistance_series_ref=0.4,
                    cells_in_series=None,
                    figure=False):
    """

    Estimate the Desoto single diode model parameters.

    Parameters
    ----------
    irradiance_poa
    temperature_cell
    vmp
    imp
    resistance_series_ref
    cells_in_series
    figure

    Returns
    -------

    """
    imp_ref = estimate_imp_ref(irradiance_poa, temperature_cell, imp,
                               figure=figure)
    vmp_ref = estimate_vmp_ref(irradiance_poa, temperature_cell, vmp,
                               figure=figure)
    photocurrent_ref = estimate_photocurrent_ref(imp_ref)

    saturation_current_ref = estimate_saturation_current_ref(
        imp_ref=imp_ref, photocurrent_ref=photocurrent_ref, vmp_ref=vmp_ref,
        resistance_series_ref=resistance_series_ref,
        cells_in_series=cells_in_series)

    desoto_params = dict(
        diode_factor=1.15,
        photocurrent_ref=photocurrent_ref,
        saturation_current_ref=saturation_current_ref,
        resistance_series_ref=resistance_series_ref,
        resistance_shunt_ref=100
    )

    return desoto_params


class PvProHandler:
    """
    Class for running pvpro analysis.

    """

    def __init__(self,
                 df,
                 system_name='Unknown',
                 voltage_key=None,
                 current_key=None,
                 temperature_module_key=None,
                 irradiance_poa_key=None,
                 modules_per_string=None,
                 parallel_strings=None,
                 delta_T=3,
                 days_per_run=365,
                 time_step_between_iterations_days=36.5,
                 use_clear_times=True,
                 irradiance_lower_lim=100,
                 temperature_cell_upper_lim=500,
                 cells_in_series=None,
                 alpha_isc=None,
                 resistance_shunt_ref=400,
                 start_point_method='last',
                 solver='L-BFGS-B',
                 lower_bounds=None,
                 upper_bounds=None,
                 p0=None,
                 singlediode_method='newton',
                 ):

        # Initialize datahandler object.

        self.dh = DataHandler(df)

        # self.df = df
        self.system_name = system_name
        self.delta_T = delta_T
        self.days_per_run = days_per_run
        self.time_step_between_iterations_days = time_step_between_iterations_days
        self.use_clear_times = use_clear_times
        self.irradiance_lower_lim = irradiance_lower_lim
        self.temperature_cell_upper_lim = temperature_cell_upper_lim
        self.cells_in_series = cells_in_series
        self.alpha_isc = alpha_isc
        self.resistance_shunt_ref = resistance_shunt_ref
        self.voltage_key = voltage_key
        self.current_key = current_key
        self.temperature_module_key = temperature_module_key
        self.irradiance_poa_key = irradiance_poa_key
        self.modules_per_string = modules_per_string
        self.parallel_strings = parallel_strings
        self.singlediode_method = singlediode_method

        self.lower_bounds = lower_bounds or dict(
            diode_factor=0.5,
            photocurrent_ref=0,
            saturation_current_ref=1e-13,
            resistance_series_ref=0,
            conductance_shunt_extra=0
        )

        self.upper_bounds = upper_bounds or dict(
            diode_factor=2,
            photocurrent_ref=20,
            saturation_current_ref=1e-5,
            resistance_series_ref=1,
            conductance_shunt_extra=10
        )

        self.p0 = p0 or dict(
            diode_factor=1.03,
            photocurrent_ref=4,
            saturation_current_ref=1e-11,
            resistance_series_ref=0.4,
            conductance_shunt_extra=0
        )

        self.solver = solver
        self.start_point_method = start_point_method

        self.dataset_length_days = (df.index[-1] - df.index[0]).days

    @property
    def df(self):
        """
        Store dataframe inside the DataHandler.
        Returns
        -------

        """
        return self.dh.data_frame

    @df.setter
    def df(self, value):
        """
        Set Dataframe by setting the version inside datahandler.

        Parameters
        ----------
        value

        Returns
        -------

        """
        self.dh.data_frame = value

    # @iteration_start_days.setter
    # def iteration_start_days_setter(self, value):
    #     print('Cannot set iteration start days directly.')

    def simulation_setup(self):

        # Remove nan from df.
        keys = [self.voltage_key,
                self.current_key,
                self.temperature_module_key,
                self.irradiance_poa_key]

        for k in keys:
            if not k in self.df.keys():
                raise Exception(
                    'Key "{}" not in dataframe. Check specification of '
                    'voltage_key, current_key, temperature_module_key and '
                    'irradiance_poa_key'.format(
                        k))

        self.df.dropna(axis=0, subset=keys, inplace=True)

        # Make power column.
        self.df['power_dc'] = self.df[self.voltage_key] * self.df[
            self.current_key] / self.modules_per_string / self.parallel_strings

        # Calculate iteration start days
        self.iteration_start_days = np.round(
            np.arange(0, self.dataset_length_days - self.days_per_run,
                      self.time_step_between_iterations_days))

        self.calculate_cell_temperature()

    def calculate_cell_temperature(self):


        # Calculate cell temperature
        self.df.loc[:, 'temperature_cell'] = sapm_cell_from_module(
            self.df[self.temperature_module_key],
            self.df[self.irradiance_poa_key],
            delta_T=self.delta_T)

    def run_preprocess(self, correct_tz=True):

        if self.df[self.temperature_module_key].max() > 85:
            warnings.warn(
                'Maximum module temperature is larger than 85 C. Double check that input temperature is in Celsius, not Farenheight.')

        self.simulation_setup()

        # Run solar-data-tools.
        self.dh.run_pipeline(power_col='power_dc',
                             correct_tz=correct_tz,
                             extra_cols=[self.temperature_module_key,
                                         self.irradiance_poa_key,
                                         self.voltage_key,
                                         self.current_key]
                             )

        # Print report.
        self.dh.report()

        # Calculate operating class.
        self.df.loc[:, 'operating_cls'] = classify_operating_mode(
            voltage=self.df[self.voltage_key]/self.modules_per_string,
            current=self.df[self.current_key]/self.parallel_strings,
            power_clip=self.dh.capacity_estimate * 0.99)

        # TODO: this always overwrites p0 and should be changed so that if the user has set p0, it is not changed.
        self.estimate_p0()

    def info(self):
        """
        Print info about the class.

        Returns
        -------

        """
        keys = ['system_name', 'delta_T', 'days_per_run',
                'time_step_between_iterations_days', 'use_clear_times',
                'irradiance_lower_lim', 'temperature_cell_upper_lim',
                'cells_in_series',
                'alpha_isc', 'voltage_key', 'current_key',
                'temperature_module_key',
                'irradiance_poa_key', 'modules_per_string', 'parallel_strings',
                'solver', 'start_point_method',
                'dataset_length_days']

        info_display = {}
        for k in keys:
            info_display[k] = self.__dict__[k]

        print(pd.Series(info_display))
        return info_display

    def get_df_for_iteration(self, k,
                             use_clear_times=False):

        if use_clear_times:

            if not 'clear_time' in self.df.keys():
                raise Exception(
                    'Need to find clear times using find_clear_times() first.')

            idx = np.logical_and.reduce((
                self.df.index >= self.df.index[0] + datetime.timedelta(
                    int(self.iteration_start_days[k])),
                self.df.index < self.df.index[0] + datetime.timedelta(
                    int(self.iteration_start_days[k] + self.days_per_run)),
                self.df['clear_time']
            ))
        else:

            idx = np.logical_and.reduce((
                self.df.index >= self.df.index[0] + datetime.timedelta(
                    int(self.iteration_start_days[k])),
                self.df.index < self.df.index[0] + datetime.timedelta(
                    int(self.iteration_start_days[k] + self.days_per_run))
            ))

        # print('Current index for df', idx)
        return self.df[idx]

    def find_clear_times(self,
                         min_length=2,
                         smoothness_hyperparam=5000):

        self.dh.find_clear_times(min_length=min_length,
                                 smoothness_hyperparam=smoothness_hyperparam)
        self.dh.augment_data_frame(self.dh.boolean_masks.clear_times,
                                   'clear_time')

    def execute(self, iteration='all',
                use_mpp_points=True,
                use_voc_points=True,
                use_clip_points=True,
                verbose=True,
                method='minimize',
                save_figs=True,
                save_figs_directory='figures',
                figure_imp_max=None):

        start_time = time.time()
        q = 1.602e-19
        kB = 1.381e-23

        # Fit params taken from p0
        self.fit_params = self.p0.keys()

        # Calculate iteration time axis
        self.time = []
        for d in self.iteration_start_days:
            self.time.append(self.df.index[0] +
                             datetime.timedelta(
                                 int(d + 0.5 * self.days_per_run)))

        # Initialize pfit dataframe.
        pfit = pd.DataFrame(index=range(len(self.iteration_start_days)),
                            columns=[*self.fit_params,
                                     *['residual', 'i_sc_ref', 'v_oc_ref',
                                       'i_mp_ref',
                                       'v_mp_ref', 'p_mp_ref', 'i_x_ref',
                                       'i_xx_ref']])

        p0 = pd.DataFrame(index=range(len(self.iteration_start_days)),
                          columns=self.fit_params)

        # for d in range(len(self.iteration_start_days)):
        fit_result = []

        if iteration == 'all':
            print('Executing fit on all start days')
            iteration = np.arange(len(self.iteration_start_days))

        n = 0
        for k in iteration:

            print('\n--\nPercent complete: {:1.1%}, Iteration: {}'.format(
                k / len(self.iteration_start_days), k))
            df = self.get_df_for_iteration(k,
                                           use_clear_times=self.use_clear_times)

            number_points_in_time_step_all = len(df)

            # Filter
            df = df[np.logical_and.reduce((
                df[self.irradiance_poa_key] > self.irradiance_lower_lim,
                df['temperature_cell'] < self.temperature_cell_upper_lim
            ))].copy()

            if len(df) > 10:

                # try:

                # Can update p0 each iteration in future.
                if self.start_point_method == 'fixed':
                    p0.loc[k] = self.p0
                elif self.start_point_method == 'last':
                    if n == 0:
                        p0.loc[k] = self.p0
                    else:
                        p0.loc[k] = pfit_iter
                else:
                    raise ValueError(
                        'start_point_method must be "fixed" or "last"')

                # Do pvpro fit on this iteration.
                pfit_iter, residual, fit_result_iter = production_data_curve_fit(
                    temperature_cell=np.array(df['temperature_cell']),
                    effective_irradiance=np.array(df[self.irradiance_poa_key]),
                    operating_cls=np.array(df['operating_cls']),
                    voltage=df[self.voltage_key] / self.modules_per_string,
                    current=df[self.current_key] / self.parallel_strings,
                    cells_in_series=self.cells_in_series,
                    alpha_isc=self.alpha_isc,
                    resistance_shunt_ref=self.resistance_shunt_ref,
                    lower_bounds=self.lower_bounds,
                    upper_bounds=self.upper_bounds,
                    p0=p0.loc[k],
                    verbose=verbose,
                    solver=self.solver,
                    method=method,
                    singlediode_method=self.singlediode_method,
                    use_mpp_points=use_mpp_points,
                    use_voc_points=use_voc_points,
                    use_clip_points=use_clip_points,
                )
                for p in pfit_iter:
                    pfit.loc[k, p] = pfit_iter[p]

                pfit.loc[k, 'residual'] = residual

                fit_result.append(fit_result_iter)

                if verbose:
                    print('Final residual: {:.4f}'.format(residual))
                    print('Startpoint:')
                    print(p0.loc[k])
                    print('Fit result:')
                    print(pfit.loc[k])

                if save_figs:

                    self.plot_Vmp_Imp_scatter(p_plot=pfit_iter,
                                              figure_number=100,
                                              iteration=k,
                                              use_clear_times=self.use_clear_times,
                                              figure_imp_max=figure_imp_max)

                    if not os.path.exists(save_figs_directory):
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

                    vmp_imp_fig_name = os.path.join(save_figs_directory,
                                                    'Vmp_Imp',
                                                    '{}_Vmp-Imp_{}.png'.format(
                                                        self.system_name, k))
                    print('Exporting: {}'.format(vmp_imp_fig_name))
                    plt.savefig(vmp_imp_fig_name,
                                resolution=350,
                                bbox_inches='tight')

                    self.plot_suns_voc_scatter(p_plot=pfit_iter,
                                               figure_number=101,
                                               iteration=k,
                                               use_clear_times=self.use_clear_times)
                    plt.savefig(os.path.join(save_figs_directory,
                                             'suns_Voc',
                                             '{}_suns-Voc_{}.png'.format(
                                                 self.system_name, k)),
                                resolution=350,
                                bbox_inches='tight')

                    self.plot_current_irradiance_clipped_scatter(
                        p_plot=pfit_iter,
                        figure_number=103,
                        iteration=k,
                        use_clear_times=self.use_clear_times)
                    plt.savefig(os.path.join(save_figs_directory,
                                             'clipped',
                                             '{}_clipped_{}.png'.format(
                                                 self.system_name, k)),
                                resolution=350,
                                bbox_inches='tight')

                    self.plot_current_irradiance_mpp_scatter(p_plot=pfit_iter,
                        figure_number=104,
                        iteration=k,
                        use_clear_times=self.use_clear_times)
                    plt.savefig(os.path.join(save_figs_directory,
                                             'poa_Imp',
                                             '{}_poa-Imp_{}.png'.format(
                                                 self.system_name, k)),
                                resolution=350,
                                bbox_inches='tight')

                n = n + 1
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
                resistance_shunt=1 / (1 / self.resistance_shunt_ref + pfit.loc[
                    k, 'conductance_shunt_extra']),
                nNsVth=pfit.loc[k, 'nNsVth_ref'])

            for p in out.keys():
                pfit.loc[k, p + '_ref'] = out[p]

            # pfit.index = self.time

        self.result = dict(
            p=pfit,
            p0=p0,
            fit_result=fit_result,
            execution_time_seconds=time.time()-start_time
        )

        # print('Elapsed time to execute fit: {}'.format((time.time()-start_time)/60))

    def estimate_p0(self):
        """
        Make a rough estimate of the startpoint for fitting the single diode
        model.

        Returns
        -------

        """

        self.simulation_setup()

        cax = self.df['operating_cls'] == 0

        self.p0 = estimate_sdm_params(
            irradiance_poa=self.df.loc[cax, self.irradiance_poa_key],
            temperature_cell=self.df.loc[cax, 'temperature_cell'],
            vmp=self.df.loc[cax, self.voltage_key] / self.modules_per_string,
            imp=self.df.loc[cax, self.current_key] / self.parallel_strings,
            cells_in_series=self.cells_in_series
        )

        #
        # imp_ref = estimate_imp_ref()
        # photocurrent_ref = self.estimate_photocurrent_ref(imp_ref)
        # saturation_current_ref = self.estimate_saturation_current_ref(
        #     photocurrent_ref)
        #
        # self.p0 = dict(
        #     diode_factor=1.10,
        #     photocurrent_ref=photocurrent_ref,
        #     saturation_current_ref=saturation_current_ref,
        #     resistance_series_ref=0.4,
        #     resistance_shunt_ref=1e3
        # )

    def plot_Vmp_Imp_scatter(self,
                             p_plot,
                             figure_number=0,
                             iteration=1,
                             vmin=0,
                             vmax=70,
                             use_clear_times=None,
                             verbose=False,
                             figure_imp_max=None):
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
        if verbose:
            print(p_plot)

        if use_clear_times == None:
            use_clear_times = self.use_clear_times

        # Make figure for inverter on.
        fig = plt.figure(figure_number, figsize=(6.5, 3.5))
        plt.clf()
        ax = plt.axes()

        temp_limits = np.linspace(vmin, vmax, 8)

        df = self.get_df_for_iteration(iteration,
                                       use_clear_times=use_clear_times)

        inv_on_points = np.array(df['operating_cls'] == 0)

        vmp = np.array(
            df.loc[inv_on_points, self.voltage_key]) / self.modules_per_string
        imp = np.array(
            df.loc[inv_on_points, self.current_key]) / self.parallel_strings

        if figure_imp_max==None:
            imp_max = 1.1 * estimate_imp_ref(
                irradiance_poa=self.df.loc[
                    self.df['operating_cls'] == 0, self.irradiance_poa_key],
                temperature_cell=self.df.loc[
                    self.df['operating_cls'] == 0, self.temperature_module_key],
                imp=self.df.loc[self.df[
                                    'operating_cls'] == 0, self.current_key] / self.parallel_strings,
            )
        else:
            imp_max = figure_imp_max

        vmp_max = 1.1 * np.nanmax(
            self.df.loc[self.df['operating_cls'] == 0, self.voltage_key] /
            self.modules_per_string)

        h_sc = plt.scatter(vmp, imp,
                           c=df.loc[inv_on_points, 'temperature_cell'],
                           s=0.2,
                           cmap='jet',
                           vmin=0,
                           vmax=70)

        one_sun_points = np.logical_and.reduce((df['operating_cls'] == 0,
                                                df[
                                                    self.irradiance_poa_key] > 995,
                                                df[
                                                    self.irradiance_poa_key] < 1005,
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
        temperature_smooth = np.linspace(0, 70, 20)

        for effective_irradiance in [100, 1000]:
            voltage_plot, current_plot = pv_system_single_diode_model(
                effective_irradiance=np.array([effective_irradiance]),
                temperature_cell=temperature_smooth,
                operating_cls=np.zeros_like(temperature_smooth),
                cells_in_series=self.cells_in_series,
                alpha_isc=self.alpha_isc,
                resistance_shunt_ref=self.resistance_shunt_ref,
                **p_plot,
            )
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

            voltage_plot, current_plot = pv_system_single_diode_model(
                effective_irradiance=irrad_smooth,
                temperature_cell=temp_curr + np.zeros_like(irrad_smooth),
                operating_cls=np.zeros_like(irrad_smooth),
                cells_in_series=self.cells_in_series,
                alpha_isc=self.alpha_isc,
                resistance_shunt_ref=self.resistance_shunt_ref,
                **p_plot,
            )

            # out = pvlib_fit_fun( np.transpose(np.array(
            #     [irrad_smooth,temp_curr + np.zeros_like(irrad_smooth), np.zeros_like(irrad_smooth) ])),
            #                     *p_plot)

            # Reshape to get V, I
            # out = np.reshape(out,(2,int(len(out)/2)))

            # find the right color to plot.
            # norm_temp = (temp_curr-df[temperature].min())/(df[temperature].max()-df[temperature].min())
            norm_temp = (temp_curr - vmin) / (vmax - vmin)
            line_color = np.array(h_sc.cmap(norm_temp))
            # line_color[0:3] =line_color[0:3]*0.9

            line_color[3] = 0.3

            plt.plot(voltage_plot, current_plot,
                     label='Fit {:2.0f} C'.format(temp_curr),
                     color=line_color,
                     # color='C' + str(j)
                     )

        text_str = 'System: {}\n'.format(self.system_name) + \
                   'Analysis days: {:.0f}-{:.0f}\n'.format(
                       self.iteration_start_days[iteration],
                       self.iteration_start_days[
                           iteration] + self.days_per_run) + \
                   'Current: {}\n'.format(self.current_key) + \
                   'Voltage: {}\n'.format(self.voltage_key) + \
                   'Temperature: {}\n'.format(self.temperature_module_key) + \
                   'Irradiance: {}\n'.format(self.irradiance_poa_key) + \
                   'Temperature module->cell delta_T: {}\n'.format(
                       self.delta_T) + \
                   'n_diode: {:1.2f} \n'.format(p_plot['diode_factor']) + \
                   'reference_photocurrent: {:1.2f} A\n'.format(
                       p_plot['photocurrent_ref']) + \
                   'saturation_current_ref: {:1.2f} nA\n'.format(
                       p_plot['saturation_current_ref'] * 1e9) + \
                   'resistance_series: {:1.2f} Ohm\n'.format(
                       p_plot['resistance_series_ref']) + \
                   'Conductance shunt extra: {:1.2f} 1/Ohm\n\n'.format(
                       p_plot['conductance_shunt_extra']) + \
                   'Clear time: {}\n'.format(use_clear_times) + \
                   'Lower Irrad limit: {}\n'.format(self.irradiance_lower_lim)

        plt.text(0.05, 0.95, text_str,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax.transAxes,
                 fontsize=8)

        plt.xlim([0, vmp_max])
        plt.ylim([0, imp_max])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        pcbar = plt.colorbar(h_sc)
        pcbar.set_label('Cell Temperature (C)')

        plt.xlabel('Vmp (V)', fontsize=9)
        plt.ylabel('Imp (A)', fontsize=9)

        plt.show()

        return fig

    def plot_suns_voc_scatter(self,
                              p_plot,
                              figure_number=1,
                              iteration=1,
                              vmin=0,
                              vmax=70,
                              use_clear_times=None):
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
        if use_clear_times == None:
            use_clear_times = self.use_clear_times

        # Make figure for inverter on.
        fig = plt.figure(figure_number, figsize=(6.5, 3.5))
        plt.clf()
        ax = plt.axes()

        temp_limits = np.linspace(vmin, vmax, 8)

        df = self.get_df_for_iteration(iteration,
                                       use_clear_times=use_clear_times)

        inv_off_points = np.array(df['operating_cls'] == 1)

        voc = np.array(
            df.loc[inv_off_points, self.voltage_key]) / self.modules_per_string
        irrad = np.array(df.loc[inv_off_points, self.irradiance_poa_key])

        voc_max = np.nanmax(self.df.loc[self.df[
                                            'operating_cls'] == 1, self.voltage_key] / self.modules_per_string) * 1.1

        h_sc = plt.scatter(voc, irrad,
                           c=df.loc[inv_off_points, 'temperature_cell'],
                           s=0.2,
                           cmap='jet',
                           vmin=0,
                           vmax=70)

        # one_sun_points = np.logical_and.reduce((df['operating_cls'] == 1,
        #                                         df[
        #                                             self.irradiance_poa_key] > 995,
        #                                         df[
        #                                             self.irradiance_poa_key] < 1005,
        #                                         ))
        # if len(one_sun_points) > 0:
        #     # print('number one sun points: ', len(one_sun_points))
        #     plt.scatter(df.loc[
        #                     one_sun_points, self.voltage_key] / self.modules_per_string,
        #                 df.loc[
        #                     one_sun_points, self.current_key] / self.parallel_strings,
        #                 c=df.loc[one_sun_points, 'temperature_cell'],
        #                 edgecolors='k',
        #                 s=0.2)

        # Plot temperature scan
        temperature_smooth = np.linspace(0, 70, 20)

        # for effective_irradiance in [100, 1000]:
        #     voltage_plot, current_plot = pv_system_single_diode_model(
        #         effective_irradiance=effective_irradiance,
        #         temperature_cell=temperature_smooth,
        #         operating_cls=np.zeros_like(temperature_smooth)+1,
        #         **p_plot,
        #         cells_in_series=self.cells_in_series,
        #         alpha_isc=self.alpha_isc,
        #     )
        #     plt.plot(voltage_plot, current_plot, 'k:')
        #     plt.text(voltage_plot[0] + 0.5, current_plot[0],
        #              '{:.1g} sun'.format(effective_irradiance / 1000),
        #              horizontalalignment='left',
        #              verticalalignment='center',
        #              fontsize=8)

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
                **p_plot,
            )

            # out = pvlib_fit_fun( np.transpose(np.array(
            #     [irrad_smooth,temp_curr + np.zeros_like(irrad_smooth), np.zeros_like(irrad_smooth) ])),
            #                     *p_plot)

            # Reshape to get V, I
            # out = np.reshape(out,(2,int(len(out)/2)))

            # find the right color to plot.
            # norm_temp = (temp_curr-df[temperature].min())/(df[temperature].max()-df[temperature].min())
            norm_temp = (temp_curr - vmin) / (vmax - vmin)
            line_color = np.array(h_sc.cmap(norm_temp))
            # line_color[0:3] =line_color[0:3]*0.9

            line_color[3] = 0.3

            plt.plot(voltage_plot, irrad_smooth,
                     label='Fit {:2.0f} C'.format(temp_curr),
                     color=line_color,
                     # color='C' + str(j)
                     )

        text_str = 'System: {}\n'.format(self.system_name) + \
                   'Analysis days: {:.0f}-{:.0f}\n'.format(
                       self.iteration_start_days[iteration],
                       self.iteration_start_days[
                           iteration] + self.days_per_run) + \
                   'Current: {}\n'.format(self.current_key) + \
                   'Voltage: {}\n'.format(self.voltage_key) + \
                   'Temperature: {}\n'.format(self.temperature_module_key) + \
                   'Irradiance: {}\n'.format(self.irradiance_poa_key) + \
                   'Temperature module->cell delta_T: {}\n'.format(
                       self.delta_T) + \
                   'n_diode: {:1.2f} \n'.format(p_plot['diode_factor']) + \
                   'reference_photocurrent: {:1.2f} A\n'.format(
                       p_plot['photocurrent_ref']) + \
                   'saturation_current_ref: {:1.2f} nA\n'.format(
                       p_plot['saturation_current_ref'] * 1e9) + \
                   'resistance_series: {:1.2f} Ohm\n'.format(
                       p_plot['resistance_series_ref']) + \
                   'conductance shunt extra: {:1.2f} Ohm\n\n'.format(
                       p_plot['conductance_shunt_extra']) + \
                   'Clear time: {}\n'.format(use_clear_times) + \
                   'Lower Irrad limit: {}\n'.format(
                       self.irradiance_lower_lim)

        plt.text(0.05, 0.95, text_str,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax.transAxes,
                 fontsize=8)

        plt.xlim([0, voc_max])
        plt.yscale('log')
        plt.ylim([1e0, 1200])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        pcbar = plt.colorbar(h_sc)
        pcbar.set_label('Cell Temperature (C)')

        plt.xlabel('Voc (V)', fontsize=9)
        plt.ylabel('POA (W/m^2)', fontsize=9)

        plt.show()

        return fig

        # mpp_fig_fname = 'figures/{}_fleets16_simultfit-MPP_clear-times-{}_irraad-lower-lim-{}_alpha-isc-{}_days-per-run_{}_temperature-upper-lim-{}_deltaT-{}_{:02d}.png'.format(
        #         system, info['use_clear_times'], info['irradiance_lower_lim'], info['alpha_isc'], info['days_per_run'], info['temperature_cell_upper_lim'],info['delta_T'], d)
        # plt.savefig(mpp_fig_fname,
        #     dpi=200,
        #     bbox_inches='tight')

    def plot_current_irradiance_clipped_scatter(self,
                                                p_plot,
                                                figure_number=1,
                                                iteration=1,
                                                vmin=0,
                                                vmax=70,
                                                use_clear_times=None):
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
        if use_clear_times == None:
            use_clear_times = self.use_clear_times

        # Make figure for inverter on.
        fig = plt.figure(figure_number, figsize=(6.5, 3.5))
        plt.clf()
        ax = plt.axes()

        temp_limits = np.linspace(vmin, vmax, 8)

        df = self.get_df_for_iteration(iteration,
                                       use_clear_times=use_clear_times)

        cax = np.array(df['operating_cls'] == 3)

        current = np.array(
            df.loc[cax, self.current_key]) / self.parallel_strings

        irrad = np.array(df.loc[cax, self.irradiance_poa_key])

        current_max = np.nanmax(self.df.loc[self.df[
                                                'operating_cls'] == 3, self.current_key] / self.parallel_strings) * 1.1

        h_sc = plt.scatter(irrad, current,
                           c=df.loc[cax, 'temperature_cell'],
                           s=0.2,
                           cmap='jet',
                           vmin=0,
                           vmax=70)

        #
        # # Plot irradiance scan
        # for j in np.flip(np.arange(len(temp_limits))):
        #     temp_curr = temp_limits[j]
        #     irrad_smooth = np.linspace(1, 1200, 500)
        #
        #     voltage_plot, current_plot = pv_system_single_diode_model(
        #         effective_irradiance=irrad_smooth,
        #         temperature_cell=temp_curr + np.zeros_like(irrad_smooth),
        #         operating_cls=np.zeros_like(irrad_smooth) + 1,
        #         cells_in_series=self.cells_in_series,
        #         alpha_isc=self.alpha_isc,
        #         **p_plot,
        #     )
        #
        #     # out = pvlib_fit_fun( np.transpose(np.array(
        #     #     [irrad_smooth,temp_curr + np.zeros_like(irrad_smooth), np.zeros_like(irrad_smooth) ])),
        #     #                     *p_plot)
        #
        #     # Reshape to get V, I
        #     # out = np.reshape(out,(2,int(len(out)/2)))
        #
        #     # find the right color to plot.
        #     # norm_temp = (temp_curr-df[temperature].min())/(df[temperature].max()-df[temperature].min())
        #     norm_temp = (temp_curr - vmin) / (vmax - vmin)
        #     line_color = np.array(h_sc.cmap(norm_temp))
        #     # line_color[0:3] =line_color[0:3]*0.9
        #
        #     line_color[3] = 0.3
        #
        #     plt.plot(voltage_plot, irrad_smooth,
        #              label='Fit {:2.0f} C'.format(temp_curr),
        #              color=line_color,
        #              # color='C' + str(j)
        #              )

        text_str = 'System: {}\n'.format(self.system_name) + \
                   'Analysis days: {:.0f}-{:.0f}\n'.format(
                       self.iteration_start_days[iteration],
                       self.iteration_start_days[
                           iteration] + self.days_per_run) + \
                   'Current: {}\n'.format(self.current_key) + \
                   'Voltage: {}\n'.format(self.voltage_key) + \
                   'Temperature: {}\n'.format(self.temperature_module_key) + \
                   'Irradiance: {}\n'.format(self.irradiance_poa_key) + \
                   'Temperature module->cell delta_T: {}\n'.format(
                       self.delta_T) + \
                   'n_diode: {:1.2f} \n'.format(p_plot['diode_factor']) + \
                   'reference_photocurrent: {:1.2f} A\n'.format(
                       p_plot['photocurrent_ref']) + \
                   'saturation_current_ref: {:1.2f} nA\n'.format(
                       p_plot['saturation_current_ref'] * 1e9) + \
                   'resistance_series: {:1.2f} Ohm\n'.format(
                       p_plot['resistance_series_ref']) + \
                   'conductance shunt extra: {:1.2f} Ohm\n\n'.format(
                       p_plot['conductance_shunt_extra']) + \
                   'Clear time: {}\n'.format(use_clear_times) + \
                   'Lower Irrad limit: {}\n'.format(
                       self.irradiance_lower_lim)

        plt.text(0.05, 0.95, text_str,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax.transAxes,
                 fontsize=8)

        plt.ylim([0, current_max])
        plt.xlim([0, 1200])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        pcbar = plt.colorbar(h_sc)
        pcbar.set_label('Cell Temperature (C)')

        plt.ylabel('Current (A)', fontsize=9)
        plt.xlabel('POA (W/m^2)', fontsize=9)

        plt.show()

        return fig

        # mpp_fig_fname = 'figures/{}_fleets16_simultfit-MPP_clear-times-{}_irraad-lower-lim-{}_alpha-isc-{}_days-per-run_{}_temperature-upper-lim-{}_deltaT-{}_{:02d}.png'.format(
        #         system, info['use_clear_times'], info['irradiance_lower_lim'], info['alpha_isc'], info['days_per_run'], info['temperature_cell_upper_lim'],info['delta_T'], d)
        # plt.savefig(mpp_fig_fname,
        #     dpi=200,
        #     bbox_inches='tight')


    def plot_current_irradiance_mpp_scatter(self,
                                                p_plot,
                                                figure_number=1,
                                                iteration=1,
                                                vmin=0,
                                                vmax=70,
                                                use_clear_times=None):
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
        if use_clear_times == None:
            use_clear_times = self.use_clear_times

        # Make figure for inverter on.
        fig = plt.figure(figure_number, figsize=(6.5, 3.5))
        plt.clf()
        ax = plt.axes()

        temp_limits = np.linspace(vmin, vmax, 8)

        df = self.get_df_for_iteration(iteration,
                                       use_clear_times=use_clear_times)

        cax = np.array(df['operating_cls'] == 0)

        current = np.array(
            df.loc[cax, self.current_key]) / self.parallel_strings

        irrad = np.array(df.loc[cax, self.irradiance_poa_key])

        current_max = np.nanmax(self.df.loc[self.df['operating_cls'] == 0, self.current_key] / self.parallel_strings) * 1.1

        h_sc = plt.scatter(irrad, current,
                           c=df.loc[cax, 'temperature_cell'],
                           s=0.2,
                           cmap='jet',
                           vmin=0,
                           vmax=70)


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
                resistance_shunt_ref=self.resistance_shunt_ref,
                **p_plot,
            )

            # out = pvlib_fit_fun( np.transpose(np.array(
            #     [irrad_smooth,temp_curr + np.zeros_like(irrad_smooth), np.zeros_like(irrad_smooth) ])),
            #                     *p_plot)

            # Reshape to get V, I
            # out = np.reshape(out,(2,int(len(out)/2)))

            # find the right color to plot.
            # norm_temp = (temp_curr-df[temperature].min())/(df[temperature].max()-df[temperature].min())
            norm_temp = (temp_curr - vmin) / (vmax - vmin)
            line_color = np.array(h_sc.cmap(norm_temp))
            # line_color[0:3] =line_color[0:3]*0.9

            line_color[3] = 0.3

            plt.plot(irrad_smooth, current_plot,
                     label='Fit {:2.0f} C'.format(temp_curr),
                     color=line_color,
                     # color='C' + str(j)
                     )

        text_str = 'System: {}\n'.format(self.system_name) + \
                   'Analysis days: {:.0f}-{:.0f}\n'.format(
                       self.iteration_start_days[iteration],
                       self.iteration_start_days[
                           iteration] + self.days_per_run) + \
                   'Current: {}\n'.format(self.current_key) + \
                   'Voltage: {}\n'.format(self.voltage_key) + \
                   'Temperature: {}\n'.format(self.temperature_module_key) + \
                   'Irradiance: {}\n'.format(self.irradiance_poa_key) + \
                   'Temperature module->cell delta_T: {}\n'.format(
                       self.delta_T) + \
                   'n_diode: {:1.2f} \n'.format(p_plot['diode_factor']) + \
                   'reference_photocurrent: {:1.2f} A\n'.format(
                       p_plot['photocurrent_ref']) + \
                   'saturation_current_ref: {:1.2f} nA\n'.format(
                       p_plot['saturation_current_ref'] * 1e9) + \
                   'resistance_series: {:1.2f} Ohm\n'.format(
                       p_plot['resistance_series_ref']) + \
                   'conductance shunt extra: {:1.2f} Ohm\n\n'.format(
                       p_plot['conductance_shunt_extra']) + \
                   'Clear time: {}\n'.format(use_clear_times) + \
                   'Lower Irrad limit: {}\n'.format(
                       self.irradiance_lower_lim)

        plt.text(0.05, 0.95, text_str,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax.transAxes,
                 fontsize=8)

        plt.ylim([0, current_max])
        plt.xlim([0, 1200])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        pcbar = plt.colorbar(h_sc)
        pcbar.set_label('Cell Temperature (C)')

        plt.ylabel('Current (A)', fontsize=9)
        plt.xlabel('POA (W/m^2)', fontsize=9)

        plt.show()

        return fig

        # mpp_fig_fname = 'figures/{}_fleets16_simultfit-MPP_clear-times-{}_irraad-lower-lim-{}_alpha-isc-{}_days-per-run_{}_temperature-upper-lim-{}_deltaT-{}_{:02d}.png'.format(
        #         system, info['use_clear_times'], info['irradiance_lower_lim'], info['alpha_isc'], info['days_per_run'], info['temperature_cell_upper_lim'],info['delta_T'], d)
        # plt.savefig(mpp_fig_fname,
        #     dpi=200,
        #     bbox_inches='tight')