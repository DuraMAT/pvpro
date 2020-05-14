import pvlib
import numpy as np
import pandas as pd
# import pytz
from collections import OrderedDict
# from functools import partial
import scipy
import datetime
import os

from pvlib.singlediode import _lambertw_i_from_v, _lambertw_v_from_i, _golden_sect_DataFrame, _pwr_optfcn

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from solardatatools import DataHandler


def _lambertw(photocurrent, saturation_current, resistance_series,
              resistance_shunt, nNsVth, ivcurve_pnts=None,
              calculate_all=False):

    if calculate_all:
        # Compute short circuit current
        i_sc = _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth, 0.,
                                  saturation_current, photocurrent)

    # Compute open circuit voltage
    v_oc = _lambertw_v_from_i(resistance_shunt, resistance_series, nNsVth, 0.,
                              saturation_current, photocurrent)

    params = {'r_sh': resistance_shunt,
              'r_s': resistance_series,
              'nNsVth': nNsVth,
              'i_0': saturation_current,
              'i_l': photocurrent}

    # Find the voltage, v_mp, where the power is maximized.
    # Start the golden section search at v_oc * 1.14
    p_mp, v_mp = _golden_sect_DataFrame(params, 0., v_oc * 1.14,
                                        _pwr_optfcn)

    # Find Imp using Lambert W
    i_mp = _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth,
                              v_mp, saturation_current, photocurrent)

    if calculate_all:
        # Find Ix and Ixx using Lambert W
        i_x = _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth,
                                 0.5 * v_oc, saturation_current, photocurrent)

        i_xx = _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth,
                                  0.5 * (v_oc + v_mp), saturation_current,
                                  photocurrent)

    if calculate_all:
        out = (i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx)
    else:
        out = (v_oc, i_mp, v_mp, p_mp)

    # create ivcurve
    if ivcurve_pnts:
        ivcurve_v = (np.asarray(v_oc)[..., np.newaxis] *
                     np.linspace(0, 1, ivcurve_pnts))

        ivcurve_i = _lambertw_i_from_v(resistance_shunt, resistance_series,
                                       nNsVth, ivcurve_v.T, saturation_current,
                                       photocurrent).T

        out += (ivcurve_i, ivcurve_v)

    return out

# Sped up singlediode by 8% by not calculating I_
def singlediode(photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth, ivcurve_pnts=None,
                method='lambertw',
                calculate_all=False):
    """
    Solve the single-diode model to obtain a photovoltaic IV curve.

    Singlediode solves the single diode equation [1]

    .. math::

        I = IL - I0*[exp((V+I*Rs)/(nNsVth))-1] - (V + I*Rs)/Rsh

    for ``I`` and ``V`` when given ``IL, I0, Rs, Rsh,`` and ``nNsVth
    (nNsVth = n*Ns*Vth)`` which are described later. Returns a DataFrame
    which contains the 5 points on the I-V curve specified in
    SAND2004-3535 [3]. If all IL, I0, Rs, Rsh, and nNsVth are scalar, a
    single curve will be returned, if any are Series (of the same
    length), multiple IV curves will be calculated.

    The input parameters can be calculated using calcparams_desoto from
    meteorological data.

    Parameters
    ----------
    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.
        0 <= photocurrent

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.
        0 < saturation_current

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.
        0 <= resistance_series < numpy.inf

    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.
        0 < resistance_shunt <= numpy.inf

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).
        0 < nNsVth

    ivcurve_pnts : None or int, default None
        Number of points in the desired IV curve. If None or 0, no
        IV curves will be produced.

    method : str, default 'lambertw'
        Determines the method used to calculate points on the IV curve. The
        options are ``'lambertw'``, ``'newton'``, or ``'brentq'``.

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

    Notes
    -----
    If the method is ``'lambertw'`` then the solution employed to solve the
    implicit diode equation utilizes the Lambert W function to obtain an
    explicit function of :math:`V=f(I)` and :math:`I=f(V)` as shown in [2].

    If the method is ``'newton'`` then the root-finding Newton-Raphson method
    is used. It should be safe for well behaved IV-curves, but the ``'brentq'``
    method is recommended for reliability.

    If the method is ``'brentq'`` then Brent's bisection search method is used
    that guarantees convergence by bounding the voltage between zero and
    open-circuit.

    If the method is either ``'newton'`` or ``'brentq'`` and ``ivcurve_pnts``
    are indicated, then :func:`pvlib.singlediode.bishop88` [4] is used to
    calculate the points on the IV curve points at diode voltages from zero to
    open-circuit voltage with a log spacing that gets closer as voltage
    increases. If the method is ``'lambertw'`` then the calculated points on
    the IV curve are linearly spaced.

    References
    -----------
    [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN
    0 86758 909 4

    [2] A. Jain, A. Kapoor, "Exact analytical solutions of the
    parameters of real solar cells using Lambert W-function", Solar
    Energy Materials and Solar Cells, 81 (2004) 269-277.

    [3] D. King et al, "Sandia Photovoltaic Array Performance Model",
    SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    [4] "Computer simulation of the effects of electrical mismatches in
    photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
    https://doi.org/10.1016/0379-6787(88)90059-2

    See also
    --------
    sapm
    calcparams_desoto
    pvlib.singlediode.bishop88
    """
    # Calculate points on the IV curve using the LambertW solution to the
    # single diode equation
    if method.lower() == 'lambertw':
        out = _lambertw(
            photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth, ivcurve_pnts,
            calculate_all=calculate_all
        )
        if calculate_all:
            i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx = out[:7]
        else:
            v_oc, i_mp, v_mp, p_mp = out[:7]

        if ivcurve_pnts:
            ivcurve_i, ivcurve_v = out[7:]
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth)  # collect args
        v_oc = pvlib.singlediode.bishop88_v_from_i(
            0.0, *args, method=method.lower()
        )
        i_mp, v_mp, p_mp = pvlib.singlediode.bishop88_mpp(
            *args, method=method.lower()
        )
        if calculate_all:
            i_sc = pvlib.singlediode.bishop88_i_from_v(
                0.0, *args, method=method.lower())
            i_x = pvlib.singlediode.bishop88_i_from_v(
                v_oc / 2.0, *args, method=method.lower()
            )
            i_xx = pvlib.singlediode.bishop88_i_from_v(
                (v_oc + v_mp) / 2.0, *args, method=method.lower()
            )

        # calculate the IV curve if requested using bishop88
        if ivcurve_pnts:
            vd = v_oc * (
                    (11.0 - np.logspace(np.log10(11.0), 0.0,
                                        ivcurve_pnts)) / 10.0
            )
            ivcurve_i, ivcurve_v, _ = pvlib.singlediode.bishop88(vd, *args)

    out = OrderedDict()

    out['v_oc'] = v_oc
    out['i_mp'] = i_mp
    out['v_mp'] = v_mp
    out['p_mp'] = p_mp

    if calculate_all:
        out['i_sc'] = i_sc
        out['i_x'] = i_x
        out['i_xx'] = i_xx

    if ivcurve_pnts:

        out['v'] = ivcurve_v
        out['i'] = ivcurve_i

    if isinstance(photocurrent, pd.Series) and not ivcurve_pnts:
        out = pd.DataFrame(out, index=photocurrent.index)

    return out


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
        reference_irradiance=1000,
        reference_temperature=25,
        method='newton',
        verbose=False,
        calculate_all=False):
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

    iph, io, rs, rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
        effective_irradiance,
        temperature_cell,
        alpha_sc=alpha_isc,
        a_ref=diode_factor*cells_in_series*kB/q*(273.15 + temperature_cell),
        I_L_ref=photocurrent_ref,
        I_o_ref=saturation_current_ref,
        R_sh_ref=resistance_shunt_ref,
        R_s=resistance_series_ref,
        EgRef=Eg_ref,
        dEgdT=dEgdT,
        )

    # if verbose:
    #     print('Photocurrent')
    #     print(photocurrent)
    #     print('Eg')
    #     print(Eg)
    #     print('Saturation current')
    #     print(saturation_current)
    #     print('nNsVth')
    #     print(nNsVth)

    out = singlediode(iph,
                      io,
                      rs,
                      rsh,
                      nNsVth,
                      method=method,
                      calculate_all=calculate_all)

    # out = pvlib.singlediode(photocurrent, saturation_current,
    #                   resistance_series,
    #                   resistance_shunt, nNsVth,
    #                   method=method,
    #                   calculate_all=calculate_all)

    return out

#
# def classify_time_stamp(voltage, current, irrad, method='fraction'):
#     """
#
#     Parameters
#     ----------
#     voltage
#     current
#     irrad
#     method
#
#     Returns
#     -------
#     cls : array
#
#         Array of classifications of each time stamp:
#             0: module at open circuit conditions, daytime
#             1: module at maximum power point, daytime.
#             2: Nighttime.
#
#     """
#
#
#     cls = np.zeros(np.shape(voltage),dtype='int')
#
#     # Inverter off
#     cls[np.logical_and(current<current.max()*0.01, voltage>voltage.max()*0.01) ] = 0
#
#     # Inverter on
#     cls[np.logical_and(voltage>voltage.max()*0.01,current>current.max()*0.01, )] = 1
#
#     # Night
#     cls[irrad<1] = 2
#
#     return cls


def import_csv(filename):
    """Import an NSRDB csv file.

    The function (df,info) = import_csv(filename) imports an NSRDB formatted
    csv file

    Parameters
    ----------
    filename

    Returns
    -------
    df
        pandas dataframe of data
    info
        pandas dataframe of header data.
    """

    # filename = '1ad06643cad4eeb947f3de02e9a0d6d7/128364_38.29_-122.14_1998.csv'

    info_df = pd.read_csv(filename, nrows=1)
    info = {}
    for p in info_df:
        info[p] = info_df[p].iloc[0]

    # See metadata for specified properties, e.g., timezone and elevation
    # timezone, elevation = info['Local Time Zone'], info['Elevation']

    # Return all but first 2 lines of csv to get data:
    df = pd.read_csv(filename, skiprows=2)

    # # Set the time index in the pandas dataframe:
    # year=str(df['Year'][0])
    #
    #
    # if np.diff(df[0:2].Minute) == 30:
    #     interval = '30'
    #     info['interval_in_hours']= 0.5
    #     df = df.set_index(
    #       pd.date_range('1/1/{yr}'.format(yr=year), freq=interval + 'Min',
    #                     periods=60*24*365 / int(interval)))
    # elif df['Minute'][1] - df['Minute'][0]==0:
    #     interval = '60'
    #     info['interval_in_hours'] = 1
    #     df = df.set_index(
    #         pd.date_range('1/1/{yr}'.format(yr=year), freq=interval + 'Min',
    #                       periods=60*24*365 / int(interval)))
    # else:
    #     print('Interval not understood!')
    #
    # df.index = df.index.tz_localize(
    #     pytz.FixedOffset(float(info['Time Zone'] * 60)))

    return (df, info)

def spectral_variation(spectral_data, reference_spectrum, wavelength):
    dwavelength = np.diff(wavelength)
    dwavelength = np.append(dwavelength, dwavelength[-1])

    total_intensity = np.sum(np.array(spectral_data) * dwavelength, axis=1)

    total_intensity_0 = np.sum(reference_spectrum * dwavelength)

    delta_E = total_intensity_0 /\
              np.reshape(total_intensity, (len(total_intensity), 1)) *\
              np.array(spectral_data) - reference_spectrum

    delta_E[total_intensity == 0, :] = 0

    return delta_E

def spectral_mismatch(spectral_data, reference_spectrum, eqe, wavelength):
    dwavelength = np.diff(wavelength)
    dwavelength = np.append(dwavelength, dwavelength[-1])

    delta_E = spectral_variation(spectral_data, reference_spectrum, wavelength)

    # # Check delta_E integrates to zero.
    # delta_E_tot = np.sum(delta_E * dwavelength, axis=1)
    # print('Delta E max: ', np.nanmax(np.abs(delta_E_tot)))


    spectral_correction = 1 + np.sum(
        delta_E * eqe * wavelength * dwavelength, axis=1) / np.sum(
        reference_spectrum * eqe * wavelength * dwavelength)

    return spectral_correction



def find_runs(x):
    """Find runs of consecutive items in an array.
    https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065"""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths

def find_clear_times(measured_matrix, clear_matrix, th_relative_power=0.1, th_relative_smoothness=200,
                     min_length=3):
    n1, n2 = measured_matrix.shape
    # calculate clearness index based on clear sky power estimates
    ci = np.zeros_like(clear_matrix)
    daytime = np.logical_and(
        measured_matrix >= 0.05 * np.percentile(clear_matrix, 95),
        clear_matrix >= 0.05 * np.percentile(clear_matrix, 95)
    )
    ci[daytime] = np.clip(np.divide(measured_matrix[daytime], clear_matrix[daytime]), 0, 2)
    # compare relative 2nd order smoothness of measured data and clear sky estimate
    diff_meas = np.r_[0, (np.diff(measured_matrix.ravel(order='F'), n=2)), 0]
    diff_clear = np.r_[0, (np.diff(clear_matrix.ravel(order='F'), n=2)), 0]
    diff_compare = (np.abs(diff_meas - diff_clear)).reshape(n1, n2, order='F')
    # assign clear times as high clearness index and similar smoothness
    clear_times = np.logical_and(
        np.abs(ci - 1) <= th_relative_power,
        diff_compare <= th_relative_smoothness
    )
    # remove clear times that are in small groups, below the minimum length threshold
    run_values, run_starts, run_lengths = find_runs(clear_times.ravel(order='F'))
    for val, start, length in zip(run_values, run_starts, run_lengths):
        if val is False:
            continue
        if length >= min_length:
            continue
        i = start % n1
        j = start // n1
        for count in range(length):
            clear_times[i + count, j] = False
    return clear_times

def make_clear_series(data_handler_obj, bool_mat):
    """Assumes that the data handler object was instantiated from a data frame with a time index"""
    start = data_handler_obj.day_index[0]
    freq = '{}min'.format(data_handler_obj.data_sampling)
    periods = data_handler_obj.filled_data_matrix.size
    tindex = pd.date_range(start=start, freq=freq, periods=periods)
    series = pd.Series(data=bool_mat.ravel(order='F'), index=tindex)
    series.name = 'clear_times'
    return series

def sapm_module_to_cell_temperature(T_module,irradiance,delta_T=3):
    # Modify module temperature to get to cell temperature


    T_cell = T_module + delta_T * irradiance / 1000

    return T_cell



def classify_operating_mode(voltage, current, method='fraction'):
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

    """


    cls = np.zeros(np.shape(voltage)) - 1

    # Inverter on
    cls[np.logical_and(
        voltage > voltage.max() * 0.01,
        current > current.max() * 0.01,
    )] = 0

    # Nighttime, low voltage and irradiance.
    cls[np.logical_and(current<current.max()*0.01, voltage<voltage.max()*0.01) ] = 2


    # Inverter off
    cls[np.logical_and(current<current.max()*0.01, voltage>voltage.max()*0.01) ] = 1




    return cls




def pv_system_single_diode_model(
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
        band_gap_ref=1.121,
        dEgdT=-0.0002677,
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


    out = pvlib_single_diode(
        effective_irradiance,
        temperature_cell,
        resistance_shunt_ref,
        resistance_series_ref,
        diode_factor,
        cells_in_series,
        alpha_isc, # note alpha_isc is fixed.
        photocurrent_ref,
        saturation_current_ref,
        Eg_ref=band_gap_ref,
        dEgdT=dEgdT)

    # First, set all points to
    voltage_fit = out['v_mp']
    current_fit = out['i_mp']


    #         Array of classifications of each time stamp.
    #         0: System at maximum power point.
    #         1: System at open circuit conditions.
    #         2: Low irradiance nighttime.

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
        alpha_isc, # note alpha_isc is fixed.
        photocurrent_ref,
        saturation_current_ref,
        reference_Eg=band_gap_ref)

    # First, set all points to
    voltage_fit = out['v_mp']
    current_fit = out['i_mp']

    # If cls is 1, then system is at open-circuit voltage.
    voltage_fit[operating_cls == 1] = out['v_oc'][operating_cls == 1]
    current_fit[operating_cls == 1] = 0

    return np.ravel((voltage_fit,current_fit))




def production_data_curve_fit_4_param(
        temperature_cell,
        effective_irradiance,
        operating_cls,
        voltage,
        current,
        lower_bounds,
        upper_bounds,
        scale,
        alpha_isc=None,
        diode_factor=None,
        photocurrent_ref=None,
        saturation_current_ref=None,
        resistance_series_ref=None,
        resistance_shunt_ref=None,
        p0=dict(
            diode_factor=1.0,
            photocurrent_ref=8,
            saturation_current_ref=10,
            resistance_series_ref=10,
        ),
        cells_in_series=72,
        band_gap_ref=1.121,
        verbose=False):
    """
    This is a test comparison function. Not to be deployed probably.

    Parameters
    ----------
    temperature_cell
    effective_irradiance
    operating_cls
    voltage
    current
    lower_bounds
    upper_bounds
    scale
    alpha_isc
    diode_factor
    photocurrent_ref
    saturation_current_ref
    resistance_series_ref
    resistance_shunt_ref
    p0
    cells_in_series
    band_gap_ref
    verbose

    Returns
    -------

    """


    def pvlib_fit_fun(X,
                      diode_ideality_factor,
                      photocurrent_ref,
                      saturation_current_ref,
                      resistance_series_ref,
                      resistance_shunt_ref):
        return pv_system_single_diode_model_old(
            effective_irradiance=X[:, 0],
            temperature_cell=X[:, 1],
            operating_cls=X[:, 2],
            diode_factor=diode_ideality_factor,
            photocurrent_ref=photocurrent_ref,
            saturation_current_ref=saturation_current_ref * 1e-12,
            resistance_series_ref=resistance_series_ref,
            resistance_shunt_ref=resistance_shunt_ref,
            cells_in_series=cells_in_series,
            alpha_isc=alpha_isc
        )



    input_data = np.concatenate(effective_irradiance, temperature_cell, operating_cls)




    IV_data = np.ravel(np.array(voltage), np.array(current))

    # try:
    p_fit, pcov = scipy.optimize.curve_fit(pvlib_fit_fun, input_data, IV_data, p0,
                            bounds=bounds,
                            )


def production_data_curve_fit(
        temperature_cell,
        effective_irradiance,
        operating_cls,
        voltage,
        current,
        lower_bounds,
        upper_bounds,
        scale,
        alpha_isc=None,
        diode_factor=None,
        photocurrent_ref=None,
        saturation_current_ref=None,
        resistance_series_ref=None,
        resistance_shunt_ref=None,
        p0=dict(
            diode_factor=1.0,
            photocurrent_ref=8,
            saturation_current_ref=10,
            resistance_series_ref=10,
            resistance_shunt_ref=10
        ),
        cells_in_series=72,
        band_gap_ref=1.121,
        verbose=False,
        solver='Nelder-Mead',
        method='minimize',
        brute_number_grid_points=2,
    ):
    """
    Use curve fitting to find best-fit single-diode-model paramters given the
    operating data.

    Data to be fit is supplied in 1xN vectors as the inputs temperature_cell,
    effective_irradiance, operation_cls, voltage and current. Each of these
    inputs must have the same length.

    The inputs lower_bound, upper_bound, scale and p0 are all dictionaries
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

    scale : dict

        dictionary providing a prescaling factor for optimization. For
        example, saturation_current is typically a number around 1e-10,
        in order to avoid numeric issues, set a scale factor to around 1e-10.

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

    model_kwargs = dict(
        effective_irradiance=effective_irradiance,
        temperature_cell=temperature_cell,
        operating_cls=operating_cls,
        cells_in_series=cells_in_series,
        band_gap_ref=band_gap_ref
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
            # return x*1e-10
        elif key == 'resistance_series_ref':
            return x
        elif key == 'resistance_shunt_ref':
            return np.exp(2*(x-1))

    def p_to_x(p, key):
        if key == 'diode_factor':
            return p
        elif key == 'photocurrent_ref':
            return p
        elif key == 'saturation_current_ref':
            return np.log(p) + 21
            # return p*1e10
        elif key == 'resistance_series_ref':
            return p
        elif key == 'resistance_shunt_ref':
            return np.log(p)/2+1

    # note that this will be the order of parameters in the model function.
    fit_params = p0.keys()

    def model(x):
        p = model_kwargs.copy()
        n = 0
        for param in fit_params:

            p[param] = x_to_p(x[n],param)
            # p[param] = x[n] * scale[param]
            n = n + 1
        return pv_system_single_diode_model(**p)

    def residual(x):

        voltage_fit, current_fit = model(x)
        return np.nanmean(
            np.abs(voltage_fit - voltage) ** 2 + np.abs(current_fit - current) ** 2)

    # print(signature(model))

    print('Starting residual: ', residual([p_to_x(p0[k],k) for k in fit_params]))



    # bounds = scipy.optimize.Bounds(
    #     [lower_bounds[k] / scale[k] for k in fit_params],
    #     [upper_bounds[k] / scale[k] for k in fit_params])


    if method=='minimize':
        bounds = scipy.optimize.Bounds(
            [p_to_x(lower_bounds[k],k) for k in fit_params],
            [p_to_x(upper_bounds[k],k) for k in fit_params])


        x0 = [p_to_x(p0[k],k) for k in fit_params]

        print('Starting x0: ', x0)
        print('bounds:', bounds)
        res = scipy.optimize.minimize(residual,
                                  x0=x0,
                                  bounds=bounds,
                                  options=dict(
                                      # maxiter=100,
                                      disp=verbose
                                  ),
                                  method=solver
                                  )


        # print(res)
        n = 0
        p_fit = {}
        for param in fit_params:
            p_fit[param] = x_to_p(res.x[n],param)
            n = n + 1

        # print('Best fit parameters (with scale included):')
        # for p in x_fit:
        #     print('{}: {}'.format(p, x_fit[p]))
        print('Final Residual: {}'.format(res['fun']))
        return p_fit, res['fun'], res




def production_data_curve_fit_old(
        temperature_cell,
        effective_irradiance,
        operating_cls,
        voltage,
        current,
        lower_bounds,
        upper_bounds,
        scale,
        alpha_isc=None,
        diode_factor=None,
        photocurrent_ref=None,
        saturation_current_ref=None,
        resistance_series_ref=None,
        resistance_shunt_ref=None,
        p0=dict(
            diode_factor=1.0,
            photocurrent_ref=8,
            saturation_current_ref=10,
            resistance_series_ref=10,
            resistance_shunt_ref=10
        ),
        cells_in_series=72,
        band_gap_ref=1.121,
        verbose=False,
        solver='Nelder-Mead',
        method='minimize',
        brute_number_grid_points=2,
    ):
    """
    Use curve fitting to find best-fit single-diode-model paramters given the
    operating data.

    Data to be fit is supplied in 1xN vectors as the inputs temperature_cell,
    effective_irradiance, operation_cls, voltage and current. Each of these
    inputs must have the same length.

    The inputs lower_bound, upper_bound, scale and p0 are all dictionaries
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

    scale : dict

        dictionary providing a prescaling factor for optimization. For
        example, saturation_current is typically a number around 1e-10,
        in order to avoid numeric issues, set a scale factor to around 1e-10.

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

    model_kwargs = dict(
        effective_irradiance=effective_irradiance,
        temperature_cell=temperature_cell,
        operating_cls=operating_cls,
        cells_in_series=cells_in_series,
        band_gap_ref=band_gap_ref
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


    # note that this will be the order of parameters in the model function.
    fit_params = p0.keys()

    def model(x):
        p = model_kwargs.copy()
        n = 0
        # Want to add more arbitrary parameter changes here using mathematical fit params and scaled params
        for param in fit_params:
            p[param] = x[n] * scale[param]
            n = n + 1
        return pv_system_single_diode_model(**p)

    def residual(x):

        voltage_fit, current_fit = model(x)
        return np.nanmean(
            np.abs(voltage_fit - voltage) ** 2 + np.abs(current_fit - current) ** 2)

    # print(signature(model))

    # print('Starting residual: ', residual([p0[k] for k in fit_params]))

    # bounds = scipy.optimize.Bounds(
    #     [lower_bounds[k] / scale[k] for k in fit_params],
    #     [upper_bounds[k] / scale[k] for k in fit_params])


    if method=='minimize':
        bounds = scipy.optimize.Bounds(
            [lower_bounds[k] / scale[k] for k in fit_params],
            [upper_bounds[k] / scale[k] for k in fit_params])


        x0 = [p0[k]/scale[k] for k in fit_params]
        res = scipy.optimize.minimize(residual,
                                  x0=x0,
                                  bounds=bounds,
                                  options=dict(
                                      maxiter=100,
                                      disp=verbose
                                  ),
                                  method=solver
                                  )


        # print(res)
        n = 0
        p_fit = {}
        for param in fit_params:
            p_fit[param] = res.x[n] * scale[param]
            n = n + 1

        # print('Best fit parameters (with scale included):')
        # for p in x_fit:
        #     print('{}: {}'.format(p, x_fit[p]))
        print('Final Residual: {}'.format(res['fun']))
        return p_fit, res['fun'], res

    elif method=='brute':
        ranges = [ (lower_bounds[k]*scale[k], upper_bounds[k]*scale[k]) for k in fit_params]

        x0, fval, grid, jout = scipy.optimize.brute(
                            residual,
                            ranges=ranges,
                            Ns=brute_number_grid_points,
                             full_output=True,
                             disp=verbose)

        n = 0
        p_fit = {}
        for param in fit_params:
            p_fit[param] = x0[n] * scale[param]
            n = n + 1

        return p_fit, fval, (grid, jout)


class pvproHandler:
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
                 start_point_method='last',
                 solver='Nelder-Mead',
                 lower_bounds=dict(
                     diode_factor=0.5,
                     photocurrent_ref=0,
                     saturation_current_ref=0,
                     resistance_series_ref=0,
                     resistance_shunt_ref=0
                 ),
                upper_bounds = dict(
                    diode_factor=2.5,
                    photocurrent_ref=20,
                    saturation_current_ref=1e4,
                    resistance_series_ref=100,
                    resistance_shunt_ref=2e6
                ),
                p0 = dict(
                    diode_factor=1.03,
                    photocurrent_ref=4,
                    saturation_current_ref=1e-11,
                    resistance_series_ref=0.4,
                    resistance_shunt_ref=1e6
                ),
                scale = dict(
                    diode_factor=1,
                    photocurrent_ref=1,
                    saturation_current_ref=1e-11,
                    resistance_series_ref=1,
                    resistance_shunt_ref=1
                ),
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
        self.voltage_key = voltage_key
        self.current_key = current_key
        self.temperature_module_key = temperature_module_key
        self.irradiance_poa_key = irradiance_poa_key
        self.modules_per_string = modules_per_string
        self.parallel_strings = parallel_strings

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.p0 = p0
        self.scale = scale
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
    def df(self,value):
        """
        Set Dataframe by setting the version inside datahandler.

        Parameters
        ----------
        value

        Returns
        -------

        """
        self.dh.data_frame = value

    def make_power_column(self):
        self.df['power_dc'] = self.df[self.voltage_key]*self.df[self.current_key]

    def run_preprocess(self):
        self.dh.run_pipeline(power_col='power_dc',
                            correct_tz=True,
                            extra_cols=[self.temperature_module_key,
                                        self.irradiance_poa_key,
                                        self.voltage_key,
                                        self.current_key]
                            )

    def preprocess_report(self):
        self.dh.report()

    def info(self):
        # max_key_length = np.max([len(k) for k in pvp.__dict__.keys()])

        for k in self.__dict__:
            print('{:34}: {}'.format(k,self.__dict__[k]))

    def remove_nan_from_df(self):
        keys = [self.voltage_key,
                self.current_key,
                self.temperature_module_key,
                self.irradiance_poa_key]

        self.df.dropna(axis=0, subset=keys, inplace=True)

    def calculate_iteration_start_days(self):
        self.iteration_start_days = np.round(
            np.arange(0, self.dataset_length_days - self.days_per_run,
                      self.time_step_between_iterations_days))

    def classify_operation_type(self):


        self.df.loc[:,'operating_cls'] = classify_operating_mode(
            self.df[self.voltage_key],
            self.df[self.current_key])


    def calculate_cell_temperature(self):
        self.df.loc[:,'temperature_cell'] = sapm_module_to_cell_temperature(
            self.df[self.temperature_module_key],
            self.df[self.irradiance_poa_key],
            delta_T=self.delta_T)


    def calculate_iteration_time_step_limits(self):
        pass

    def calculate_iteration_time_axis(self):
        self.time = []
        for d in self.iteration_start_days:
            self.time.append(self.df.index[0]+datetime.timedelta(int(d)))

    def set_fit_params(self):
        self.fit_params = self.p0.keys()


    def get_df_for_iteration(self,k,
                             use_clear_times=False):


        if use_clear_times:


            if not 'clear_time' in self.df.keys():
                raise Exception('Need to find clear times using find_clear_times() first.')

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

    def execute(self,iteration='all',verbose=True,method='minimize',
                save_figs=True,
                save_figs_directory='figures'):

        # Set cell temperature according to model.
        self.calculate_cell_temperature()

        # Fit params taken from p0
        self.set_fit_params()

        pfit = pd.DataFrame(index=range(len(self.iteration_start_days)),
                         columns=[*self.fit_params, *['residual']])

        p0 = pd.DataFrame(index=range(len(self.iteration_start_days)),
                         columns=self.fit_params)

        # for d in range(len(self.iteration_start_days)):
        fit_result = []

        if iteration=='all':
            print('Executing fit on all start days')
            iteration = np.arange(len(self.iteration_start_days))

        n=0
        for k in iteration:

            print('\n--\nPercent complete: {:1.1%}, Iteration: {}'.format(
                    k / len(self.iteration_start_days),k))
            df = self.get_df_for_iteration(k,
                                           use_clear_times=self.use_clear_times)

            number_points_in_time_step_all = len(df)

            # Filter
            df = df[np.logical_and.reduce((
                    df[self.irradiance_poa_key] > self.irradiance_lower_lim,
                    df['temperature_cell'] < self.temperature_cell_upper_lim
                ))].copy()


            if len(df)>10:

                # try:


                # Can update p0 each iteration in future.
                if self.start_point_method=='fixed':
                    p0.loc[k] = self.p0
                elif self.start_point_method=='last':
                    if n==0:
                        p0.loc[k] = self.p0
                    else:
                        p0.loc[k] = pfit_iter
                else:
                    raise ValueError('start_point_method must be "fixed" or "last"')

                pfit_iter, residual, fit_result_iter = production_data_curve_fit(
                    temperature_cell=np.array(df['temperature_cell']),
                    effective_irradiance=np.array(df[self.irradiance_poa_key]),
                    operating_cls=np.array(df['operating_cls']),
                    voltage=df[self.voltage_key] / self.modules_per_string,
                    current=df[self.current_key] / self.parallel_strings,
                    cells_in_series=self.cells_in_series,
                    alpha_isc=self.alpha_isc,
                    lower_bounds=self.lower_bounds,
                    upper_bounds=self.upper_bounds,
                    p0=p0.loc[k],
                    scale=self.scale,
                    verbose=verbose,
                    solver=self.solver,
                    method=method
                )
                pfit.iloc[k] = pfit_iter
                # print(res)
                pfit.loc[k, 'residual'] = residual
                fit_result.append(fit_result_iter)

                if verbose:
                    print('Startpoint:')
                    print(p0.loc[k])
                    print('Fit result:')
                    print(pfit.loc[k])

                if save_figs:
                    self.plot_Vmp_Imp_scatter(p_plot=pfit_iter,
                                          figure_number=100,
                                          iteration=k,
                                          use_clear_times=self.use_clear_times)

                    if not os.path.exists(save_figs_directory):
                        os.mkdir(save_figs_directory)

                    export_folders = [
                        os.path.join(save_figs_directory,'Vmp_Imp'),
                        os.path.join(save_figs_directory,'suns_Voc'),
                    ]
                    for folder in export_folders:
                        if not os.path.exists(folder):
                            os.mkdir(folder)

                    plt.savefig( os.path.join(save_figs_directory,
                                              'Vmp_Imp',
                                              '{}_Vmp-Imp_{}.png'.format(self.system_name,k)),
                                 resolution=200,
                                 bbox_inches='tight')

                    self.plot_suns_voc_scatter(p_plot=pfit_iter,
                                               figure_number=101,
                                               iteration=k,
                                               use_clear_times=self.use_clear_times)
                    plt.savefig( os.path.join(save_figs_directory,
                                              'suns_Voc',
                                              '{}_suns-Voc_{}.png'.format(self.system_name,k)),
                                 resolution=200,
                                 bbox_inches='tight')
                n = n + 1
                # except:
                #     print('** Error with this iteration.')

            self.result = dict(
                p=pfit,
                p0=p0,
                fit_result=fit_result
            )



    def estimate_imp_ref(self,makefigure=False):

        df_estimate = self.df[np.logical_and.reduce((
            self.df[self.irradiance_poa_key] < 1100,
            self.df[self.irradiance_poa_key] > 900
        ))
        ].copy()

        x = df_estimate['temperature_cell']
        y = df_estimate[self.current_key] / df_estimate[
            self.irradiance_poa_key] * 1000 / self.parallel_strings

        #
        # print('estimate imp ref')
        # print('x: ', x)
        # print('df_estimate: ', df_estimate)

        cax = np.logical_and(y > np.nanmean(y) * 0.5, y < np.nanmean(y) * 1.5)
        x = x[cax]
        y = y[cax]

        imp_fit = np.polyfit(x, y, 1)
        imp_ref_estimate = np.polyval(imp_fit, 25)

        if makefigure:
            plt.figure()
            plt.clf()

            x_smooth = np.linspace(x.min(), x.max(), 5)
            plt.hist2d(x, y, bins=(100, 100))
            plt.plot(x_smooth, np.polyval(imp_fit, x_smooth), 'r')
            plt.xlabel('Cell temperature (C)')
            plt.ylabel('Imp (A)')
            plt.show()

        return imp_ref_estimate

    def estimate_photocurrent_ref(self, imp_ref_estimate):
        return imp_ref_estimate/0.934

    def estimate_saturation_current_ref(self,photocurrent_ref_estimate):

        kB = 1.381e-23
        q = 1.602e-19
        T = 25 + 273.15
        Vth = kB * T / q

        saturation_current_ref_estimate = photocurrent_ref_estimate / np.exp(0.600 / (1.0 * Vth))
        return saturation_current_ref_estimate

    def estimate_p0(self):
        """
        Make a rough estimate of the startpoint for fitting the single diode
        model.

        Returns
        -------

        """
        imp_ref = self.estimate_imp_ref()
        photocurrent_ref = self.estimate_photocurrent_ref(imp_ref)
        saturation_current_ref = self.estimate_saturation_current_ref(photocurrent_ref)

        self.p0 = dict(
            diode_factor=1.10,
            photocurrent_ref=photocurrent_ref,
            saturation_current_ref=saturation_current_ref,
            resistance_series_ref=0.4,
            resistance_shunt_ref=1e3
        )

    def plot_Vmp_Imp_scatter(self,
                             p_plot,
                             figure_number=0,
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
        if use_clear_times==None:
            use_clear_times = self.use_clear_times



        # Make figure for inverter on.
        fig = plt.figure(figure_number,figsize=(6.5,3.5))
        plt.clf()
        ax = plt.axes()

        temp_limits = np.linspace(vmin,vmax, 8)

        df = self.get_df_for_iteration(iteration,
                                       use_clear_times=use_clear_times)



        inv_on_points = np.array(df['operating_cls']==0)


        vmp = np.array(df.loc[inv_on_points,self.voltage_key])/self.modules_per_string
        imp = np.array(df.loc[inv_on_points,self.current_key])/self.parallel_strings

        imp_max = np.nanmax(self.df.loc[self.df['operating_cls']==0,self.current_key]/self.parallel_strings)*1.1
        vmp_max = np.nanmax(self.df.loc[self.df['operating_cls']==0,self.voltage_key]/self.modules_per_string)*1.1

        h_sc = plt.scatter(vmp,imp,
                    c=df.loc[inv_on_points,'temperature_cell'],
                    s=0.2,
                    cmap='jet',
                    vmin=0,
                    vmax=70)



        one_sun_points = np.logical_and.reduce((df['operating_cls']==0,
                                                df[self.irradiance_poa_key]>995,
                                                df[self.irradiance_poa_key] < 1005,
                                                ))
        if len(one_sun_points)>0:
            # print('number one sun points: ', len(one_sun_points))
            plt.scatter(df.loc[one_sun_points, self.voltage_key]/self.modules_per_string,
                        df.loc[one_sun_points, self.current_key]/self.parallel_strings,
                        c=df.loc[one_sun_points, 'temperature_cell'],
                        edgecolors='k',
                        s=0.2)



        # Plot temperature scan
        temperature_smooth = np.linspace(0,70,20)

        for effective_irradiance in [100, 1000]:
            voltage_plot, current_plot = pv_system_single_diode_model(
                effective_irradiance=effective_irradiance,
                temperature_cell=temperature_smooth,
                operating_cls=np.zeros_like(temperature_smooth),
                **p_plot,
                cells_in_series=self.cells_in_series,
                alpha_isc=self.alpha_isc,
            )
            plt.plot(voltage_plot, current_plot,'k:')
            plt.text(voltage_plot[-1]-0.5, current_plot[-1],
                     '{:.1g} sun'.format(effective_irradiance/1000),
                     horizontalalignment='right',
                     verticalalignment='center',
                     fontsize=8)




        # Plot irradiance scan
        for j in np.flip(np.arange(len(temp_limits) )):
            temp_curr = temp_limits[j]
            irrad_smooth = np.linspace(1, 1000, 500)

            voltage_plot, current_plot = pv_system_single_diode_model(
                effective_irradiance=irrad_smooth,
               temperature_cell=temp_curr + np.zeros_like(irrad_smooth),
               operating_cls=np.zeros_like(irrad_smooth),
               **p_plot,
               cells_in_series=self.cells_in_series,
               alpha_isc=self.alpha_isc,
               )

            # out = pvlib_fit_fun( np.transpose(np.array(
            #     [irrad_smooth,temp_curr + np.zeros_like(irrad_smooth), np.zeros_like(irrad_smooth) ])),
            #                     *p_plot)

            # Reshape to get V, I
            # out = np.reshape(out,(2,int(len(out)/2)))

            # find the right color to plot.
            # norm_temp = (temp_curr-df[temperature].min())/(df[temperature].max()-df[temperature].min())
            norm_temp = (temp_curr - vmin)/(vmax-vmin)
            line_color = np.array(h_sc.cmap(norm_temp))
            # line_color[0:3] =line_color[0:3]*0.9

            line_color[3]=0.3

            plt.plot(voltage_plot, current_plot,
                         label='Fit {:2.0f} C'.format(temp_curr),
                     color=line_color,
                         # color='C' + str(j)
                     )



        text_str = 'System: {}\n'.format(self.system_name) + \
            'Analysis days: {:.0f}-{:.0f}\n'.format(self.iteration_start_days[iteration],self.iteration_start_days[iteration]+self.days_per_run) + \
        'Current: {}\n'.format(self.current_key) + \
        'Voltage: {}\n'.format(self.voltage_key) + \
        'Temperature: {}\n'.format(self.temperature_module_key) + \
        'Irradiance: {}\n'.format(self.irradiance_poa_key) + \
        'Temperature module->cell delta_T: {}\n'.format(self.delta_T) + \
        'n_diode: {:1.2f} \n'.format(p_plot['diode_factor']) + \
               'reference_photocurrent: {:1.2f} A\n'.format(p_plot['photocurrent_ref']) + \
               'reference_saturation_current: {:1.2f} pA\n'.format(p_plot['saturation_current_ref']*1e12) + \
               'resistance_series: {:1.2f} Ohm\n'.format(p_plot['resistance_series_ref']) + \
               'resistance_shunt: {:1.2f} Ohm\n\n'.format(p_plot['resistance_shunt_ref']) + \
            'Clear time: {}\n'.format(use_clear_times) + \
            'Lower Irrad limit: {}\n'.format(self.irradiance_lower_lim)

        plt.text(0.05,0.95,text_str,
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform = ax.transAxes,
                 fontsize=8)

        plt.xlim([0,vmp_max])
        plt.ylim([0,imp_max])
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        pcbar = plt.colorbar(h_sc)
        pcbar.set_label('Cell Temperature (C)')

        plt.xlabel('Vmp (V)',fontsize=9)
        plt.ylabel('Imp (A)',fontsize=9)



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

        voc = np.array(df.loc[inv_off_points, self.voltage_key]) / self.modules_per_string
        irrad = np.array(df.loc[inv_off_points, self.irradiance_poa_key])

        voc_max = np.nanmax(self.df.loc[self.df['operating_cls'] == 1, self.voltage_key] / self.modules_per_string) * 1.1

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
        for j in np.flip(np.arange(len(temp_limits))):
            temp_curr = temp_limits[j]
            irrad_smooth = np.linspace(1, 1200, 500)

            voltage_plot, current_plot = pv_system_single_diode_model(
                effective_irradiance=irrad_smooth,
                temperature_cell=temp_curr + np.zeros_like(irrad_smooth),
                operating_cls=np.zeros_like(irrad_smooth) + 1,
                **p_plot,
                cells_in_series=self.cells_in_series,
                alpha_isc=self.alpha_isc,
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
                   'reference_saturation_current: {:1.2f} pA\n'.format(
                       p_plot['saturation_current_ref'] * 1e12) + \
                   'resistance_series: {:1.2f} Ohm\n'.format(
                       p_plot['resistance_series_ref']) + \
                   'resistance_shunt: {:1.2f} Ohm\n\n'.format(
                       p_plot['resistance_shunt_ref']) + \
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
