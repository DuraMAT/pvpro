"""
Functions about single diode modelling
"""

import numpy as np
import pandas as pd

from array import array
from typing import Union
from pvlib.pvsystem import calcparams_desoto, singlediode
from pvlib.singlediode import _lambertw_i_from_v, _lambertw_v_from_i, \
    bishop88_mpp, bishop88_v_from_i

def single_diode_predict(pvbasics,
                        effective_irradiance : pd.Series,
                        temperature_cell : pd.Series,
                        operating_cls : pd.Series,
                        params : dict
                        ):

    voltage, current = pv_system_single_diode_model(
        effective_irradiance=effective_irradiance,
        temperature_cell=temperature_cell,
        operating_cls=operating_cls,
        cells_in_series=pvbasics.cells_in_series,
        alpha_isc=pvbasics.alpha_isc,
        resistance_shunt_ref=params['resistance_shunt_ref'],
        diode_factor=params['diode_factor'],
        photocurrent_ref=params['photocurrent_ref'],
        saturation_current_ref=params['saturation_current_ref'],
        resistance_series_ref=params['resistance_series_ref'],
        conductance_shunt_extra=params['conductance_shunt_extra'],
        band_gap_ref = pvbasics.Eg_ref,
        dEgdT = pvbasics.dEgdT
    )

    return voltage, current

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

def calcparams_pvpro(effective_irradiance : pd.Series, temperature_cell : pd.Series,
                    alpha_isc : pd.Series, nNsVth_ref : pd.Series, photocurrent_ref : pd.Series,
                    saturation_current_ref : pd.Series,
                    resistance_shunt_ref : pd.Series, resistance_series_ref : pd.Series,
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

    Gsh_extra = 1e-5 # To avoid Rsh becomes unbounded when G is near 0
    rsh = 1/(1/rsh+Gsh_extra)
    

    return iph, io, rs, rsh, nNsVth

def singlediode_fast(photocurrent : pd.Series, 
                    saturation_current : pd.Series, 
                    resistance_series : pd.Series,
                    resistance_shunt : pd.Series, nNsVth : pd.Series, calculate_voc : bool =False):
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
        effective_irradiance : Union[pd.Series, float],
        temperature_cell : Union[pd.Series, float],
        resistance_shunt_ref : pd.Series,
        resistance_series_ref : pd.Series,
        diode_factor : pd.Series,
        cells_in_series : pd.Series,
        alpha_isc : pd.Series,
        photocurrent_ref : pd.Series,
        saturation_current_ref : pd.Series,
        conductance_shunt_extra : pd.Series = 0,
        irradiance_ref : float =1000,
        temperature_ref : float =25,
        ivcurve_pnts : int =None,
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

    iph, io, rs, rsh, nNsVth = calcparams_pvpro(effective_irradiance = effective_irradiance,
                                                temperature_cell = temperature_cell,
                                                alpha_isc = alpha_isc,
                                                nNsVth_ref = nNsVth_ref,
                                                photocurrent_ref = photocurrent_ref,
                                                saturation_current_ref = saturation_current_ref,
                                                resistance_shunt_ref = resistance_shunt_ref,
                                                resistance_series_ref = resistance_series_ref,
                                                band_gap_ref = band_gap_ref, 
                                                dEgdT=dEgdT,
                                                irradiance_ref=irradiance_ref,
                                                temperature_ref=temperature_ref)
    # if len(iph)>1: 
    #     iph[iph <= 0] = 0 

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
        effective_irradiance : pd.Series,
        temperature_cell : pd.Series,
        operating_cls : pd.Series,
        diode_factor : pd.Series,
        photocurrent_ref : pd.Series,
        saturation_current_ref : pd.Series,
        resistance_series_ref : pd.Series,
        conductance_shunt_extra : pd.Series,
        resistance_shunt_ref : pd.Series,
        cells_in_series : int,
        alpha_isc : float,
        voltage_operation : pd.Series = None,
        current_operation : pd.Series = None,
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
