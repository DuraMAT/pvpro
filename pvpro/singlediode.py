# import pvlib
import numpy as np

from pvlib.singlediode import _lambertw_i_from_v, _lambertw_v_from_i
from pvlib.pvsystem import calcparams_desoto, singlediode

# from pvterms import rename

def calcparams_pvpro(effective_irradiance, temperature_cell,
                     alpha_isc, nNsVth_ref, photocurrent_ref,
                     saturation_current_ref,
                     resistance_shunt_ref, resistance_series_ref,
                     conductance_shunt_extra,
                     Eg_ref=1.121, dEgdT=-0.0002677,
                     irradiance_ref=1000, temperature_ref=25):
    """
    Similar to pvlib calcparams_desoto, except an extra shunt conductance is
    added.

    Parameters
    ----------
    effective_irradiance
    temperature_cell
    alpha_isc
    nNsVth_ref
    photocurrent_ref
    saturation_current_ref
    resistance_shunt_ref
    resistance_series_ref
    conductance_shunt_extra
    Eg_ref
    dEgdT
    irradiance_ref
    temperature_ref

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
        EgRef=Eg_ref,
        dEgdT=dEgdT,
        irrad_ref=irradiance_ref,
        temp_ref=temperature_ref
    )


    # Increase shunt resistance by the extra shunt conductance, avoid divide by zero error.
    rsh = 1 / (np.abs(1 / rsh) + np.abs(conductance_shunt_extra) + 1e-100)


    return iph, io, rs, rsh, nNsVth


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
        irradiance_ref=1000,
        temperature_ref=25,
        method='newton',
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
    nNsVth_ref = diode_factor * cells_in_series * kB / q * (
            273.15 + temperature_ref)

    iph, io, rs, rsh, nNsVth = calcparams_pvpro(effective_irradiance,
                                                temperature_cell,
                                                alpha_isc,
                                                nNsVth_ref,
                                                photocurrent_ref,
                                                saturation_current_ref,
                                                resistance_shunt_ref,
                                                resistance_series_ref,
                                                conductance_shunt_extra,
                                                Eg_ref=Eg_ref, dEgdT=dEgdT,
                                                irradiance_ref=irradiance_ref,
                                                temperature_ref=temperature_ref)

    out = singlediode(iph,
                      io,
                      rs,
                      rsh,
                      nNsVth,
                      method=method,
                      ivcurve_pnts=ivcurve_pnts,
                      )
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


def calculate_temperature_coeffs(
        diode_factor,
        photocurrent_ref,
        saturation_current_ref,
        resistance_series_ref,
        resistance_shunt_ref,
        cells_in_series,
        alpha_isc,
        conductance_shunt_extra=0,
        effective_irradiance=1000,
        temperature_cell=25,
        band_gap_ref=1.121,
        dEgdT=-0.0002677,
        method='newton',
        irradiance_ref=1000,
        temperature_ref=25
):
    """
    Calculate temperature coefficients given single diode model parameters.

    Parameters
    ----------
    diode_factor
    photocurrent_ref
    saturation_current_ref
    resistance_series_ref
    resistance_shunt_ref
    cells_in_series
    alpha_isc
    conductance_shunt_extra
    effective_irradiance
    temperature_cell
    band_gap_ref
    dEgdT
    method
    irradiance_ref
    temperature_ref

    Returns
    -------

    """
    out = []
    for temperature_offset in [0, 1]:
        out_iter = pvlib_single_diode(
            effective_irradiance=effective_irradiance,
            temperature_cell=temperature_cell + temperature_offset,
            resistance_shunt_ref=resistance_shunt_ref,
            resistance_series_ref=resistance_series_ref,
            diode_factor=diode_factor,
            cells_in_series=cells_in_series,
            alpha_isc=alpha_isc,
            photocurrent_ref=photocurrent_ref,
            saturation_current_ref=saturation_current_ref,
            Eg_ref=band_gap_ref,
            dEgdT=dEgdT,
            conductance_shunt_extra=conductance_shunt_extra,
            irradiance_ref=irradiance_ref,
            temperature_ref=temperature_ref,
            method=method,
            ivcurve_pnts=None,
            output_all_params=True
        )
        out.append(out_iter)

    temp_co_name = {'i_sc': 'alpha_isc',
                    'v_oc': 'beta_voc',
                    'i_mp': 'alpha_imp',
                    'v_mp': 'beta_vmp',
                    'p_mp': 'gamma_pmp',
                    'i_x': 'alpha_i_x',
                    'i_xx': 'alpha_i_xx',
                    'photocurrent': 'tempco_photocurrent',
                    'saturation_current': 'tempco_saturation_current',
                    'resistance_series': 'tempco_resistance_series',
                    'resistace_shunt': 'tempco_resistance_shunt',
                    'nNsVth': 'tempco_nNsVth'
                    }
# temp_co_name = {'i_sc': 'alpha_isc',
#                     'v_oc': 'beta_voc',
#                     'i_mp': 'alpha_imp',
#                     'v_mp': 'beta_vmp',
#                     'p_mp': 'gamma_pmp',
#                     'i_x': 'alpha_i_x',
#                     'i_xx': 'alpha_i_xx',
#                     'photocurrent': 'tempco_photocurrent',
#                     'saturation_current': 'tempco_saturation_current',
#                     'resistance_series': 'tempco_resistance_series',
#                     'resistace_shunt': 'tempco_resistance_shunt',
#                     'nNsVth': 'tempco_nNsVth'
#                     }

    temp_co = {}
    for p in out_iter:
        temp_co[temp_co_name[p]] = out[1][p] - out[0][p]
    return temp_co


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

    """
    Find the closest point on the IV curve, using brute force calculation.

    Parameters
    ----------
    voltage
    current
    effective_irradiance
    temperature_cell
    resistance_shunt_ref
    resistance_series_ref
    diode_factor
    cells_in_series
    alpha_isc
    photocurrent_ref
    saturation_current_ref
    Eg_ref
    dEgdT
    reference_irradiance
    reference_temperature
    method
    verbose
    ivcurve_pnts

    Returns
    -------

    """
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
    """
    Calculate voltage at a particular point on the IV curve.

    Parameters
    ----------
    current
    effective_irradiance
    temperature_cell
    resistance_shunt_ref
    resistance_series_ref
    diode_factor
    cells_in_series
    alpha_isc
    photocurrent_ref
    saturation_current_ref
    Eg_ref
    dEgdT
    reference_irradiance
    reference_temperature
    method
    verbose

    Returns
    -------

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
    """
    Calculate current at a particular voltage on the IV curve.

    Parameters
    ----------
    voltage
    effective_irradiance
    temperature_cell
    resistance_shunt_ref
    resistance_series_ref
    diode_factor
    cells_in_series
    alpha_isc
    photocurrent_ref
    saturation_current_ref
    Eg_ref
    dEgdT
    reference_irradiance
    reference_temperature
    method
    verbose

    Returns
    -------

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
        EgRef=Eg_ref,
        dEgdT=dEgdT,
    )
    current = _lambertw_i_from_v(rsh, rs, nNsVth, voltage, io, iph)

    return current


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

    Parameters
    ----------
    effective_irradiance
    temperature_cell
    operating_cls
    diode_factor
    photocurrent_ref
    saturation_current_ref
    resistance_series_ref
    conductance_shunt_extra
    resistance_shunt_ref
    cells_in_series
    alpha_isc
    voltage_operation
    current_operation
    band_gap_ref
    dEgdT
    method
    kwargs

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
