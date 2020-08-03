import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvlib.temperature import sapm_cell_from_module
from pvlib.ivtools import fit_sdm_desoto
from pvlib.pvsystem import calcparams_desoto


def estimate_imp_ref(irradiance_poa,
                     temperature_cell,
                     imp,
                     figure=False,
                     figure_number=11,
                     temperature_ref=25,
                     irradiance_ref=1000
                     ):
    """
    Estimate imp_ref using operation data. Note that typically imp for an
    array would be divided by parallel_strings before calling this function.
    This function can also be used to estimate pmp_ref and gamma_pmp

    Parameters
    ----------
    irradiance_poa
    temperature_cell
    imp
    makefigure

    Returns
    -------

    """


    # print('irradiance poa',irradiance_poa)

    cax = np.logical_and.reduce((
        irradiance_poa > np.nanpercentile(irradiance_poa, 50),
        irradiance_poa < 1100,
        imp < np.nanpercentile(imp, 95),
        imp > np.nanpercentile(imp, 10)
    ))

    cax = irradiance_poa > np.nanpercentile(irradiance_poa, 30)

    x = temperature_cell[cax]
    y = imp[cax] / irradiance_poa[cax] * irradiance_ref

    # # cax = np.logical_and(y > np.nanpercentile(y,80), y < np.nanmean(y) * 1.5)
    # cax = y > np.nanpercentile(y,50)
    # x = x[cax]
    # y = y[cax]

    imp_fit = np.polyfit(x, y, 1)
    imp_ref_estimate = np.polyval(imp_fit, temperature_ref)
    alpha_imp = imp_fit[0]

    if figure:
        plt.figure(figure_number)
        plt.clf()

        x_smooth = np.linspace(x.min(), x.max(), 5)
        plt.hist2d(x, y, bins=(100, 100))
        plt.plot(x_smooth, np.polyval(imp_fit, x_smooth), 'r')
        plt.xlabel('Cell temperature (C)')
        plt.ylabel('Imp (A)')
        plt.show()

    return imp_ref_estimate, alpha_imp


def estimate_pmp_ref(irradiance_poa,
                     temperature_cell,
                     pmp,
                     figure=False,
                     figure_number=12,
                     ):
    pmp_ref, gamma_pmp = estimate_imp_ref(irradiance_poa,
                                          temperature_cell, pmp,
                                          figure=figure,
                                          figure_number=figure_number)

    return pmp_ref, gamma_pmp


def estimate_vmp_ref(irradiance_poa,
                     temperature_cell,
                     imp,
                     vmp,
                     figure=False,
                     figure_number=12,
                     ):
    """
    Estimate vmp_ref using operation data. Note that typically vmp for an
    array would be divided by parallel_strings before calling this function.

    vmp does not depend much on irradiance so long as the irradiance is not
    too low.

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
        irradiance_poa > np.nanpercentile(irradiance_poa, 70),
        irradiance_poa < 1100,
        imp > np.nanmax(imp) * 0.5
    ))

    x = temperature_cell[cax]
    y = vmp[cax]

    # # cax = np.logical_and(y > np.nanpercentile(y,80), y < np.nanmean(y) * 1.5)
    # cax = y > np.nanpercentile(y,50)
    # x = x[cax]
    # y = y[cax]

    vmp_fit = np.polyfit(x, y, 1)
    vmp_ref_estimate = np.polyval(vmp_fit, 25)
    beta_vmp = vmp_fit[0]

    if figure:
        plt.figure(figure_number)
        plt.clf()

        x_smooth = np.linspace(x.min(), x.max(), 5)
        plt.hist2d(x, y, bins=(100, 100))
        plt.plot(x_smooth, np.polyval(vmp_fit, x_smooth), 'r')
        plt.xlabel('Cell temperature (C)')
        plt.ylabel('Vmp (V)')
        plt.show()

    return vmp_ref_estimate, beta_vmp


def estimate_diode_factor(technology):
    diode_factor_all = {'multi-Si': 1.0229594245606348,
                        'mono-Si': 1.029537190437867,
                        'thin-film': 1.0891515236168812,
                        'cigs': 0.7197110756474172,
                        'cdte': 1.0949601271176865}

    return diode_factor_all[technology]


def estimate_photocurrent_ref(imp_ref, technology='mono-Si'):
    photocurrent_imp_ratio = {'multi-Si': 1.0746167586063207,
                              'mono-Si': 1.0723051517913444,
                              'thin-film': 1.1813401654607967,
                              'cigs': 1.1706462692015707,
                              'cdte': 1.1015249105470803}

    photocurrent_ref = imp_ref * photocurrent_imp_ratio[technology]

    return photocurrent_ref



def estimate_saturation_current_full(imp_ref,
                                    photocurrent_ref,
                                    vmp_ref,
                                    cells_in_series,
                                    resistance_series_ref=0.4,
                                    resistance_shunt_ref=100,
                                    diode_factor=1.1,
                                    temperature_ref=25,
                                    ):
    """
    If vmp or cells_in_series is unknown, use vmp_ref=0.6 and cells_in_series=1

    Parameters
    ----------
    imp_ref
    photocurrent_ref
    vmp_ref
    cells_in_series
    resistance_series_ref
    resistance_shunt_ref

    Returns
    -------

    """
    kB = 1.381e-23
    q = 1.602e-19
    T = temperature_ref + 273.15
    Vth = kB * T / q
    # voc_cell = 0.6

    Rsh = resistance_shunt_ref
    Rs = resistance_series_ref

    nNsVth = diode_factor * cells_in_series * Vth

    saturation_current_ref_estimate = (photocurrent_ref - imp_ref - (
            vmp_ref + imp_ref * Rs) / Rsh) / np.exp(
        (vmp_ref + imp_ref * Rs) / nNsVth)

    return saturation_current_ref_estimate

# def estimate_saturation_current_ref(isc,voc,cells_in_series,temperature_ref=25):
#     """
#         .. [2] John A Dufﬁe, William A Beckman, "Solar Engineering of Thermal
#        Processes", Wiley, 2013
#
#     Parameters
#     ----------
#     isc
#     voc
#     cells_in_series
#     temperature_ref
#
#     Returns
#     -------
#
#     """
#
#     # initial guesses of variables for computing convergence:
#     # Values are taken from [2], p753
#     Rsh_0 = 100.0
#     kB = 1.381e-23
#     q = 1.602e-19
#     Tref = temperature_ref + 273.15
#     a_0 = 1.5*kB*Tref*cells_in_series/q
#
#     I0 = isc * np.exp(-voc/a_0)
#
#     return I0


def estimate_saturation_current(isc, voc, nNsVth):
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


def estimate_saturation_current_ref(i_mp, v_mp, photocurrent_ref,
                                    temperature_cell, poa,
                                    diode_factor=1.15,
                                    cells_in_series=60,
                                    temperature_ref=25,
                                    irradiance_ref=1000,
                                    resistance_series=0.4,
                                    resistance_shunt_ref=400,
                                    EgRef=1.121,
                                    dEgdT=-0.0002677,
                                    alpha_sc=0.001,
                                    figure=False,
                                    figure_number=15):
    cax = np.logical_and.reduce((
        i_mp * v_mp > np.nanpercentile(i_mp * v_mp, 1),
    ))

    i_mp = i_mp[cax]
    v_mp = v_mp[cax]
    temperature_cell = temperature_cell[cax]
    poa = poa[cax]

    kB = 1.381e-23
    q = 1.602e-19
    Tcell_K = temperature_cell + 273.15
    Tref_K = temperature_ref + 273.15

    nNsVth_ref = diode_factor * cells_in_series * kB * Tref_K / q


    IL, I0ref_to_I0, Rs, Rsh, nNsVth = calcparams_desoto(
        I_L_ref=photocurrent_ref,
        I_o_ref=1,
        R_sh_ref=resistance_shunt_ref,
        R_s=resistance_series,
        effective_irradiance=poa,
        temp_cell=temperature_cell,
        alpha_sc=alpha_sc,
        a_ref=nNsVth_ref,
        EgRef=EgRef,
        dEgdT=dEgdT,
        irrad_ref=irradiance_ref,
        temp_ref=temperature_ref)


    I0 = (IL - i_mp - (v_mp + i_mp * Rs) / Rsh) / np.expm1(
        (v_mp + i_mp * Rs) / nNsVth)

    I0_ref = I0 / I0ref_to_I0
    I0_ref_mean = np.mean(I0_ref)

    if figure:
        plt.figure(figure_number)
        plt.clf()
        ax = plt.gca()
        bin_min = np.median(I0_ref) * 1e-2
        bin_max = np.median(I0_ref) * 1e2

        bins = np.linspace(bin_min ** 0.1, bin_max ** 0.1, 150) ** 10
        plt.hist(I0_ref, bins=bins)
        ax.axvline(I0_ref_mean,color='r',ymax=0.5)
        plt.xscale('log')
        plt.ylabel('Occurrences')
        plt.xlabel('I0_ref (A)')

    # plt.figure(15)
    # plt.clf()
    # plt.hist2d(temperature_cell, np.log10(I0_ref/1e-9),bins=100)
    # plt.ylabel('log10(I0/nA)')
    # plt.xlabel('Temperature Cell (C)')

    return I0_ref.mean()

def estimate_cells_in_series(voc_ref, technology='mono-Si'):
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


def estimate_voc_ref(vmp_ref, technology='mono-Si'):
    voc_vmp_ratio = {'thin-film': 1.3069503474012514,
                     'multi-Si': 1.2365223483476515,
                     'cigs': 1.2583291018540534,
                     'mono-Si': 1.230866745147029,
                     'cdte': 1.2188176469944012}
    voc_ref = vmp_ref * voc_vmp_ratio[technology]

    return voc_ref


def estimate_beta_voc(beta_vmp, technology='mono-Si'):
    beta_voc_to_beta_vmp_ratio = {'thin-film': 0.9594252453485964,
                                  'multi-Si': 0.9782579114165342,
                                  'cigs': 0.9757373267198366,
                                  'mono-Si': 0.9768254239046427,
                                  'cdte': 0.9797816054754396}
    beta_voc = beta_vmp * beta_voc_to_beta_vmp_ratio[technology]
    return beta_voc


def estimate_alpha_isc(isc, technology):
    alpha_isc_to_isc_ratio = {'multi-Si': 0.0005864523754010862,
                              'mono-Si': 0.0005022410194560715,
                              'thin-film': 0.00039741211251133725,
                              'cigs': -8.422066533574996e-05,
                              'cdte': 0.0005573603056215652}

    alpha_isc = isc * alpha_isc_to_isc_ratio[technology]
    return alpha_isc


def estimate_isc_ref(imp_ref, technology):
    isc_to_imp_ratio = {'multi-Si': 1.0699135787527263,
                        'mono-Si': 1.0671785412770871,
                        'thin-film': 1.158663685900219,
                        'cigs': 1.1566217151572733, 'cdte': 1.0962996330688608}

    isc_ref = imp_ref * isc_to_imp_ratio[technology]

    return isc_ref


def estimate_resistance_series(vmp, imp, saturation_current, photocurrent,
                               nNsVth):
    Rs = (nNsVth * np.log1p(
        (photocurrent - imp) / saturation_current) - vmp) / imp
    return Rs


# def estimate_shunt_resistance_ref():


def estimate_singlediode_params(irradiance_poa,
                                temperature_module,
                                vmp,
                                imp,
                                delta_T=3,
                                alpha_isc=None,
                                cells_in_series=None,
                                technology='mono-Si',
                                temperature_ref=25,
                                irradiance_ref=1000,
                                resistance_shunt_ref=400,
                                figure=False):
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

    if irradiance_poa.size == 0:
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

    irradiance_poa = np.array(irradiance_poa)
    temperature_module = np.array(temperature_module)
    vmp = np.array(vmp)
    imp = np.array(imp)

    temperature_cell = sapm_cell_from_module(
        module_temperature=temperature_module,
        poa_global=irradiance_poa,
        deltaT=delta_T,
        irrad_ref=irradiance_ref)

    imp_ref, alpha_imp = estimate_imp_ref(irradiance_poa, temperature_cell, imp,
                                          figure=figure,
                                          temperature_ref=temperature_ref,
                                          irradiance_ref=irradiance_ref)

    vmp_ref, beta_vmp = estimate_vmp_ref(irradiance_poa, temperature_cell, imp,
                                         vmp,
                                         figure=figure)

    pmp_ref, gamma_pmp = estimate_pmp_ref(irradiance_poa, temperature_cell,
                                          imp * vmp,
                                          figure=figure)

    diode_factor = estimate_diode_factor(technology)
    voc_ref = estimate_voc_ref(vmp_ref, technology=technology)

    if cells_in_series == None:
        cells_in_series = estimate_cells_in_series(voc_ref,
                                                   technology=technology)
    kB = 1.381e-23
    q = 1.602e-19

    nNsVth_ref = diode_factor * cells_in_series * kB * (
                temperature_ref + 273.15) / q

    beta_voc = estimate_beta_voc(beta_vmp, technology=technology)

    photocurrent_ref = estimate_photocurrent_ref(imp_ref, technology=technology)

    # saturation_current_ref = estimate_saturation_current_full(
    #     imp_ref=imp_ref, photocurrent_ref=photocurrent_ref, vmp_ref=vmp_ref,
    #     resistance_series_ref=0.4,
    #     cells_in_series=cells_in_series)

    isc_ref = estimate_isc_ref(imp_ref, technology=technology)

    saturation_current_ref_rough_est = estimate_saturation_current(isc=isc_ref,
                                                         voc=voc_ref,
                                                         nNsVth=nNsVth_ref)



    if alpha_isc == None:
        alpha_isc = estimate_alpha_isc(isc_ref, technology=technology)

    kB = 1.381e-23
    q = 1.602e-19
    Tref = temperature_ref + 273.15

    nNsVth_ref = diode_factor * cells_in_series * kB * Tref / q

    resistance_series_ref = estimate_resistance_series(vmp_ref,
                                                       imp_ref,
                                                       saturation_current_ref_rough_est,
                                                       photocurrent_ref,
                                                       nNsVth=nNsVth_ref)

    saturation_current_ref = estimate_saturation_current_ref(
        i_mp=imp,
        v_mp=vmp,
        photocurrent_ref=photocurrent_ref,
        temperature_cell=temperature_cell,
        poa=irradiance_poa,
        cells_in_series=cells_in_series,
        resistance_series=resistance_series_ref,
        resistance_shunt_ref=resistance_shunt_ref,
        figure=False)

    # desoto = fit_sdm_desoto(vmp_ref, imp_ref, voc_ref, isc_ref,
    #                         alpha_isc, beta_voc, cells_in_series,
    #                         EgRef=1.121, dEgdT=-0.0002677,
    #                         temp_ref=temperature_ref, irrad_ref=irradiance_ref,
    #                         root_kwargs={})
    # print(desoto)

    params = dict(
        diode_factor=diode_factor,
        photocurrent_ref=photocurrent_ref,
        saturation_current_ref=saturation_current_ref,
        resistance_series_ref=resistance_series_ref,
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
        gamma_pmp=gamma_pmp,
        cells_in_series=cells_in_series,
        nNsVth_ref=nNsVth_ref,
    )

    return params
