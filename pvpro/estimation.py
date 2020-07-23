import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvlib.temperature import sapm_cell_from_module


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
        imp < np.nanpercentile(imp,95),
        imp > np.nanpercentile(imp,10)
    ))

    x = temperature_cell[cax]
    y = imp[cax] / irradiance_poa[cax] * 1000

    # # cax = np.logical_and(y > np.nanpercentile(y,80), y < np.nanmean(y) * 1.5)
    # cax = y > np.nanpercentile(y,50)
    # x = x[cax]
    # y = y[cax]

    imp_fit = np.polyfit(x, y, 1)
    imp_ref_estimate = np.polyval(imp_fit, 25)
    print(imp_fit)

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


def estimate_singlediode_params(irradiance_poa,
                        temperature_module,
                        vmp,
                        imp,
                        delta_T = 3,
                        resistance_series_ref=0.4,
                        irradiance_ref=1000,
                        cells_in_series=None,
                        figure=False,
                        all_params=True):

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

    temperature_cell = sapm_cell_from_module(module_temperature=temperature_module,
                                             poa_global=irradiance_poa,
                                             deltaT=delta_T,
                                             irrad_ref=irradiance_ref)

    imp_ref = estimate_imp_ref(irradiance_poa, temperature_cell, imp,
                               figure=figure)

    vmp_ref = estimate_vmp_ref(irradiance_poa, temperature_cell, vmp,
                               figure=figure)
    photocurrent_ref = estimate_photocurrent_ref(imp_ref)

    saturation_current_ref = estimate_saturation_current_ref(
        imp_ref=imp_ref, photocurrent_ref=photocurrent_ref, vmp_ref=vmp_ref,
        resistance_series_ref=resistance_series_ref,
        cells_in_series=cells_in_series)


    params = dict(
        diode_factor=1.15,
        photocurrent_ref=photocurrent_ref,
        saturation_current_ref=saturation_current_ref,
        resistance_series_ref=resistance_series_ref,
        conductance_shunt_extra=0.001
    )
    if all_params:
        params['imp_ref'] = imp_ref
        params['vmp_ref'] = vmp_ref
        params['pmp_ref'] = imp_ref*vmp_ref

    return params
