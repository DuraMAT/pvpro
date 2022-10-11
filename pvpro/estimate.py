from array import array
import string
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pvlib.temperature import sapm_cell_from_module
from pvlib.ivtools.sdm import fit_desoto as fit_sdm_desoto
from pvlib.pvsystem import calcparams_desoto, singlediode
from scipy.special import lambertw
from numpy.linalg import pinv

from time import time
from sklearn.linear_model import HuberRegressor

def estimate_imp_ref(poa : array,
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

def estimate_vmp_ref(poa : array,
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

def get_average_diode_factor(technology : string):
    diode_factor_all = {'multi-Si': 1.0229594245606348,
                        'mono-Si': 1.029537190437867,
                        'thin-film': 1.0891515236168812,
                        'cigs': 0.7197110756474172,
                        'cdte': 1.0949601271176865}

    return diode_factor_all[technology]

def estimate_beta_voc(beta_vmp : float, technology : string ='mono-Si'):
    beta_voc_beta_vmp_ratio = {
        'multi-Si': 0.978053067,
        'mono-Si': 0.977447359,
        'thin-film': 0.957477452,
        'cigs': 0.974727107,
        'cdte': 0.979597793}

    beta_voc = beta_vmp * beta_voc_beta_vmp_ratio[technology]

    return beta_voc

def estimate_diode_factor(vmp_ref : array, beta_vmp : float, imp_ref : array,
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
    beta_voc = estimate_beta_voc(beta_vmp, technology=technology)

    # Rough estimate, voc_ref is a little larger than vmp_ref.
    voc_ref = estimate_voc_ref(vmp_ref,technology=technology)

    beta_voc_norm = beta_voc / voc_ref

    delta_0 = (1 - 298.15 * beta_voc_norm) / (
            50.05 - 298.15 * alpha_isc_norm)

    w0 = lambertw(np.exp(1 / delta_0 + 1))

    nNsVth = (vmp_ref + resistance_series * imp_ref) / (w0 - 1)

    diode_factor = nNsVth / (cells_in_series * Vth)

    return diode_factor.real

def estimate_photocurrent_ref_simple(imp_ref: array, technology : string ='mono-Si'):
    photocurrent_imp_ratio = {'multi-Si': 1.0746167586063207,
                              'mono-Si': 1.0723051517913444,
                              'thin-film': 1.1813401654607967,
                              'cigs': 1.1706462692015707,
                              'cdte': 1.1015249105470803}

    photocurrent_ref = imp_ref * photocurrent_imp_ratio[technology]

    return photocurrent_ref

def estimate_saturation_current(isc : array, voc : array, nNsVth : array):
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

def estimate_saturation_current_ref(i_mp : array, 
                                    v_mp : array, 
                                    photocurrent_ref : array,
                                    temperature_cell : array, 
                                    poa : array,
                                    diode_factor : float =1.10,
                                    cells_in_series : int =60,
                                    temperature_ref : float =25,
                                    irradiance_ref : float =1000,
                                    resistance_series : float =0.4,
                                    resistance_shunt_ref : float =400,
                                    EgRef : float =1.121,
                                    dEgdT : float =-0.0002677,
                                    alpha_isc : float =0.001,
                                    figure : bool =False,
                                    figure_number : int =24):
    cax = np.logical_and.reduce((
        i_mp * v_mp > np.nanpercentile(i_mp * v_mp, 1),
        np.isfinite(i_mp),
        np.isfinite(v_mp),
        np.isfinite(temperature_cell),
        np.isfinite(poa),
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
        alpha_sc=alpha_isc,
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
        ax.axvline(I0_ref_mean, color='r', ymax=0.5)
        plt.xscale('log')
        plt.ylabel('Occurrences')
        plt.xlabel('I0_ref (A)')

    return I0_ref.mean()

def estimate_photocurrent_ref(current : array, 
                              voltage : array, 
                              saturation_current_ref : array,
                              temperature_cell : array, 
                              poa : array,
                              diode_factor : float =1.10,
                              cells_in_series : int =60,
                              temperature_ref : float =25,
                              irradiance_ref : float =1000,
                              resistance_series : float =0.4,
                              resistance_shunt_ref : float =400,
                              EgRef : float =1.121,
                              dEgdT : float =-0.0002677,
                              alpha_isc_norm : float =0.001 / 6,
                              figure : bool =False,
                              figure_number : int =25):
    cax = np.logical_and.reduce((
        current * voltage > np.nanpercentile(current * voltage, 1),
        np.isfinite(current),
        np.isfinite(voltage),
        np.isfinite(temperature_cell),
        np.isfinite(poa),
    ))

    current = current[cax]
    voltage = voltage[cax]
    temperature_cell = temperature_cell[cax]
    poa = poa[cax]

    kB = 1.381e-23
    q = 1.602e-19
    Tcell_K = temperature_cell + 273.15
    Tref_K = temperature_ref + 273.15

    nNsVth_ref = diode_factor * cells_in_series * kB * Tref_K / q

    _, I0, Rs, Rsh, nNsVth = calcparams_desoto(
        I_L_ref=1,
        I_o_ref=saturation_current_ref,
        R_sh_ref=resistance_shunt_ref,
        R_s=resistance_series,
        effective_irradiance=poa,
        temp_cell=temperature_cell,
        alpha_sc=1,
        a_ref=nNsVth_ref,
        EgRef=EgRef,
        dEgdT=dEgdT,
        irrad_ref=irradiance_ref,
        temp_ref=temperature_ref)

    IL_ref_all = (current + I0 * np.expm1((voltage + current * Rs) / nNsVth) + (
            voltage + current * Rs) / Rsh) / (poa / irradiance_ref * (
            1 + alpha_isc_norm * (temperature_cell - temperature_ref)))

    IL_ref_mean = np.mean(IL_ref_all)

    if figure:
        plt.figure(figure_number)
        plt.clf()
        ax = plt.gca()
        bin_min = np.median(IL_ref_all) * 0.5
        bin_max = np.median(IL_ref_all) * 1.5

        bins = np.linspace(bin_min ** 0.1, bin_max ** 0.1, 150) ** 10
        plt.hist(IL_ref_all, bins=bins)
        ax.axvline(IL_ref_mean, color='r', ymax=0.5)
        plt.ylabel('Occurrences')
        plt.xlabel('photocurrent_ref (A)')

    return IL_ref_mean

def estimate_cells_in_series(voc_ref : array, technology : string ='mono-Si'):
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

def estimate_voc_ref(vmp_ref : array, technology : string =None):
    voc_vmp_ratio = {'thin-film': 1.3069503474012514,
                     'multi-Si': 1.2365223483476515,
                     'cigs': 1.2583291018540534,
                     'mono-Si': 1.230866745147029,
                     'cdte': 1.2188176469944012}
    voc_ref = vmp_ref * voc_vmp_ratio[technology]
    

    return voc_ref

def estimate_beta_voc(beta_vmp : float, technology : string ='mono-Si'):
    beta_voc_to_beta_vmp_ratio = {'thin-film': 0.9594252453485964,
                                  'multi-Si': 0.9782579114165342,
                                  'cigs': 0.9757373267198366,
                                  'mono-Si': 0.9768254239046427,
                                  'cdte': 0.9797816054754396}
    beta_voc = beta_vmp * beta_voc_to_beta_vmp_ratio[technology]
    return beta_voc

def estimate_alpha_isc(isc : array, technology : string):
    alpha_isc_to_isc_ratio = {'multi-Si': 0.0005864523754010862,
                              'mono-Si': 0.0005022410194560715,
                              'thin-film': 0.00039741211251133725,
                              'cigs': -8.422066533574996e-05,
                              'cdte': 0.0005573603056215652}

    alpha_isc = isc * alpha_isc_to_isc_ratio[technology]
    return alpha_isc

def estimate_isc_ref(imp_ref : array, technology: string):
    isc_to_imp_ratio = {'multi-Si': 1.0699135787527263,
                        'mono-Si': 1.0671785412770871,
                        'thin-film': 1.158663685900219,
                        'cigs': 1.1566217151572733, 
                        'cdte': 1.0962996330688608}

    isc_ref = imp_ref * isc_to_imp_ratio[technology]

    return isc_ref

def estimate_resistance_series_simple(vmp : array, imp : array,
                                      saturation_current : array,
                                      photocurrent : array,
                                      nNsVth : array):
    Rs = (nNsVth * np.log1p(
        (photocurrent - imp) / saturation_current) - vmp) / imp
    return Rs

def estimate_resistance_series(poa : array,
                               temperature_cell : array,
                               voltage : array,
                               current : array,
                               photocurrent_ref : array,
                               saturation_current_ref : array,
                               diode_factor : array,
                               cells_in_series : int =60,
                               temperature_ref : float =25,
                               irradiance_ref : float=1000,
                               resistance_shunt_ref : float=400,
                               EgRef : float =1.121,
                               dEgdT : float =-0.0002677,
                               alpha_isc : float =0.001,
                               figure : bool =False,
                               figure_number :int =26,
                               verbose : bool =False
                               ):
    cax = np.logical_and.reduce((
        current * voltage > np.nanpercentile(current * voltage, 70),
        np.isfinite(current),
        np.isfinite(voltage),
        np.isfinite(temperature_cell),
        np.isfinite(poa),
    ))

    current = current[cax]
    voltage = voltage[cax]
    temperature_cell = temperature_cell[cax]
    poa = poa[cax]

    kB = 1.381e-23
    q = 1.602e-19
    Tcell_K = temperature_cell + 273.15
    Tref_K = temperature_ref + 273.15

    nNsVth_ref = diode_factor * cells_in_series * kB * (
            temperature_ref + 273.15) / q
    IL, I0, _, Rsh, nNsVth = calcparams_desoto(
        I_L_ref=photocurrent_ref,
        I_o_ref=saturation_current_ref,
        R_sh_ref=resistance_shunt_ref,
        R_s=1,
        effective_irradiance=poa,
        temp_cell=temperature_cell,
        alpha_sc=alpha_isc,
        a_ref=nNsVth_ref,
        EgRef=EgRef,
        dEgdT=dEgdT,
        irrad_ref=irradiance_ref,
        temp_ref=temperature_ref)

    Wz = lambertw(
        I0 * Rsh / nNsVth * np.exp(Rsh / nNsVth * (IL + I0 - current)))
    Rs = 1 / current * (Rsh * (IL + I0 - current) - voltage - nNsVth * Wz)

    Rs_mean = np.abs(np.mean(Rs))
    if figure:
        plt.figure(figure_number)
        plt.clf()
        ax = plt.gca()
        bin_min = 0
        bin_max = np.abs(np.max(Rs)) * 1.5

        bins = np.linspace(bin_min, bin_max, 100)
        plt.hist(Rs, bins=bins)
        ax.axvline(Rs_mean, color='r', ymax=0.5)
        plt.ylabel('Occurrences')
        plt.xlabel('Series Resistance (Ohm)')

    return Rs_mean

def estimate_singlediode_params(poa: array,
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

    start_time = time()
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

    out = estimate_imp_ref(poa=poa,
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

    out = estimate_vmp_ref(
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
    voc_ref = estimate_voc_ref(vmp_ref, technology=technology)
    if cells_in_series == None:
        cells_in_series = estimate_cells_in_series(voc_ref,
                                                   technology=technology)

    diode_factor = estimate_diode_factor(vmp_ref=vmp_ref,
                                         beta_vmp=beta_vmp,
                                         imp_ref=imp_ref,
                                         technology=technology,
                                         cells_in_series=cells_in_series)

    kB = 1.381e-23
    q = 1.602e-19

    nNsVth_ref = diode_factor * cells_in_series * kB * (
            temperature_ref + 273.15) / q

    beta_voc = estimate_beta_voc(beta_vmp, technology=technology)

    photocurrent_ref = estimate_photocurrent_ref_simple(imp_ref,
                                                        technology=technology)

    isc_ref = estimate_isc_ref(imp_ref, technology=technology)
    if verbose:
        print('Simple isc_ref estimate: {}'.format(isc_ref))

    saturation_current_ref = estimate_saturation_current(isc=isc_ref,
                                                         voc=voc_ref,
                                                         nNsVth=nNsVth_ref,
                                                         )
    if verbose:
        print('Simple saturation current ref estimate: {}'.format(
            saturation_current_ref))

    if alpha_isc == None:
        alpha_isc = estimate_alpha_isc(isc_ref, technology=technology)

    kB = 1.381e-23
    q = 1.602e-19
    Tref = temperature_ref + 273.15

    nNsVth_ref = diode_factor * cells_in_series * kB * Tref / q

    if resistance_series_ref == None:
        resistance_series_ref = estimate_resistance_series_simple(vmp_ref,
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
