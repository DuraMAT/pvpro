import numpy as np
import pandas as pd

# import matplotlib
#
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pvlib.temperature import sapm_cell_from_module
from pvlib.ivtools.sdm import fit_desoto as fit_sdm_desoto
from pvlib.pvsystem import calcparams_desoto, singlediode
from scipy.special import lambertw
from numpy.linalg import pinv

from time import time


# def estimate_imp_ref(poa,
#                      temperature_cell,
#                      imp,
#                      figure=False,
#                      figure_number=11,
#                      temperature_ref=25,
#                      temperature_fit_range=20,
#                      irradiance_ref=1000
#                      ):
#     """
#     Estimate imp_ref using operation data. Note that typically imp for an
#     array would be divided by parallel_strings before calling this function.
#     This function can also be used to estimate pmp_ref and gamma_pmp
#
#     Parameters
#     ----------
#     poa
#     temperature_cell
#     imp
#     makefigure
#
#     Returns
#     -------
#
#     """
#
#     # print('irradiance poa',poa)
#
#     cax = np.logical_and.reduce((
#         # poa > np.nanpercentile(poa, 50),
#         # imp < np.nanpercentile(imp, 95),
#         poa > 500,
#         imp > np.nanpercentile(imp, 50),
#         temperature_cell > temperature_ref - np.abs(temperature_fit_range) / 2,
#         temperature_cell < temperature_ref + np.abs(temperature_fit_range) / 2,
#     ))
#
#     if np.sum(cax) < 2:
#         return np.nan, np.nan
#
#     # cax = poa > np.nanpercentile(poa, 30)
#
#     x = temperature_cell[cax]
#     y = imp[cax] / poa[cax] * irradiance_ref
#
#     # # cax = np.logical_and(y > np.nanpercentile(y,80), y < np.nanmean(y) * 1.5)
#     # cax = y > np.nanpercentile(y,50)
#     # x = x[cax]
#     # y = y[cax]
#
#     imp_fit = np.polyfit(x, y, 1)
#     imp_ref_estimate = np.polyval(imp_fit, temperature_ref)
#     alpha_imp = imp_fit[0]
#
#     if figure:
#         plt.figure(figure_number)
#         plt.clf()
#         ax = plt.gca()
#         x_smooth = np.linspace(x.min(), x.max(), 5)
#         # plt.hist2d(x, y, bins=(25, 25))
#         plt.scatter(x, y)
#         plt.plot(x_smooth, np.polyval(imp_fit, x_smooth), 'r')
#         plt.plot(temperature_ref, np.polyval(imp_fit, temperature_ref), 'r.')
#         ax.axvline(temperature_ref - np.abs(temperature_fit_range) / 2)
#         ax.axvline(temperature_ref + np.abs(temperature_fit_range) / 2)
#         plt.xlabel('Cell temperature (C)')
#         plt.ylabel('Imp (A)')
#         plt.show()
#
#     return imp_ref_estimate, alpha_imp


def estimate_imp_ref(poa,
                     temperature_cell,
                     imp,
                     irradiance_ref=1000,
                     temperature_ref=25,
                     figure=False,
                     figure_number=20,
                     model='sandia',
                     verbose=False
                     ):
    """
    Estimate imp_ref and beta_imp using operation data. Note that typically
    imp for an array would be divided by parallel_strings before calling this
    function.


    Parameters
    ----------
    poa
    temperature_cell
    imp
    figure


    Returns
    -------

    """

    cax = np.logical_and.reduce((
        poa > 200,
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
            # line_color[0:3] =line_color[0:3]*0.9

            line_color[3] = 0.9
            plt.plot(x_smooth, imp_model(temperature_plot, x_smooth),
                     color=line_color)
            if temperature_plot == 25:
                plt.plot(irradiance_ref, imp_ref, '.',
                         color=line_color)
                # plt.plot(temperature_ref*np.ones(2), [0,vmp_ref],'k--')
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

        # plt.figure(figure_number + 1)
        # plt.clf()
        # plt.scatter(poa[cax], y,
        #             c=x,
        #             s=1,
        #             cmap='jet',
        #             vmin=0,
        #             vmax=70)
        # poa_smooth = np.linspace(1,1000,100)
        # plt.plot(poa_smooth, model(temperature=25,
        #                            logEe= np.log(poa_smooth/irradiance_ref)))
        # plt.xlabel('POA (W/m2)')
        # plt.ylabel('Vmp (V)')
        # plt.show()

    return out
#
#
# def estimate_mpp_ref_full(poa,
#                           temperature_cell,
#                           imp,
#                           vmp,
#                           photocurrent_ref,
#                           diode_factor,
#                           saturation_current_ref,
#                           cells_in_series=60,
#                           resistance_series=0.4,
#                           resistance_shunt_ref=400,
#                           EgRef=1.121,
#                           dEgdT=-0.0002677,
#                           alpha_isc=0.001,
#                           figure=False,
#                           figure_number=50,
#                           temperature_ref=25,
#                           temperature_fit_range=20,
#                           irradiance_ref=1000
#                           ):
#     # print('irradiance poa',poa)
#
#     cax = np.logical_and.reduce((
#         # poa > np.nanpercentile(poa, 50),
#         # imp < np.nanpercentile(imp, 95),
#         poa > 400,
#         # imp > np.nanpercentile(imp, 50),
#         temperature_cell > temperature_ref - np.abs(temperature_fit_range) / 2,
#         temperature_cell < temperature_ref + np.abs(temperature_fit_range) / 2,
#     ))
#
#     if np.sum(cax) < 2:
#         return np.nan, np.nan, np.nan
#
#     # cax = poa > np.nanpercentile(poa, 30)
#
#     kB = 1.381e-23
#     q = 1.602e-19
#     T = temperature_ref + 273.15
#     Vth = kB * T / q
#
#     # print('remove this!')
#     # saturation_current_ref=5e-9
#     nNsVth_ref = diode_factor * cells_in_series * Vth
#
#     IL, I0ref_to_I0, Rs, Rsh, a = calcparams_desoto(
#         I_L_ref=photocurrent_ref,
#         I_o_ref=saturation_current_ref,
#         R_sh_ref=resistance_shunt_ref,
#         R_s=resistance_series,
#         effective_irradiance=poa,
#         temp_cell=temperature_cell,
#         alpha_sc=alpha_isc,
#         a_ref=nNsVth_ref,
#         EgRef=EgRef,
#         dEgdT=dEgdT,
#         irrad_ref=irradiance_ref,
#         temp_ref=temperature_ref)
#     out = singlediode(IL, I0ref_to_I0, Rs, Rsh, a)
#
#     IL, I0ref_to_I0, Rs, Rsh, a = calcparams_desoto(
#         I_L_ref=photocurrent_ref,
#         I_o_ref=saturation_current_ref,
#         R_sh_ref=resistance_shunt_ref,
#         R_s=resistance_series,
#         effective_irradiance=irradiance_ref,
#         temp_cell=temperature_cell,
#         alpha_sc=alpha_isc,
#         a_ref=nNsVth_ref,
#         EgRef=EgRef,
#         dEgdT=dEgdT,
#         irrad_ref=irradiance_ref,
#         temp_ref=temperature_ref)
#     out_one_sun = singlediode(IL, I0ref_to_I0, Rs, Rsh, a)
#
#     IL, I0ref_to_I0, Rs, Rsh, a = calcparams_desoto(
#         I_L_ref=photocurrent_ref,
#         I_o_ref=saturation_current_ref,
#         R_sh_ref=resistance_shunt_ref,
#         R_s=resistance_series,
#         effective_irradiance=poa,
#         temp_cell=temperature_ref,
#         alpha_sc=alpha_isc,
#         a_ref=nNsVth_ref,
#         EgRef=EgRef,
#         dEgdT=dEgdT,
#         irrad_ref=irradiance_ref,
#         temp_ref=temperature_ref)
#     out_temperature_correct = singlediode(IL, I0ref_to_I0, Rs, Rsh, a)
#
#     # pmp = imp * vmp
#
#     imp_corrected = (imp * out_one_sun['i_mp'] / out['i_mp']) - out['i_mp'] + \
#                     out_temperature_correct['i_mp']
#     #
#     # print('vmp addition:')
#     # print((-out['v_mp'] + out_one_sun['v_mp']) + (
#     #             -out['v_mp'] + out_temperature_correct['v_mp']))
#     vmp_corrected = vmp + (-out['v_mp'] + out_one_sun['v_mp']) + (
#             -out['v_mp'] + out_temperature_correct['v_mp'])
#
#     imp_ref = np.mean(imp_corrected[cax])
#     vmp_ref = np.mean(vmp_corrected[cax])
#     pmp_ref = imp_ref * vmp_ref
#
#     if figure:
#         plt.figure(figure_number)
#         plt.clf()
#         plt.hist(imp_ref, bins=20)
#         plt.xlabel('Imp (A)')
#         plt.ylabel('Occurrences')
#         plt.show()
#
#         plt.figure(figure_number + 1)
#         plt.clf()
#         plt.hist(vmp_ref, bins=20)
#         plt.xlabel('Vmp (V)')
#         plt.ylabel('Occurrences')
#         plt.show()
#
#     return imp_ref, vmp_ref, pmp_ref

    #
    # imp_irradiance_corrected = imp * out_one_sun['i_mp'] / out['i_mp']
    #
    # # TODO: vmp is not proportional to irradiance.
    # vmp_irradiance_corrected = vmp - out['v_mp'] + out_one_sun['v_mp']
    # pmp_irradiance_corrected = pmp * out_one_sun['p_mp'] / out['p_mp']
    #
    # x = temperature_cell[cax]
    # y = imp_irradiance_corrected[cax]
    #
    # imp_fit = np.polyfit(x, y, 1)
    # imp_ref = np.polyval(imp_fit, temperature_ref)
    # alpha_imp = imp_fit[0]
    #
    # if figure:
    #     plt.figure(figure_number)
    #     plt.clf()
    #     ax = plt.gca()
    #     x_smooth = np.linspace(x.min(), x.max(), 5)
    #     # plt.hist2d(x, y, bins=(25, 25))
    #     plt.scatter(x, y)
    #     plt.plot(x_smooth, np.polyval(imp_fit, x_smooth), 'r')
    #     plt.plot(temperature_ref, np.polyval(imp_fit, temperature_ref), 'r.')
    #     ax.axvline(temperature_ref - np.abs(temperature_fit_range) / 2)
    #     ax.axvline(temperature_ref + np.abs(temperature_fit_range) / 2)
    #     plt.xlabel('Cell temperature (C)')
    #     plt.ylabel('Imp (A)')
    #     plt.show()
    #
    # # Vmp
    # x = temperature_cell[cax]
    # y = vmp_irradiance_corrected[cax]
    #
    # vmp_fit = np.polyfit(x, y, 1)
    # vmp_ref = np.polyval(vmp_fit, temperature_ref)
    # beta_vmp = vmp_fit[0]
    #
    # if figure:
    #     plt.figure(figure_number + 1)
    #     plt.clf()
    #     ax = plt.gca()
    #     x_smooth = np.linspace(x.min(), x.max(), 5)
    #     # plt.hist2d(x, y, bins=(25, 25))
    #     plt.scatter(x, y)
    #     plt.plot(x_smooth, np.polyval(vmp_fit, x_smooth), 'r')
    #     plt.plot(temperature_ref, np.polyval(vmp_fit, temperature_ref), 'r.')
    #     ax.axvline(temperature_ref - np.abs(temperature_fit_range) / 2)
    #     ax.axvline(temperature_ref + np.abs(temperature_fit_range) / 2)
    #     plt.xlabel('Cell temperature (C)')
    #     plt.ylabel('Vmp (A)')
    #     plt.show()
    #
    # # PMP
    # x = temperature_cell[cax]
    # y = pmp_irradiance_corrected[cax]
    #
    # pmp_fit = np.polyfit(x, y, 1)
    # pmp_ref = np.polyval(pmp_fit, temperature_ref)
    # gamma_pmp = pmp_fit[0]
    #
    # if figure:
    #     plt.figure(figure_number + 2)
    #     plt.clf()
    #     ax = plt.gca()
    #     x_smooth = np.linspace(x.min(), x.max(), 5)
    #     # plt.hist2d(x, y, bins=(25, 25))
    #     plt.scatter(x, y)
    #     plt.plot(x_smooth, np.polyval(pmp_fit, x_smooth), 'r')
    #     plt.plot(temperature_ref, np.polyval(pmp_fit, temperature_ref), 'r.')
    #     ax.axvline(temperature_ref - np.abs(temperature_fit_range) / 2)
    #     ax.axvline(temperature_ref + np.abs(temperature_fit_range) / 2)
    #     plt.xlabel('Cell temperature (C)')
    #     plt.ylabel('Pmp (A)')
    #     plt.show()
    # return imp_ref, alpha_imp, vmp_ref, beta_vmp, pmp_ref, gamma_pmp
    #


#
# def estimate_pmp_ref(poa,
#                      temperature_cell,
#                      pmp,
#                      temperature_ref=25,
#                      figure=False,
#                      figure_number=12,
#                      ):
#     pmp_ref, gamma_pmp = estimate_imp_ref(poa,
#                                           temperature_cell, pmp,
#                                           figure=figure,
#                                           figure_number=figure_number)
#     if figure:
#         plt.ylabel('pmp (W)')
#     return pmp_ref, gamma_pmp


def estimate_vmp_ref(poa,
                     temperature_cell,
                     vmp,
                     irradiance_ref=1000,
                     temperature_ref=25,
                     figure=False,
                     figure_number=21,
                     model='sandia'
                     ):
    """
    Estimate vmp_ref using operation data. Note that typically vmp for an
    array would be divided by parallel_strings before calling this function.

    vmp does not depend much on irradiance so long as the irradiance is not
    too low.

    Parameters
    ----------
    poa
    temperature_cell
    imp
    makefigure

    Returns
    -------

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
    #
    # kB = 1.381e-23
    # q = 1.602e-19
    # Vth = kB * (temperature_cell + 273.15) / q

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

    # # cax = np.logical_and(y > np.nanpercentile(y,80), y < np.nanmean(y) * 1.5)
    # cax = y > np.nanpercentile(y,50)
    # x = x[cax]
    # y = y[cax]
    #
    # vmp_fit = np.polyfit(x, y, 1)
    # vmp_ref_estimate = np.polyval(vmp_fit, 25)
    # beta_vmp = vmp_fit[0]
    #
    # print(beta_vmp)

    if model.lower() == 'sandia':
        X = np.zeros(shape=(len(temperature_cell), 4))
        X[:, 0] = 1
        X[:, 1] = temperature_cell - temperature_ref
        X[:, 2] = logEe
        X[:, 3] = logEe ** 2

        coeff = np.dot(pinv(X), vmp)

        vmp_ref = coeff[0]
        beta_vmp = coeff[1]
        coeff_irrad_1 = coeff[2]
        coeff_irrad_2 = coeff[3]

        def vmp_model(temperature, irradiance):
            return vmp_ref + beta_vmp * (temperature - temperature_ref) + \
                   coeff_irrad_1 * np.log(irradiance / irradiance_ref) + \
                   coeff_irrad_2 * np.log(irradiance / irradiance_ref) ** 2

        out = {'v_mp_ref': vmp_ref,
               'beta_vmp': beta_vmp,
               'coeff_irrad_1': coeff_irrad_1,
               'coeff_irrad_2': coeff_irrad_2,
               'vmp_model': vmp_model}

    elif model.lower() == 'temperature':
        X = np.zeros(shape=(len(temperature_cell), 2))
        X[:, 0] = 1
        X[:, 1] = temperature_cell - temperature_ref

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
            # line_color[0:3] =line_color[0:3]*0.9

            line_color[3] = 0.9
            plt.plot(x_smooth, vmp_model(x_smooth, poa_plot),
                     color=line_color)
            if poa_plot == 1000:
                plt.plot(temperature_ref, vmp_ref, '.',
                         color=line_color)
                # plt.plot(temperature_ref*np.ones(2), [0,vmp_ref],'k--')
                plt.text(temperature_ref, vmp_ref,
                         '  vmp_ref: {:.2f} V'.format(vmp_ref),
                         horizontalalignment='left')

        plt.xlabel('Cell temperature (C)')
        plt.ylabel('Vmp (V)')
        # plt.xlim([0,70])
        pcbar = plt.colorbar()
        pcbar.set_label('POA (W/m2)')

        plt.show()

        # plt.figure(figure_number + 1)
        # plt.clf()
        # plt.scatter(poa[cax], y,
        #             c=x,
        #             s=1,
        #             cmap='jet',
        #             vmin=0,
        #             vmax=70)
        # poa_smooth = np.linspace(1,1000,100)
        # plt.plot(poa_smooth, model(temperature=25,
        #                            logEe= np.log(poa_smooth/irradiance_ref)))
        # plt.xlabel('POA (W/m2)')
        # plt.ylabel('Vmp (V)')
        # plt.show()

    return out


#
# def estimate_vmp_ref(poa,
#                      temperature_cell,
#                      imp,
#                      vmp,
#                      temperature_ref=25,
#                      figure=False,
#                      figure_number=8,
#                      ):
#     """
#     Estimate vmp_ref using operation data. Note that typically vmp for an
#     array would be divided by parallel_strings before calling this function.
#
#     vmp does not depend much on irradiance so long as the irradiance is not
#     too low.
#
#     Parameters
#     ----------
#     poa
#     temperature_cell
#     imp
#     makefigure
#
#     Returns
#     -------
#
#     """
#
#     cax = np.logical_and.reduce((
#         poa > 700,
#         poa < 1100,
#         # imp > np.nanmax(imp) * 0.5
#     ))
#
#     if np.sum(cax) < 2:
#         return np.nan, np.nan
#
#     x = temperature_cell[cax]
#     y = vmp[cax]
#
#     # # cax = np.logical_and(y > np.nanpercentile(y,80), y < np.nanmean(y) * 1.5)
#     # cax = y > np.nanpercentile(y,50)
#     # x = x[cax]
#     # y = y[cax]
#
#     vmp_fit = np.polyfit(x, y, 1)
#     vmp_ref_estimate = np.polyval(vmp_fit, 25)
#     beta_vmp = vmp_fit[0]
#
#     if figure:
#         plt.figure(figure_number)
#         plt.clf()
#         #
#         # x_smooth = np.linspace(x.min(), x.max(), 5)
#         # plt.hist2d(x, y, bins=(100, 100))
#         # plt.plot(x_smooth, np.polyval(vmp_fit, x_smooth), 'r')
#         x_smooth = np.linspace(x.min(), x.max(), 5)
#         # plt.hist2d(x, y, bins=(25, 25))
#         plt.scatter(x, y)
#         plt.plot(x_smooth, np.polyval(vmp_fit, x_smooth), 'r')
#         plt.plot(temperature_ref, np.polyval(vmp_fit, temperature_ref), 'r.')
#         plt.plot(x, y, '.')
#         plt.xlabel('Cell temperature (C)')
#         plt.ylabel('Vmp (V)')
#         plt.show()
#
#     return vmp_ref_estimate, beta_vmp


def get_average_diode_factor(technology):
    diode_factor_all = {'multi-Si': 1.0229594245606348,
                        'mono-Si': 1.029537190437867,
                        'thin-film': 1.0891515236168812,
                        'cigs': 0.7197110756474172,
                        'cdte': 1.0949601271176865}

    return diode_factor_all[technology]


def estimate_diode_factor(i_mp, v_mp, photocurrent_ref,
                          saturation_current_ref,
                          temperature_cell, poa,
                          cells_in_series=60,
                          resistance_series=0.4,
                          resistance_shunt_ref=400,
                          EgRef=1.121,
                          dEgdT=-0.0002677,
                          alpha_isc=0.001,
                          temperature_ref=22,
                          irradiance_ref=1000,
                          figure=False,
                          figure_number=28):
    cax = np.logical_and.reduce((
        i_mp * v_mp > np.nanpercentile(i_mp * v_mp, 1),
        np.isfinite(i_mp),
        np.isfinite(i_mp),
        np.isfinite(poa),
        np.isfinite(temperature_cell),
    ))

    i_mp = i_mp[cax]
    v_mp = v_mp[cax]
    temperature_cell = temperature_cell[cax]
    poa = poa[cax]

    kB = 1.381e-23
    q = 1.602e-19
    Tcell_K = temperature_cell + 273.15
    Tref_K = temperature_ref + 273.15

    IL, I0ref_to_I0, Rs, Rsh, aref_to_a = calcparams_desoto(
        I_L_ref=photocurrent_ref,
        I_o_ref=1,
        R_sh_ref=resistance_shunt_ref,
        R_s=resistance_series,
        effective_irradiance=poa,
        temp_cell=temperature_cell,
        alpha_sc=alpha_isc,
        a_ref=1,
        EgRef=EgRef,
        dEgdT=dEgdT,
        irrad_ref=irradiance_ref,
        temp_ref=temperature_ref)

    I0 = saturation_current_ref * I0ref_to_I0

    log1p_arg = -1 / I0 * (i_mp - IL + (v_mp + i_mp * Rs) / Rsh)
    log1p_arg[log1p_arg<-0.999] = -0.999
    nNsVth = (v_mp + i_mp * Rs) / np.log1p(log1p_arg)

    # nNsVth_ref = a_to_aref * nNsVth
    nNsVth_ref = 1 / aref_to_a * nNsVth

    Vth = kB * (Tref_K) / q
    diode_factor = nNsVth_ref / cells_in_series / Vth

    diode_factor_mean = np.mean(diode_factor)

    if figure:
        plt.figure(figure_number)
        plt.clf()
        ax = plt.gca()
        bin_min = 0.5
        bin_max = 2

        bins = np.linspace(bin_min, bin_max, 100)
        plt.hist(diode_factor, bins=bins)
        ax.axvline(diode_factor_mean, color='r', ymax=0.5)
        plt.ylabel('Occurrences')
        plt.xlabel('diode_factor')

    return diode_factor_mean


def estimate_photocurrent_ref_simple(imp_ref, technology='mono-Si'):
    photocurrent_imp_ratio = {'multi-Si': 1.0746167586063207,
                              'mono-Si': 1.0723051517913444,
                              'thin-film': 1.1813401654607967,
                              'cigs': 1.1706462692015707,
                              'cdte': 1.1015249105470803}

    photocurrent_ref = imp_ref * photocurrent_imp_ratio[technology]

    return photocurrent_ref

#
# def estimate_saturation_current_full(imp_ref,
#                                      photocurrent_ref,
#                                      vmp_ref,
#                                      cells_in_series,
#                                      resistance_series_ref=0.4,
#                                      resistance_shunt_ref=100,
#                                      diode_factor=1.2,
#                                      temperature_ref=25,
#                                      figure_number=23,
#                                      ):
#     """
#     If vmp or cells_in_series is unknown, use vmp_ref=0.6 and cells_in_series=1
#
#     Parameters
#     ----------
#     imp_ref
#     photocurrent_ref
#     vmp_ref
#     cells_in_series
#     resistance_series_ref
#     resistance_shunt_ref
#
#     Returns
#     -------
#
#     """
#     kB = 1.381e-23
#     q = 1.602e-19
#     T = temperature_ref + 273.15
#     Vth = kB * T / q
#     # voc_cell = 0.6
#
#     Rsh = resistance_shunt_ref
#     Rs = resistance_series_ref
#
#     nNsVth = diode_factor * cells_in_series * Vth
#
#     saturation_current_ref_estimate = (photocurrent_ref - imp_ref - (
#             vmp_ref + imp_ref * Rs) / Rsh) / np.exp(
#         (vmp_ref + imp_ref * Rs) / nNsVth)
#
#     return saturation_current_ref_estimate


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
                                    diode_factor=1.10,
                                    cells_in_series=60,
                                    temperature_ref=25,
                                    irradiance_ref=1000,
                                    resistance_series=0.4,
                                    resistance_shunt_ref=400,
                                    EgRef=1.121,
                                    dEgdT=-0.0002677,
                                    alpha_isc=0.001,
                                    figure=False,
                                    figure_number=24):
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

    # plt.figure(15)
    # plt.clf()
    # plt.hist2d(temperature_cell, np.log10(I0_ref/1e-9),bins=100)
    # plt.ylabel('log10(I0/nA)')
    # plt.xlabel('Temperature Cell (C)')

    return I0_ref.mean()


def estimate_photocurrent_ref(current, voltage, saturation_current_ref,
                              temperature_cell, poa,
                              diode_factor=1.10,
                              cells_in_series=60,
                              temperature_ref=25,
                              irradiance_ref=1000,
                              resistance_series=0.4,
                              resistance_shunt_ref=400,
                              EgRef=1.121,
                              dEgdT=-0.0002677,
                              alpha_isc_norm=0.001 / 6,
                              figure=False,
                              figure_number=25):
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

    # plt.figure(15)
    # plt.clf()
    # plt.hist2d(temperature_cell, np.log10(I0_ref/1e-9),bins=100)
    # plt.ylabel('log10(I0/nA)')
    # plt.xlabel('Temperature Cell (C)')

    return IL_ref_mean


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


def estimate_resistance_series_simple(vmp, imp,
                                      saturation_current,
                                      photocurrent,
                                      nNsVth):
    Rs = (nNsVth * np.log1p(
        (photocurrent - imp) / saturation_current) - vmp) / imp
    return Rs


def estimate_resistance_series(poa,
                               temperature_cell,
                               voltage,
                               current,
                               photocurrent_ref,
                               saturation_current_ref,
                               diode_factor,
                               cells_in_series=60,
                               temperature_ref=25,
                               irradiance_ref=1000,
                               resistance_shunt_ref=400,
                               EgRef=1.121,
                               dEgdT=-0.0002677,
                               alpha_isc=0.001,
                               figure=False,
                               figure_number=26,
                               verbose=False
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
    #

    # imp = current
    # vmp = voltage
    # a = nNsVth
    # Isat = I0
    #
    # Rs = (a / 2 + (IL * Rsh) / 2 - (imp * Rsh) / 2 + (Isat * Rsh) / 2 - (
    #             a * np.sqrt((4 * (IL * Rsh + Isat * Rsh - 2 * vmp)) / a + (
    #                 1 - (IL * Rsh) / a + (imp * Rsh) / a - (Isat * Rsh) / a + (
    #                     2 * vmp) / a) ** 2)) / 2) / imp

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
        #
        # plt.figure(20)
        # plt.clf()
        # plt.hist2d(poa,Rs,bins=100)
        #
        # plt.figure(21)
        # plt.clf()
        # plt.hist2d(temperature_cell, Rs, bins=100)

    return Rs_mean


# def estimate_shunt_resistance_ref():


def estimate_singlediode_params(poa,
                                temperature_module,
                                vmp,
                                imp,
                                delta_T=3,
                                band_gap_ref=1.121,
                                dEgdT=-0.0002677,
                                alpha_isc=None,
                                cells_in_series=None,
                                technology='mono-Si',
                                convergence_test=0.0001,
                                temperature_ref=25,
                                irradiance_ref=1000,
                                resistance_series_ref=None,
                                resistance_shunt_ref=400,
                                figure=False,
                                figure_number_start=20,
                                imp_model='sandia',
                                verbose=False,
                                max_iter=10,
                                optimize_Rs_Io=False
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
    temperature_module = np.array(temperature_module)
    vmp = np.array(vmp)
    imp = np.array(imp)
    figure_number = figure_number_start

    temperature_cell = sapm_cell_from_module(
        module_temperature=temperature_module,
        poa_global=poa,
        deltaT=delta_T,
        irrad_ref=irradiance_ref)

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
    figure_number+=1

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
        figure_number=figure_number)
    figure_number += 1

    vmp_ref = out['v_mp_ref']
    beta_vmp = out['beta_vmp']

    pmp_ref = vmp_ref * imp_ref

    # pmp_ref, gamma_pmp = estimate_pmp_ref(poa, temperature_cell,
    #                                       imp * vmp,
    #                                       figure=figure)

    diode_factor = get_average_diode_factor(technology)

    voc_ref = estimate_voc_ref(vmp_ref, technology=technology)

    if cells_in_series == None:
        cells_in_series = estimate_cells_in_series(voc_ref,
                                                   technology=technology)
    kB = 1.381e-23
    q = 1.602e-19

    nNsVth_ref = diode_factor * cells_in_series * kB * (
            temperature_ref + 273.15) / q

    beta_voc = estimate_beta_voc(beta_vmp, technology=technology)


    photocurrent_ref = estimate_photocurrent_ref_simple(imp_ref,
                                                        technology=technology)

    # saturation_current_ref = estimate_saturation_current_full(
    #     imp_ref=imp_ref, photocurrent_ref=photocurrent_ref, vmp_ref=vmp_ref,
    #     resistance_series_ref=0.4,
    #     cells_in_series=cells_in_series)

    isc_ref = estimate_isc_ref(imp_ref, technology=technology)
    if verbose:
        print('Simple isc_ref estimate: {}'.format(isc_ref))

    saturation_current_ref = estimate_saturation_current(isc=isc_ref,
                                                         voc=voc_ref,
                                                         nNsVth=nNsVth_ref,
                                                         )
    if verbose:
        print('Simple saturation current ref estimate: {}'.format(saturation_current_ref))

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
        print('resistance_series_ref estimate: {}'.format(resistance_series_ref))


    num_iter = max_iter

    # More complex optimization not working so well on real data.
    if optimize_Rs_Io:
        results = pd.DataFrame(columns=['saturation_current_ref'])
        results.loc[0, 'saturation_current_ref'] = saturation_current_ref
        results.loc[0, 'diode_factor'] = diode_factor
        results.loc[0, 'resistance_series_ref'] = resistance_series_ref
        results.loc[0, 'photocurrent_ref'] = photocurrent_ref

        # print('Time to point a: {}'.format(time() - start_time))
        last_iteration = False
        for k in range(num_iter):
            # if verbose:
                # print(k)

            draw_figure_this_iteration = np.logical_and(figure, last_iteration)

            # Update saturation current with better estimate.
            saturation_current_ref = estimate_saturation_current_ref(
                i_mp=imp,
                v_mp=vmp,
                photocurrent_ref=photocurrent_ref,
                temperature_cell=temperature_cell,
                diode_factor=diode_factor,
                poa=poa,
                cells_in_series=cells_in_series,
                resistance_series=resistance_series_ref,
                resistance_shunt_ref=resistance_shunt_ref,
                EgRef=band_gap_ref,
                dEgdT=dEgdT,
                alpha_isc=alpha_isc,
                figure=draw_figure_this_iteration,
                figure_number=figure_number)

            if draw_figure_this_iteration:
                figure_number+=1

            if verbose:
                print('Inputs')
                print(poa)
                print(temperature_cell)
                print(photocurrent_ref)
                print('saturation_current_ref:', saturation_current_ref)
                print('ndiode: ', diode_factor)

            # Important to have a good estimate of saturation current first.
            resistance_series_ref = estimate_resistance_series(
                poa=poa,
                temperature_cell=temperature_cell,
                voltage=vmp,
                current=imp,
                photocurrent_ref=photocurrent_ref,
                saturation_current_ref=saturation_current_ref,
                diode_factor=diode_factor,
                cells_in_series=cells_in_series,
                temperature_ref=temperature_ref,
                irradiance_ref=irradiance_ref,
                resistance_shunt_ref=resistance_shunt_ref,
                EgRef=band_gap_ref,
                dEgdT=dEgdT,
                alpha_isc=alpha_isc,
                figure=draw_figure_this_iteration,
                figure_number=figure_number
            )
            if draw_figure_this_iteration:
                figure_number+=1

            results.loc[k + 1, 'saturation_current_ref'] = saturation_current_ref
            results.loc[k + 1, 'resistance_series_ref'] = resistance_series_ref
            results.loc[k + 1, 'diode_factor'] = diode_factor
            results.loc[k + 1, 'photocurrent_ref'] = photocurrent_ref

            if verbose:
                print('resistance_series_ref: {}'.format(resistance_series_ref))
                print('saturation_current_ref: {}'.format(saturation_current_ref))
            if draw_figure_this_iteration:
                break

            if np.abs(results.loc[k + 1, 'resistance_series_ref'] - \
                      results.loc[k, 'resistance_series_ref']) / results.loc[
                k, 'resistance_series_ref'] < convergence_test:
                if verbose:
                    print('optimization stopped at iteration {}'.format(k))

                #     Draw figures then break loop
                last_iteration=True
    else:
        results = {}

        # print('Time to point b: {}'.format(time() - start_time))

        # Update diode factor
        diode_factor = estimate_diode_factor(
            i_mp=imp, v_mp=vmp,
            photocurrent_ref=photocurrent_ref,
            saturation_current_ref=saturation_current_ref,
            temperature_cell=temperature_cell,
            poa=poa,
            cells_in_series=cells_in_series,
            resistance_series=resistance_series_ref,
            resistance_shunt_ref=resistance_shunt_ref,
            EgRef=band_gap_ref,
            dEgdT=dEgdT,
            alpha_isc=alpha_isc,
            temperature_ref=temperature_ref,
            irradiance_ref=irradiance_ref,
            figure=figure,
            figure_number=figure_number)
        if np.isnan(diode_factor):
            print('TODO: fix diode factor nan reason')
            diode_factor = get_average_diode_factor(technology)
        # print('Diode factor: {:.3f}'.format(diode_factor))
        figure_number+=1


    # print('Time to point c: {}'.format(time() - start_time))

    #
    # for k in range(num_iter):
    #     if verbose:
    #         print(k)
    #
    #     draw_figure = np.logical_and(figure,k==num_iter-1)
    #
    #     photocurrent_ref = estimate_photocurrent_ref(
    #         current=imp, voltage=vmp,
    #         temperature_cell=temperature_cell, poa=poa,
    #         saturation_current_ref=saturation_current_ref,
    #         diode_factor=diode_factor,
    #         cells_in_series=cells_in_series,
    #         temperature_ref=temperature_ref,
    #         irradiance_ref=irradiance_ref,
    #         resistance_series=resistance_series_ref,
    #         resistance_shunt_ref=resistance_shunt_ref,
    #         EgRef=band_gap_ref,
    #         dEgdT=dEgdT,
    #         alpha_isc_norm=alpha_isc/photocurrent_ref,
    #         figure=draw_figure)
    #
    #     # # Update diode factor
    #     # diode_factor = estimate_diode_factor(
    #     #     i_mp=imp, v_mp=vmp,
    #     #     photocurrent_ref=photocurrent_ref,
    #     #     saturation_current_ref=saturation_current_ref,
    #     #     temperature_cell=temperature_cell,
    #     #     poa=poa,
    #     #     cells_in_series=cells_in_series,
    #     #     resistance_series=resistance_series_ref,
    #     #     resistance_shunt_ref=resistance_shunt_ref,
    #     #     EgRef=band_gap_ref,
    #     #     dEgdT=dEgdT,
    #     #     alpha_isc=alpha_isc,
    #     #     temperature_ref=temperature_ref,
    #     #     irradiance_ref=irradiance_ref,
    #     #     figure=draw_figure)
    #
    #     # saturation_current_ref = estimate_saturation_current_ref(
    #     #     i_mp=imp,
    #     #     v_mp=vmp,
    #     #     photocurrent_ref=photocurrent_ref,
    #     #     temperature_cell=temperature_cell,
    #     #     diode_factor=diode_factor,
    #     #     poa=poa,
    #     #     cells_in_series=cells_in_series,
    #     #     resistance_series=resistance_series_ref,
    #     #     resistance_shunt_ref=resistance_shunt_ref,
    #     #     EgRef=band_gap_ref,
    #     #     dEgdT=dEgdT,
    #     #     alpha_isc=alpha_isc,
    #     #     figure=draw_figure)
    #
    #
    #
    #     # Important to have a good estimate of saturation current first.
    #     resistance_series_ref = estimate_resistance_series(
    #         poa=poa,
    #         temperature_cell=temperature_cell,
    #         voltage=vmp,
    #         current=imp,
    #         photocurrent_ref=photocurrent_ref,
    #         saturation_current_ref=saturation_current_ref,
    #         diode_factor=diode_factor,
    #         cells_in_series=cells_in_series,
    #         temperature_ref=temperature_ref,
    #         irradiance_ref=irradiance_ref,
    #         resistance_shunt_ref=resistance_shunt_ref,
    #         EgRef=band_gap_ref,
    #         dEgdT=dEgdT,
    #         alpha_isc=alpha_isc,
    #         figure=draw_figure,
    #     )
    #
    #     results.loc[k + 1, 'diode_factor'] = diode_factor
    #     results.loc[k + 1, 'photocurrent_ref'] = photocurrent_ref
    #

    #
    # print('pmp_ref before update: {:.2f}'.format(pmp_ref))
    # imp_ref, vmp_ref, pmp_ref = estimate_mpp_ref_full(
    #     poa,
    #     temperature_cell,
    #     imp,
    #     vmp,
    #     photocurrent_ref,
    #     diode_factor,
    #     saturation_current_ref,
    #     cells_in_series=cells_in_series,
    #     resistance_series=resistance_series_ref,
    #     resistance_shunt_ref=resistance_shunt_ref,
    #     EgRef=band_gap_ref,
    #     dEgdT=dEgdT,
    #     alpha_isc=alpha_isc,
    #     figure=figure,
    #     temperature_ref=temperature_ref,
    #     temperature_fit_range=temperature_fit_range,
    #     irradiance_ref=irradiance_ref
    # )
    # print('pmp_ref after update: {:.2f}'.format(pmp_ref))

    # TODO: add gamma_pmp

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
