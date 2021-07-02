import numpy as np
import pandas as pd

import scipy

from pvpro.singlediode import pvlib_single_diode, pv_system_single_diode_model, \
    singlediode_closest_point

from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression

from functools import partial

def _fit_singlediode_linear(voltage, current, temperature_cell, poa,
                            resistance_series, diode_factor, cells_in_series,
                            alpha_sc,
                            photocurrent_ref_min=0,
                            photocurrent_ref_max=100,
                            saturation_current_ref_min=1e-15,
                            saturation_current_ref_max=1e-3,
                            resistance_shunt_ref_min=1e-5,
                            resistance_shunt_ref_max=1e5,
                            Eg_ref=1.121, dEgdT=-0.0002677,
                            temperature_ref=25, irrad_ref=1000,
                            solver='lsq_linear',
                            model='pvpro',
                            tol=1e-8,
                            verbose=False):
    """
    Fit a set of voltage, current, temperature_cell and poa values to the
    Desoto single diode model at fixed values of series resistance and diode
    factor.



    Parameters
    ----------
    voltage : numeric

        Array of DC voltage values, all inputs: 'voltage', 'current',
        'temperature_cell' and 'poa' must be the same length.

    current : numeric

        Array of DC current values, all inputs: 'voltage', 'current',
        'temperature_cell' and 'poa' must be the same length.

    temperature_cell : numeric

        Array of cell temperature values, all inputs: 'voltage', 'current',
        'temperature_cell' and 'poa' must be the same length.

    poa : numeric

        Array of effective irradiance value in plane-of-array, all inputs:
        'voltage', 'current', 'temperature_cell' and 'poa' must be the same
        length.

    resistance_series : float

        Fixed series resistance to perform the linear fit, Ohms.

    diode_factor : float

        Diode ideality factor at which to perform the linear fit, unitless.

    cells_in_series : int

        cells in series for module, array.

    alpha_sc : float

        Temperature coefficient of short circuit current, A/C.

    photocurrent_ref_max : float, default=100

        Maximum fit value of photocurrent_ref, A.

    saturation_current_ref_max : float, default=1e-3

        Maximum fit value of saturaction_current_ref, A.

    resistance_shunt_ref_min : float, default=1e-5

        Minimum fit value of resistance_shunt_ref

    resistance_shunt_ref_max : float, default=1e5

        Minimum fit value of resistance_shunt_ref

    Eg_ref : float, default=1.121

        Reference band gap, eV

    dEgdT : float, default=-0.0002677

        Band gap temperature coefficient, eV/C.

    temperature_ref : float, default=25

        Temperature at reference conditions, C

    irrad_ref : float, default=1000

        Irradiance at reference conditions, W/m^2.

    solver : str, default='lsq_linear'

        Solver for linear fit. Can be 'lsq_linear' or 'pinv'

    model : str

        Model can be 'desoto', 'pvpro'

        'desoto' is the Desoto single diode model

        'pvpro' is the Desoto single diode model with a constant shunt
        resistance.


    tol : float, defualt=1e-8

        Tolerence for solver 'lsq_linear'

    Returns
    -------
    dict with keys:

        'photocurrent_ref' : float

            Reference photocurrent, A.

        'saturation_current_ref' : float

            Saturation current at reference conditions, A.

        'resistance_shunt_ref' : float

            Shunt resistance at reference conditions, Ohms.

        'resistance_series_ref': float

            Series resistance at reference conditions, same as provided as
            input, Ohms.

        'diode_factor' : float

            Diode factor, same as provided as input.

        'nNsVth_ref': float

            diode_factor * cells_in_series * k * Tref_K,

        'loss' : loss

            Mean absolute error between fit and data.

        'solution' : dict

            Output of solver.


    """
    Tcell_K = temperature_cell + 273.15
    Tref_K = temperature_ref + 273.15

    # Boltzmann constant in eV/K
    k = 8.617332478e-05

    Eg = Eg_ref * (1 + dEgdT * (Tcell_K - Tref_K))

    nNsVth = diode_factor * cells_in_series * k * Tcell_K

    sat_current_multiplier = (((Tcell_K / Tref_K) ** 3) *
                              (np.exp(Eg_ref / (k * (Tref_K)) - (
                                      Eg / (k * (Tcell_K))))))

    scale_photocurrent = 1e-1
    scale_sat_current = 1e-9
    scale_shunt_conductance = 1e-2

    A = np.zeros(shape=(len(voltage), 3))
    A[:, 0] = scale_photocurrent * poa / irrad_ref
    A[:, 1] = -1 * scale_sat_current * sat_current_multiplier * (
            np.exp((voltage + current * resistance_series) / nNsVth) - 1)
    if model.lower() == 'desoto':
        A[:, 2] = -1 * scale_shunt_conductance * (poa / irrad_ref) * (
                voltage + current * resistance_series)
    elif model.lower() == 'pvpro':
        A[:, 2] = -1 * scale_shunt_conductance * (
                voltage + current * resistance_series)

    y = current - poa / irrad_ref * alpha_sc * (
            temperature_cell - temperature_ref)

    bounds = ([photocurrent_ref_min / scale_photocurrent,
               saturation_current_ref_min / scale_sat_current,
               1 / (resistance_shunt_ref_max * scale_shunt_conductance)],
              [photocurrent_ref_max / scale_photocurrent,
               saturation_current_ref_max / scale_sat_current,
               1 / (resistance_shunt_ref_min * scale_shunt_conductance)])

    # Solve the problem
    if solver == 'lsq_linear':
        soln = lsq_linear(A, y,
                          bounds=bounds,
                          method='trf',
                          tol=tol,
                          lsq_solver='lsmr')
        coeff = soln['x']

    elif solver == 'pinv':
        soln = {'solver': 'pinv'}
        coeff = np.dot(np.linalg.pinv(A), y)
    elif solver == 'lstsq':
        soln = {'solver': 'lstsq'}
        coeff, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        # print(m)
        # print(coeff)
    elif solver.lower() == 'linearregression':
        soln = {'solver': 'linearregression'}
        reg = LinearRegression(fit_intercept=False, copy_X=True).fit(A, y)

        coeff = reg.coef_

    # elif solver == 'cvxpy':
    #     # NOT WORKING YET!
    #
    #     # Construct the problem.
    #     x = cp.Variable(3)
    #     objective = cp.Minimize(cp.sum_squares(A @ x - y))
    #     constraints = [0 <= x, x <= 1e5]
    #     prob = cp.Problem(objective, constraints)
    #
    #     print("Optimal value", prob.solve())
    #     print("Optimal var")
    #     print(x.value)  # A numpy ndarray.
    #
    #     soln = {}
    #     coeff = x.value

    # loss = np.mean(np.abs(np.dot(A, coeff) - y))
    loss = np.mean((np.dot(A, coeff) - y) ** 2)
    # TODO: try l2 norm as well, may work better for data without large outliers.

    out = {
        'photocurrent_ref': coeff[0] * scale_photocurrent,
        'saturation_current_ref': coeff[1] * scale_sat_current,
        'resistance_shunt_ref': 1 / (coeff[2] * scale_shunt_conductance),
        'resistance_series_ref': resistance_series,
        'diode_factor': diode_factor,
        'nNsVth_ref': diode_factor * cells_in_series * k * Tref_K,
        'loss': loss,
        'solution': soln,
    }

    if verbose:
        print('* Linear problem solved')
        for k in ['photocurrent_ref', 'saturation_current_ref',
                  'resistance_shunt_ref', 'resistance_series_ref',
                  'diode_factor', 'nNsVth_ref', 'loss']:
            print('{}: {}'.format(k, out[k]))

    return out

def _fit_singlediode_linear_power(voltage, current, temperature_cell, poa,
                            resistance_series, diode_factor, cells_in_series,
                            alpha_sc,
                            photocurrent_ref_min=0,
                            photocurrent_ref_max=100,
                            saturation_current_ref_min=1e-15,
                            saturation_current_ref_max=1e-3,
                            resistance_shunt_ref_min=1e-5,
                            resistance_shunt_ref_max=1e5,
                            Eg_ref=1.121, dEgdT=-0.0002677,
                            temperature_ref=25, irrad_ref=1000,
                            solver='lsq_linear',
                            model='pvpro',
                            tol=1e-8,
                            verbose=False):

    Tcell_K = temperature_cell + 273.15
    Tref_K = temperature_ref + 273.15

    # Boltzmann constant in eV/K
    k = 8.617332478e-05

    Eg = Eg_ref * (1 + dEgdT * (Tcell_K - Tref_K))

    nNsVth = diode_factor * cells_in_series * k * Tcell_K

    sat_current_multiplier = (((Tcell_K / Tref_K) ** 3) *
                              (np.exp(Eg_ref / (k * (Tref_K)) - (
                                      Eg / (k * (Tcell_K))))))

    scale_photocurrent = 1e-1
    scale_sat_current = 1e-9
    scale_shunt_conductance = 1e-2


    # weights = 1 - voltage/np.max(voltage)
    # weights = 1 - 0.75*voltage/np.max(voltage)
    weights = current/np.max(current) + 0.2

    # weights = np.ones_like(voltage)

    A = np.zeros(shape=(len(voltage), 3))
    A[:, 0] = scale_photocurrent * poa / irrad_ref * weights
    A[:, 1] = -1 * scale_sat_current * sat_current_multiplier * (
            np.exp((voltage + current * resistance_series) / nNsVth) - 1)  * weights
    if model.lower() == 'desoto':
        A[:, 2] = -1 * scale_shunt_conductance * (poa / irrad_ref) * (
                voltage + current * resistance_series) * weights
    elif model.lower() == 'pvpro':
        A[:, 2] = -1 * scale_shunt_conductance * (
                voltage + current * resistance_series) * weights

    y = weights*(current - poa / irrad_ref * alpha_sc * (
            temperature_cell - temperature_ref))

    bounds = ([photocurrent_ref_min / scale_photocurrent,
               saturation_current_ref_min / scale_sat_current,
               1 / (resistance_shunt_ref_max * scale_shunt_conductance)],
              [photocurrent_ref_max / scale_photocurrent,
               saturation_current_ref_max / scale_sat_current,
               1 / (resistance_shunt_ref_min * scale_shunt_conductance)])

    # Solve the problem
    if solver == 'lsq_linear':
        soln = lsq_linear(A, y,
                          bounds=bounds,
                          method='trf',
                          tol=tol,
                          lsq_solver='lsmr')
        coeff = soln['x']

    elif solver == 'pinv':
        soln = {'solver': 'pinv'}
        coeff = np.dot(np.linalg.pinv(A), y)
    elif solver == 'lstsq':
        soln = {'solver': 'lstsq'}
        coeff, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        # print(m)
        # print(coeff)
    elif solver.lower() == 'linearregression':
        soln = {'solver': 'linearregression'}
        reg = LinearRegression(fit_intercept=False, copy_X=True).fit(A, y)

        coeff = reg.coef_

    # loss = np.mean(np.abs(np.dot(A, coeff) - y))
    loss = np.mean((np.dot(A, coeff) - y) ** 2)

    out = {
        'photocurrent_ref': coeff[0] * scale_photocurrent,
        'saturation_current_ref': coeff[1] * scale_sat_current,
        'resistance_shunt_ref': 1 / (coeff[2] * scale_shunt_conductance),
        'resistance_series_ref': resistance_series,
        'diode_factor': diode_factor,
        'nNsVth_ref': diode_factor * cells_in_series * k * Tref_K,
        'loss': loss,
        'solution': soln,
    }

    if verbose:
        print('* Linear problem solved')
        for k in ['photocurrent_ref', 'saturation_current_ref',
                  'resistance_shunt_ref', 'resistance_series_ref',
                  'diode_factor', 'nNsVth_ref', 'loss']:
            print('{}: {}'.format(k, out[k]))

    return out



def fit_singlediode_model(voltage, current, temperature_cell, poa,
                          cells_in_series,
                          alpha_sc,
                          resistance_series_start=0.35,
                          diode_factor_start=1.0,
                          resistance_series_min=0,
                          resistance_series_max=2,
                          diode_factor_min=0.5,
                          diode_factor_max=1.5,
                          photocurrent_ref_min=0,
                          photocurrent_ref_max=100,
                          saturation_current_ref_min=1e-15,
                          saturation_current_ref_max=1e-3,
                          resistance_shunt_ref_min=1e-5,
                          resistance_shunt_ref_max=1e5,
                          tol=1e-8,
                          Eg_ref=1.121, dEgdT=-0.0002677,
                          temperature_ref=25, irrad_ref=1000,
                          model='desoto',
                          linear_solver='lsq_linear',
                          nonlinear_solver='L-BFGS-B',
                          verbose=False,
                          ):
    """
    Fit a set of voltage, current, temperature_cell and poa values to the
    Desoto single diode model.

    This method is motivated by the fact that the single diode model can be
    turned into a linear problem if the series resistance and diode factor
    are known. This method iteratively solves this linear problem, minimizing
    the mean absolute error by adjusting series resistance and diode factor.
    The end result is to fit the Desoto single diode model to the data.

    Parameters
    ----------
    voltage : numeric

        Array of DC voltage values, all inputs: 'voltage', 'current',
        'temperature_cell' and 'poa' must be the same length.

    current : numeric

        Array of DC current values, all inputs: 'voltage', 'current',
        'temperature_cell' and 'poa' must be the same length.

    temperature_cell : numeric

        Array of cell temperature values, all inputs: 'voltage', 'current',
        'temperature_cell' and 'poa' must be the same length.

    poa : numeric

        Array of effective irradiance value in plane-of-array, all inputs:
        'voltage', 'current', 'temperature_cell' and 'poa' must be the same
        length.

    resistance_series : float

        Fixed series resistance to perform the linear fit, Ohms.

    diode_factor : float

        Diode ideality factor at which to perform the linear fit, unitless.

    cells_in_series : int

        cells in series for module, array.

    alpha_sc : float, default=0.35

        Temperature coefficient of short circuit current, A/C.

    resistance_series_start : float

        Startpoint for series resistance for solver, Ohms.

    diode_factor_start : float, default=1.0

        Startpoint for diode factor for solver, Ohms.

    resistance_series_min : float, default=0

        Lower bound for series resistance, Ohms.

    resistance_series_max : float, default=2

        Upper bound for series resitance, Ohms

    diode_factor_min : float, default=0.5

        Lower bound for diode factor.

    diode_factor_max : float, default=1.5

        Upper bound for diode factor.

    Eg_ref : float, default=1.121

        Reference band gap, eV

    dEgdT : float, default=-0.0002677

        Band gap temperature coefficient, eV/C.

    temperature_ref : float, default=25

        Temperature at reference conditions, C

    irrad_ref : float, default=1000

        Irradiance at reference conditions, W/m^2.

    solver : str, default='lsq_linear'

        Solver for linear fit. Can be 'lsq_linear' or 'pinv'

    tol : float, defualt=1e-8

        Tolerence for solver 'lsq_linear'


    Returns
    -------
    dict of best fit with keys:

        'photocurrent_ref' : float

            Reference photocurrent, A.

        'saturation_current_ref' : float

            Saturation current at reference conditions, A.

        'resistance_shunt_ref' : float

            Shunt resistance at reference conditions, Ohms.

        'resistance_series_ref': float

            Series resistance at reference conditions, Ohms.

        'diode_factor' : float

            Diode factor.

        'nNsVth_ref': float

            diode_factor * cells_in_series * k * Tref_K,

        'loss' : loss

            Mean absolute error between fit and data.

        'solution' : dict

            Output of linear solver at optimal values of series resistance
            and diode factor.

        'optimization' : dict

            Output of scipy.optimize.minimize.
    """
    if len(voltage) == 0:
        raise Exception('No values given.')
    # Remove nans

    isfinite = np.logical_and.reduce((
        np.isfinite(voltage),
        np.isfinite(current),
        np.isfinite(temperature_cell),
        np.isfinite(poa),
    ))

    voltage = voltage[isfinite]
    current = current[isfinite]
    temperature_cell = temperature_cell[isfinite]
    poa = poa[isfinite]

    bounds = [(resistance_series_min, resistance_series_max),
              (diode_factor_min, diode_factor_max)]
    x0 = np.array([resistance_series_start, diode_factor_start])

    args = dict(
        voltage=voltage,
        current=current,
        temperature_cell=temperature_cell,
        poa=poa,
        cells_in_series=cells_in_series,
        alpha_sc=alpha_sc,
        Eg_ref=Eg_ref,
        dEgdT=dEgdT,
        temperature_ref=temperature_ref,
        irrad_ref=irrad_ref, solver=linear_solver,
        model=model,
        tol=tol,
        verbose=verbose,
        photocurrent_ref_min=photocurrent_ref_min,
        photocurrent_ref_max=photocurrent_ref_max,
        saturation_current_ref_min=saturation_current_ref_min,
        saturation_current_ref_max=saturation_current_ref_max,
        resistance_shunt_ref_min=resistance_shunt_ref_min,
        resistance_shunt_ref_max=resistance_shunt_ref_max,
    )

    fsl = partial(_fit_singlediode_linear_power,**args)

    def residual(x):
        # x is list: [resistance_series, diode factor]
        out = fsl(resistance_series=x[0],diode_factor=x[1])
        return out['loss']

    ret = minimize(residual,
                   x0=x0,
                   bounds=bounds,
                   method=nonlinear_solver,
                   # verbose=2
                   )

    # Get values at best fit.
    out = fsl(resistance_series=ret['x'][0], diode_factor=ret['x'][1])
    out['optimziation'] = ret

    return out


def fit_singlediode_brute(voltage, current, temperature_cell, poa,
                          cells_in_series,
                          alpha_sc, Eg_ref=1.121, dEgdT=-0.0002677,
                          temperature_ref=25, irrad_ref=1000,
                          resistance_series_list=None,
                          diode_factor_list=None):
    if resistance_series_list is None:
        resistance_series_list = np.linspace(0.1, 0.9, 20)
    if diode_factor_list is None:
        diode_factor_list = np.linspace(0.9, 1.6, 21)

    s = (len(diode_factor_list), len(resistance_series_list))

    loss = np.zeros(s)
    diode_factor = np.zeros(s)
    resistance_series = np.zeros(s)

    loss_minimal = np.inf
    for j in range(len(diode_factor_list)):
        for k in range(len(resistance_series_list)):
            out = _fit_singlediode_linear(voltage, current, temperature_cell,
                                          poa,
                                          resistance_series_list[k],
                                          diode_factor_list[j], cells_in_series,
                                          alpha_sc, Eg_ref=Eg_ref, dEgdT=dEgdT,
                                          temperature_ref=temperature_ref,
                                          irrad_ref=irrad_ref)
            loss[j, k] = out['loss']
            diode_factor[j, k] = diode_factor_list[j]
            resistance_series[j, k] = resistance_series_list[k]

            if out['loss'] < loss_minimal:
                loss_minimal = out['loss']
                out_minimal = out.copy()
    min_pos = np.argmin(loss)

    out = {'photocurrent_ref': out_minimal['photocurrent_ref'],
           'saturation_current_ref': out_minimal['saturation_current_ref'],
           'resistance_shunt_ref': out_minimal['resistance_shunt_ref'],
           'resistance_series_ref': out_minimal['resistance_series_ref'],
           'diode_factor': out_minimal['diode_factor'],
           'nNsVth_ref': out_minimal['nNsVth_ref'],
           'resistance_series': resistance_series.flatten()[min_pos],
           'diode_factor': diode_factor.flatten()[min_pos],
           'optimization': {'loss': loss,
                            'diode_factor': diode_factor,
                            'diode_factor_list': diode_factor_list,
                            'resistance_series': resistance_series,
                            'resistance_series_list': resistance_series_list}}
    return out


def _x_to_p(x, key):
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
    elif key == 'nNsVth':
        return x
    elif key == 'photocurrent':
        return x
    elif key == 'saturation_current':
        return np.exp(x - 21)
    elif key == 'resistance_series':
        return x
    elif key == 'resistance_shunt':
        return np.exp(2 * (x - 1))
    else:
        raise Exception("Key '{}' not understood".format(key))


def _p_to_x(p, key):
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
    elif key == 'nNsVth':
        return p
    elif key == 'photocurrent':
        return p
    elif key == 'saturation_current':
        return np.log(p) + 21
    elif key == 'resistance_series':
        return p
    elif key == 'resistance_shunt':
        return np.log(p) / 2 + 1
    else:
        raise Exception("Key '{}' not understood".format(key))


#
# def fit_singlediode_linear_lsq(voltage, current, temperature_cell, poa,
#                                resistance_series, diode_factor, cells_in_series,
#                                alpha_sc, Eg_ref=1.121, dEgdT=-0.0002677,
#                                temperature_ref=25, irrad_ref=1000):
#     Tcell_K = temperature_cell + 273.15
#     Tref_K = temperature_ref + 273.15
#
#     # Boltzmann constant in eV/K
#     k = 8.617332478e-05
#     q = 1.602e-19
#
#     Eg = Eg_ref * (1 + dEgdT * (Tcell_K - Tref_K))
#
#     nNsVth = diode_factor * cells_in_series * k * Tcell_K
#
#     sat_current_multiplier = (((Tcell_K / Tref_K) ** 3) *
#                               (np.exp(Eg_ref / (k * (Tref_K)) - (
#                                       Eg / (k * (Tcell_K))))))
#
#     scale = 1e-9
#     X = np.zeros(shape=(len(temperature_cell), 3))
#     X[:, 0] = poa / irrad_ref
#     X[:, 1] = -1 * scale * sat_current_multiplier * (
#             np.exp((voltage + current * resistance_series) / nNsVth) - 1)
#     X[:, 2] = -(voltage + current * resistance_series) / (poa / irrad_ref)
#
#     Y = current - poa / irrad_ref * alpha_sc * (
#             temperature_cell - temperature_ref)
#
#     bounds = ([0, -np.inf, 0.0001], [1e5, np.inf, 1e5])
#
#     # Solve the problem
#     soln = lsq_linear(X, Y, bounds=bounds)
#     coeff = soln['x']
#     #     coeff = np.dot(pinv(X), Y)
#
#     loss = np.mean(np.abs(np.dot(X, coeff) - Y))
#     out = {
#         'photocurrent_ref': coeff[0],
#         'saturation_current_ref': coeff[1] * scale,
#         'resistance_shunt_ref': 1 / coeff[2],
#         'resistance_series_ref': resistance_series,
#         'diode_factor': diode_factor,
#         'nNsVth_ref': diode_factor * cells_in_series * k * Tref_K,
#         'loss': loss,
#         'solution': soln,
#     }
#
#     return out


def production_data_curve_fit(
        temperature_cell,
        effective_irradiance,
        operating_cls,
        voltage,
        current,
        lower_bounds=None,
        upper_bounds=None,
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
        singlediode_method='fast',
        method='minimize',
        use_mpp_points=True,
        use_voc_points=True,
        use_clip_points=True,
        fit_params=None,
        saturation_current_multistart=None,
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
    voltage = np.array(voltage)
    current = np.array(current)
    effective_irradiance = np.array(effective_irradiance)
    temperature_cell = np.array(temperature_cell)
    operating_cls = np.array(operating_cls)

    if p0 is None:
        p0 = dict(
            diode_factor=1.0,
            photocurrent_ref=6,
            saturation_current_ref=1e-9,
            resistance_series_ref=0.2,
            conductance_shunt_extra=0.001
        )

    if fit_params is None:
        # Parameters that are optimized in fit.
        fit_params = ['diode_factor',
                      'photocurrent_ref',
                      'saturation_current_ref',
                      'resistance_series_ref',
                      'conductance_shunt_extra']

    if lower_bounds is None:
        lower_bounds = dict(
            diode_factor=0.5,
            photocurrent_ref=0.01,
            saturation_current_ref=1e-13,
            resistance_series_ref=0,
            conductance_shunt_extra=0
        )

    if upper_bounds is None:
        upper_bounds = dict(
            diode_factor=2,
            photocurrent_ref=20,
            saturation_current_ref=1e-5,
            resistance_series_ref=1,
            conductance_shunt_extra=10
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

    # print('Effective irradiance: ')
    # print(effective_irradiance)
    #
    # Weights (equally weighted currently)
    weights = np.zeros_like(operating_cls)
    weights[operating_cls == 0] = 1
    weights[operating_cls == 1] = 1
    weights[operating_cls == 2] = 1

    if verbose:
        print('Total number points: {}'.format(len(operating_cls)))
        print('Number MPP points: {}'.format(np.sum(operating_cls == 0)))
        print('Number Voc points: {}'.format(np.sum(operating_cls == 1)))
        print('Number Clipped points: {}'.format(np.sum(operating_cls == 2)))

    # If no points left after removing unused classes, function returns.
    if len(effective_irradiance) == 0 or len(
            operating_cls) == 0 or len(voltage) == 0 or len(current) == 0:
        p = dict(
            diode_factor=np.nan,
            photocurrent_ref=np.nan,
            saturation_current_ref=np.nan,
            resistance_series_ref=np.nan,
            conductance_shunt_extra=np.nan
        )
        if verbose:
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
        singlediode_method=singlediode_method,
    )

    # Set kwargs
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

    # note that this will be the order of parameters in the model function.
    # fit_params = p0.keys()

    # Set scale factor for current and voltage:
    # current_median = np.median(current[operating_cls == 0])
    # voltage_median = np.median(voltage[operating_cls == 0])

    current_median = np.median(current)
    voltage_median = np.median(voltage)

    def model(x):
        """
        Return vdc, idc operating point given the fixed parameters.

        Parameters
        ----------
        x

        Returns
        -------
        voltage_fit

        current_fit

        """
        # TODO: Maybe there is a better way to pass in the extra args.
        p = model_kwargs.copy()
        n = 0
        for param in fit_params:
            p[param] = _x_to_p(x[n], param)
            n = n + 1
        voltage_fit, current_fit = pv_system_single_diode_model(**p)

        return voltage_fit, current_fit

    # def barron_loss(x, alpha=0.1, c=1):
    #     return np.abs(alpha - 2) / alpha * (
    #                 ((x / c) ** 2 / np.abs(alpha - 2) + 1) ** (alpha / 2) - 1)
    #
    # def huber_loss(x,delta=1):
    #     return np.where(x<=1,x**2/2,delta*(np.abs(x)-delta/2))

    def residual(x):
        # TODO: should explicitly pass in voltage_median, current_median. Use functools.partial to define the full function and then freeze certain params.
        # TODO: change to huber or L2 because BFGS shouldn't work well with L1.

        # Maybe consider Huber loss.
        voltage_fit, current_fit = model(x)
        # return np.nanmean(((voltage_fit - voltage) * weights / voltage_median) ** 2 + \
        #                   ((current_fit - current) * weights / current_median) ** 2)

        # Mean absolute error
        # Note that summing first and then calling nanmean is slightly faster.
        return np.nanmean(
            np.abs(voltage_fit - voltage) * weights / voltage_median +
            np.abs(current_fit - current) * weights / current_median)

        # return np.nanmean(np.log(np.cosh( (voltage_fit - voltage) * weights)) + \
        #                   np.log(np.cosh( (current_fit - current) * weights)) )

        # return np.nanmean( barron_loss( (voltage_fit - voltage) * weights) + \
        #                    barron_loss((current_fit - current) * weights) )
        # return np.nanmean( huber_loss( (voltage_fit - voltage) * weights) + \
        #                    huber_loss((current_fit - current) * weights) )

    # print(signature(model))

    # print('Starting residual: ',
    #       residual([p_to_x(p0[k], k) for k in fit_params]))

    if method == 'minimize':
        if solver.lower() == 'nelder-mead':
            bounds = None
        elif solver.upper() == 'L-BFGS-B':
            bounds = scipy.optimize.Bounds(
                [_p_to_x(lower_bounds[k], k) for k in fit_params],
                [_p_to_x(upper_bounds[k], k) for k in fit_params])
        else:
            raise Exception('solver must be "Nelder-Mead" or "L-BFGS-B"')

        if verbose:
            print('--')
            print('p0:')
            for k in fit_params:
                print(k, p0[k])
            print('--')

        # print('p0: ', p0)
        # print('bounds:', bounds)

        # print('Method: {}'.format(solver))
        if verbose:
            print('Performing minimization... ')

        if 'saturation_current_ref' in fit_params:
            saturation_current_ref_start = p0['saturation_current_ref']
            loss = np.inf
            for Io_multiplier in saturation_current_multistart:

                p0[
                    'saturation_current_ref'] = saturation_current_ref_start * Io_multiplier
                # Get numerical fit start point.
                x0 = [_p_to_x(p0[k], k) for k in fit_params]

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
            x0 = [_p_to_x(p0[k], k) for k in fit_params]

            res = minimize(residual,
                           x0=x0,
                           bounds=bounds,
                           method=solver,
                           options=dict(
                               # maxiter=100,
                               disp=False,
                               # ftol=0.001,
                           ),
                           )

        loss = np.inf

        if verbose:
            print(res)
        n = 0
        p_fit = {}
        for param in fit_params:
            p_fit[param] = _x_to_p(res.x[n], param)
            n = n + 1

        # print('Best fit parameters (with scale included):')
        # for p in x_fit:
        #     print('{}: {}'.format(p, x_fit[p]))
        # print('Final Residual: {}'.format(res['fun']))
        return p_fit, res['fun'], res
    elif method == 'basinhopping':
        # lower_bounds_x = [p_to_x(lower_bounds[k], k) for k in fit_params]
        # upper_bounds_x = [p_to_x(upper_bounds[k], k) for k in fit_params]
        x0 = [_p_to_x(p0[k], k) for k in fit_params]

        res = basinhopping(residual,
                           x0=x0,
                           niter=100,
                           T=0.2,
                           stepsize=0.1)
        n = 0
        p_fit = {}
        for param in fit_params:
            p_fit[param] = _x_to_p(res.x[n], param)
            n = n + 1

        # print('Best fit parameters (with scale included):')
        # for p in x_fit:
        #     print('{}: {}'.format(p, x_fit[p]))
        # print('Final Residual: {}'.format(res['fun']))
        return p_fit, res['fun'], res
