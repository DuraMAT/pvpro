# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:27:50 2020

@author: cliff, toddkarin
"""

import pickle
import numpy as np
import pandas as pd
import pvlib
import pvpro
import time
from pvpro.singlediode import calculate_temperature_coeffs

# Load weather data
data = np.load('123796_37.89_-122.26_search-point_37.876_-122.247.npz')
# tmy3 = pvlib.iotools.tmy.read_tmy3(datafile)


df = pd.DataFrame({'dni': data['dni'],
                   'ghi': data['ghi'],
                   'temperature_air': data['temp_air'],
                   'wind_speed': data['wind_speed'],
                   'year': data['year'],
                   'month': data['month'],
                   'day': data['day'],
                   'hour': data['hour'],
                   'minute': data['minute']
                   })

df.index = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

# Clip shorter
df = df[:'2002-01-01']

# 15 minute interpolation
df = df.resample('15T').interpolate('linear')

# drop times when GHI is <= 10
df.drop(df.index[df['ghi'] <= 1.], inplace=True)

# assume poa = ghi, e.g., horizontal module
df['poa_actual'] = df['ghi']
# Simulate some noise on the measured poa irradiance.
poa_noise_level = 10
df['poa_meas'] = df['poa_actual'] * (
            1 + poa_noise_level * (np.random.random(df['ghi'].shape) - 0.5))

# estimate module/cell temperature
df['temperature_module_actual'] = pvlib.temperature.sapm_module(
    poa_global=df['poa_actual'],
    temp_air=df['temperature_air'],
    wind_speed=df['wind_speed'],
    a=-3.56,
    b=-0.075
)

df['temperature_cell_actual'] = pvlib.temperature.sapm_cell(
    poa_global=df['poa_actual'],
    temp_air=df['temperature_air'],
    wind_speed=df['wind_speed'],
    a=-3.56,
    b=-0.075,
    deltaT=3)

# "measured" module temperature has noise.
temperature_noise_level = 3
df['temperature_module_meas'] = df['temperature_module_actual'] + (
        np.random.random(df['ghi'].shape) - 0.5) * temperature_noise_level

q = 1.60218e-19  # Elementary charge in units of coulombs
kb = 1.38066e-23  # Boltzmann's constant in units of J/K

# time vector in years
t_years = (df.index - df.index[0]).days / 365


def step_change(start_val, end_val, t_years, t_step):
    y = np.zeros_like(t_years) + start_val
    y = y + (end_val - start_val) * (
            np.arctan(10 * (t_years - 2)) / np.pi + 0.5)
    return y


# make up a parameter set for the De Soto model
df['cells_in_series'] = 60
df['alpha_sc'] = 0.001
df['diode_factor'] = 1.1 - 0 * 0.01 * t_years
df['nNsVth_ref'] = df['diode_factor'] * df['cells_in_series'] * kb / q * (
        273.15 + 25)
df['photocurrent_ref'] = 6.0 - 0 * (
            0.1 * t_years - 0.1 * np.sin(2 * np.pi * t_years))
# df['photocurrent_ref'] = 6
df['saturation_current_ref'] = 1e-9 + 0 * 1e-9 * t_years
# df['resistance_shunt_ref'] = step_change(1000, 100, t_years, 2)
df['resistance_shunt_ref'] = 400
df['conductance_shunt_extra'] = 0.000 + 0.0004 * t_years
# df['resistance_shunt_ref'] = 1000 - 950/4*t_years
# df.loc[t_years>2,'resistance_shunt_ref'] = 13
# df['resistance_series_ref'] = 0.2 +  0.4 * t_years
df['resistance_series_ref'] = 0.3 + 0 * 0.05 * t_years
df['EgRef'] = 1.121
df['dEgdT'] = -0.0002677

out = pvpro.pvlib_single_diode(
    effective_irradiance=df['poa_actual'],
    temperature_cell=df['temperature_cell_actual'],
    resistance_shunt_ref=df['resistance_shunt_ref'],
    resistance_series_ref=df['resistance_series_ref'],
    diode_factor=df['diode_factor'],
    cells_in_series=df['cells_in_series'],
    alpha_isc=df['alpha_sc'],
    photocurrent_ref=df['photocurrent_ref'],
    saturation_current_ref=df['saturation_current_ref'],
    Eg_ref=df['EgRef'],
    dEgdT=df['dEgdT'],
    conductance_shunt_extra=df['conductance_shunt_extra'],
    method='newton',
    ivcurve_pnts=None,
)

# column_renamer = {'v_oc': 'voc', 'i_mp': 'imp', 'v_mp': 'vmp', 'p_mp': 'pmp',
#                   'i_sc': 'isc'}
# out.rename(columns=column_renamer, inplace=True)

for k in out.keys():
    df[k] = out[k]

out_ref = pvpro.pvlib_single_diode(
    effective_irradiance=1000,
    temperature_cell=25,
    resistance_shunt_ref=df['resistance_shunt_ref'],
    resistance_series_ref=df['resistance_series_ref'],
    diode_factor=df['diode_factor'],
    cells_in_series=df['cells_in_series'],
    alpha_isc=df['alpha_sc'],
    photocurrent_ref=df['photocurrent_ref'],
    saturation_current_ref=df['saturation_current_ref'],
    Eg_ref=df['EgRef'],
    dEgdT=df['dEgdT'],
    conductance_shunt_extra=df['conductance_shunt_extra'],
    method='newton',
    ivcurve_pnts=None,
)
# out_ref.rename(columns=column_renamer, inplace=True)

for k in out_ref.keys():
    df[k + '_ref'] = out_ref[k]

tempco = calculate_temperature_coeffs(
    resistance_shunt_ref=df['resistance_shunt_ref'],
    resistance_series_ref=df['resistance_series_ref'],
    diode_factor=df['diode_factor'],
    cells_in_series=df['cells_in_series'],
    alpha_isc=df['alpha_sc'],
    photocurrent_ref=df['photocurrent_ref'],
    saturation_current_ref=df['saturation_current_ref'],
    band_gap_ref=df['EgRef'],
    dEgdT=df['dEgdT'],
    conductance_shunt_extra=df['conductance_shunt_extra'],
)

for k in tempco:
    df[k] = tempco[k]

# Set operation point at v_mp/i_mp except at low irradiance.
df['v_dc'] = df['v_mp']
df['i_dc'] = df['i_mp']
# Set Voc points at low irradiances.
df.loc[df['poa_actual'] < 50, 'v_dc'] = df.loc[
    df['poa_actual'] < 50, 'v_oc']
df.loc[df['poa_actual'] < 50, 'i_dc'] = 0

df.to_pickle('synth01_out.pkl')
print('done!')

# with open('synth_data.dat', 'wb') as outfile:
#     pickle.dump([module_parameters, supplement, ivcurves], outfile)
