# -*- coding: utf-8 -*-
"""
Generate a synthetic dataset for testing pvpro methods.

Created on Sun Apr 19 09:27:50 2020

@author: cliff, toddkarin
"""

import numpy as np
import pandas as pd
from pvpro.singlediode import calculate_temperature_coeffs, pvlib_single_diode
from pvlib.temperature import sapm_module, sapm_cell

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

# drop times when GHI is <= 1
df.drop(df.index[df['ghi'] <= 1.], inplace=True)

# assume poa = ghi, e.g., horizontal module
df['poa_actual'] = df['ghi']

# Simulate some noise on the measured poa irradiance.
poa_noise_level = 0.01
# poa_noise_level = 0.0001
# poa_noise_level = 0

np.random.seed(0)
df['poa_meas'] = df['poa_actual'] * np.random.normal(1, poa_noise_level, df['ghi'].shape)

# estimate module/cell temperature
df['temperature_module_actual'] = sapm_module(
    poa_global=df['poa_actual'],
    temp_air=df['temperature_air'],
    wind_speed=df['wind_speed'],
    a=-3.56,
    b=-0.075
)

df['temperature_cell_actual'] = sapm_cell(
    poa_global=df['poa_actual'],
    temp_air=df['temperature_air'],
    wind_speed=df['wind_speed'],
    a=-3.56,
    b=-0.075,
    deltaT=3)

# "measured" module temperature has noise.
temperature_noise_level = 1
# temperature_noise_level = 0.1
# temperature_noise_level = 0.0
df['temperature_module_meas'] = df['temperature_module_actual'] + \
        np.random.normal(0,temperature_noise_level, df['ghi'].shape)

q = 1.60218e-19  # Elementary charge in units of coulombs
kb = 1.38066e-23  # Boltzmann's constant in units of J/K

# time vector in years
t_years = (df.index - df.index[0]).days / 365


def step_change(start_val, end_val, t_years, t_step=2):
    y = np.zeros_like(t_years) + start_val
    y = y + (end_val - start_val) * (
            np.arctan(10 * (t_years - 2)) / np.pi + 0.5)
    return y


# make up a parameter set for the De Soto model
df['cells_in_series'] = 60
df['alpha_sc'] = 0.001
df['diode_factor'] = 1.1 - 0.01 * t_years
df['nNsVth_ref'] = df['diode_factor'] * df['cells_in_series'] * kb / q * (
        273.15 + 25)
df['photocurrent_ref'] = 6.0 - (0.1 * t_years - 0.05 * np.sin(2 * np.pi * t_years))
df['saturation_current_ref'] = 1e-9 + 0.01e-9 * t_years
# df['resistance_shunt_ref'] = step_change(1000, 100, t_years, 2)
df['resistance_shunt_ref'] = 400
df['conductance_shunt_extra'] = 0.000 + 0.0003 * t_years
# df['resistance_shunt_ref'] = 1000 - 950/4*t_years
# df.loc[t_years>2,'resistance_shunt_ref'] = 13
# df['resistance_series_ref'] = 0.2 +  0.4 * t_years
df['resistance_series_ref'] = 0.3 + 0.05 * t_years
df['resistance_series_ref'] = step_change(0.35, 0.7, t_years)
df['EgRef'] = 1.121
df['dEgdT'] = -0.0002677

# Calculate module operation over time.
out = pvlib_single_diode(
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
    singlediode_method='newton',
    ivcurve_pnts=None,
)

# Add to the dataframe
for k in out.keys():
    df[k] = out[k]

# Calculate module reference conditions over time.
out_ref = pvlib_single_diode(
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
    singlediode_method='newton',
    ivcurve_pnts=None,
)

for k in out_ref.keys():
    df[k + '_ref'] = out_ref[k]

# Calculate temperature coefficients
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

# Set DC operation point at MPP
df['v_dc'] = df['v_mp']
df['i_dc'] = df['i_mp']

# Change DC operation point to Voc during low irradiance conditions
df.loc[df['poa_actual'] < 50, 'v_dc'] = df.loc[
    df['poa_actual'] < 50, 'v_oc']
df.loc[df['poa_actual'] < 50, 'i_dc'] = 0


df.loc[df.index.day==1, 'v_dc'] = df.loc[df.index.day==1, 'v_oc']
df.loc[df.index.day==1, 'i_dc'] = 0


# Save data
df.to_pickle('synth01_out.pkl')
print('done!')



# print(df[['temperature_module_meas','poa_meas','v_dc','i_dc']][:100].to_csv(index=False))