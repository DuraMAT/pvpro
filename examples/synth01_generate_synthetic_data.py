# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:27:50 2020

@author: cliff, toddkarin
"""

import pickle
import numpy as np
import pandas as pd
import pvlib


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

# drop times when GHI is <= 10
df.drop(df.index[df['ghi'] <= 10.], inplace=True)

# assume poa = ghi, e.g., horizontal module
df['poa_actual'] = df['ghi']
# Simulate some noise on the measured poa irradiance.
df['poa_meas'] = df['poa_actual'] + (np.random.random(df['ghi'].shape) - 0.5) * 2

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
df['temperature_module_meas'] = df['temperature_module_actual'] + (
        np.random.random(df['ghi'].shape) - 0.5) * 2




q = 1.60218e-19  # Elementary charge in units of coulombs
kb = 1.38066e-23  # Boltzmann's constant in units of J/K

# time vector in years
t_years = (df.index-df.index[0]).seconds/60/60/24/365

def step_change(start_val, end_val, t_years, t_step):
    y = np.zeros_like(t_years) + start_val
    y = y + (end_val - start_val) * (
            np.arctan(10 * (t_years - 2)) / np.pi + 0.5)
    return y

# make up a parameter set for the De Soto model
df['cells_in_series'] = 60
df['alpha_sc'] = 0.001
df['diode_factor'] = 1.15
df['a_ref'] = df['diode_factor'] * df['cells_in_series'] * kb / q * (
        273.15 + 25)
df['photocurrent_ref'] = 6.0 - 0.1 * t_years
# df['photocurrent_ref'] = 6
df['saturation_current_ref'] = 1e-9
df['resistance_shunt_ref'] = step_change(1000, 100, t_years, 2)
# df['resistance_shunt_ref'] = 1000
# df.loc[t_years>2,'resistance_shunt_ref'] = 13
df['resistance_series_ref'] = 0.5
df['EgRef'] = 1.121
df['dEgdT'] = -0.0002677

# Calculate Desoto params
iph, io, rs, rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
    df['poa_actual'],
    df['temperature_cell_actual'],
    alpha_sc=df['alpha_sc'],
    a_ref=df['a_ref'],
    I_L_ref=df['photocurrent_ref'],
    I_o_ref=df['saturation_current_ref'],
    R_sh_ref=df['resistance_shunt_ref'],
    R_s=df[
        'resistance_series_ref'],
    EgRef=df['EgRef'],
    dEgdT=df['dEgdT'],
)



df['photocurrent'] = iph
df['saturation_current'] = io
df['resistance_series'] = rs
df['resistance_shunt'] = rsh
df['nNsVth'] = nNsVth

# Calcuate iv curve key points.
ivcurves = pvlib.pvsystem.singlediode(iph, io, rs, rsh, nNsVth)

# Do the same at reference conditions.
iph_ref, io_ref, rs_ref, rsh_ref, nNsVth_ref = pvlib.pvsystem.calcparams_desoto(
    effective_irradiance=1000,
    temp_cell=25,
    alpha_sc=df['alpha_sc'],
    a_ref=df['a_ref'],
    I_L_ref=df['photocurrent_ref'],
    I_o_ref=df['saturation_current_ref'],
    R_sh_ref=df['resistance_shunt_ref'],
    R_s=df[
        'resistance_series_ref'],
    EgRef=df['EgRef'],
    dEgdT=df['dEgdT'],
)
ivcurves_ref = pvlib.pvsystem.singlediode(iph_ref, io_ref, rs_ref, rsh_ref, nNsVth_ref)

# Add to dataframe.
for k in ivcurves.keys():
    df[k] = ivcurves[k]


# Set operation point at v_mp/i_mp except at low irradiance.
df['v_operation'] = df['v_mp']
df['i_operation'] = df['i_mp']
# Set Voc points at low irradiances.
df.loc[df['poa_actual'] < 50, 'v_operation'] = df.loc[df['poa_actual'] < 50, 'v_oc']
df.loc[df['poa_actual'] < 50, 'i_operation'] = 0


# Add reference points to dataframe.
for k in ivcurves_ref.keys():
    df[k + '_ref'] = ivcurves_ref[k]



df.to_pickle('synth01_out.pkl')


# with open('synth_data.dat', 'wb') as outfile:
#     pickle.dump([module_parameters, supplement, ivcurves], outfile)
