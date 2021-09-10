"""
Example for fitting the Desoto single diode model

"""

import numpy as np
import pandas as pd
from pvpro.fit import fit_singlediode_model
import time
from pvlib.pvsystem import calcparams_desoto, singlediode
from pvlib.ivtools.sdm import fit_desoto_sandia

# Generate synthetic data.
poa_list = np.linspace(200, 1000, 15)
temperature_cell_list = np.linspace(15, 25, 9)

specs = dict(
    alpha_sc=0.053 * 1e-2 * 8.6,
    cells_in_series=60,
    beta_voc=-0.351 * 1e-2 * 37.0
)
module = dict(photocurrent_ref=11,
              saturation_current_ref=1e-9,
              resistance_series_ref=0.3,
              resistance_shunt_ref=50,
              diode_factor=1.0,
              )
Vth_ref = 1.381e-23 * (273.15 + 25) / 1.602e-19
module['nNsVth_ref'] = module['diode_factor'] * specs[
    'cells_in_series'] * Vth_ref

voltage = np.array([])
current = np.array([])
temperature_cell = np.array([])
poa = np.array([])
ivcurve_number = np.array([])
ivcurves = {'v': [], 'i': [], 'v_oc': [], 'ee': [], 'tc': [], 'v_oc': [],
            'v_mp': [], 'i_mp': [], 'i_sc': []}

ivcurve_points = 50
n = 0
np.random.seed(0)
for j in range(len(poa_list)):
    for k in range(len(temperature_cell_list)):

        poa_curr = float(poa_list[j] * np.random.normal(1,0.01,1))
        tc_curr = float(temperature_cell_list[k] + np.random.normal(0, 0.5, 1))
        IL, I0, Rs, Rsh, nNsVth = calcparams_desoto(
            effective_irradiance=poa_curr,
            temp_cell=tc_curr,
            alpha_sc=specs['alpha_sc'],
            a_ref=module['nNsVth_ref'],
            I_L_ref=module['photocurrent_ref'],
            I_o_ref=module['saturation_current_ref'],
            R_sh_ref=module['resistance_shunt_ref'],
            R_s=module['resistance_series_ref'])
        out = singlediode(IL, I0, Rs, Rsh, nNsVth, ivcurve_pnts=ivcurve_points)

        # Add random noise
        # out['i'] = out['i'] + 0.2 * (np.random.rand(len(out['i'])) - 0.5)

        voltage = np.append(voltage, out['v'])
        current = np.append(current, out['i'])
        temperature_cell = np.append(temperature_cell,
                                     np.repeat(temperature_cell_list[k],
                                               len(out['v'])))
        poa = np.append(poa, np.repeat(poa_list[j], len(out['v'])))
        ivcurve_number = np.append(ivcurve_number, np.repeat(n, len(out['v'])))

        ivcurves['v'].append(out['v'])
        ivcurves['i'].append(out['i'])
        ivcurves['v_oc'].append(out['v_oc'])
        ivcurves['i_sc'].append(out['i_sc'])
        ivcurves['v_mp'].append(out['v_mp'])
        ivcurves['i_mp'].append(out['i_mp'])
        ivcurves['tc'].append(temperature_cell_list[k])
        ivcurves['ee'].append(poa_list[j])

        n = n + 1

ivcurves.keys()
for key in ['v_oc', 'ee', 'tc', 'v_mp', 'i_mp', 'i_sc']:
    ivcurves[key] = np.array(ivcurves[key])

# Perform fit using fit_desoto_lbl
start_time = time.time()
ret = fit_singlediode_model(voltage=voltage,
                     current=current,
                     temperature_cell=temperature_cell,
                     poa=poa,
                     cells_in_series=specs['cells_in_series'],
                     alpha_isc=specs['alpha_sc'],
                     linear_solver='lsq_linear',
                     model='desoto',
                     tol=1e-9,
                            verbose=False)
print('Elapsed Time (fit_singlediode_model): {} s'.format(
    time.time() - start_time))

# Perform fit using fit_desoto_sandia
start_time = time.time()
retds = fit_desoto_sandia(ivcurves, specs=specs)
print('Elapsed Time (fit_desoto_sandia): {} s'.format(time.time() - start_time))

# Inspect results.
df = pd.DataFrame()
for key in ['photocurrent_ref', 'saturation_current_ref',
            'resistance_shunt_ref', 'resistance_series_ref', 'diode_factor',
            'nNsVth_ref']:
    df.loc['true', key] = module[key]
    df.loc['fit_singlediode_model', key] = ret[key]

df.loc['fit_desoto_sandia', 'photocurrent_ref'] = retds['I_L_ref']
df.loc['fit_desoto_sandia', 'saturation_current_ref'] = retds['I_o_ref']
df.loc['fit_desoto_sandia', 'resistance_shunt_ref'] = retds['R_sh_ref']
df.loc['fit_desoto_sandia', 'resistance_series_ref'] = retds['R_s']
df.loc['fit_desoto_sandia', 'nNsVth_ref'] = retds['a_ref']
df.loc['fit_desoto_sandia', 'diode_factor'] = retds['a_ref'] / (
            specs['cells_in_series'] * Vth_ref)

pd.set_option('display.float_format', lambda x: '%.2e' % x)
print(df.transpose())
