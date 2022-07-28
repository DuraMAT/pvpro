"""
Example process of iv curves from NIST dataset.

Data available here:
https://pvdata.nist.gov/

"""

import os
import pandas as pd
import numpy as np
from pvlib.ivtools.sdm import fit_desoto_sandia
from pvlib.temperature import sapm_cell_from_module


# Filenames.
iv_filename = os.path.join('data','ivcurves-Ground-108-2015-07-01.csv')
ws1_filename = os.path.join('data','onemin-WS_1-2015-07-01.csv')
ws2_filename = os.path.join('data','onemin-WS_2-2015-07-01.csv')

# Load files
df = pd.read_csv(iv_filename)
ws1 = pd.read_csv(ws1_filename)
ws2 = pd.read_csv(ws2_filename)

# Get voltage sweep
voltage = np.array(list(df.keys())[1:]).astype('float')

# Choose fields for sensors.
irradiance_poa_key = 'RefCell5_Wm2_Avg'
temperature_module_key = 'RTD_C_Avg_13'
deltaT=3
specs = {'cells_in_series': 60,
         'alpha_sc': 0.053 * 1e-2 * 8.6,
         'beta_voc': -0.351 * 1e-2 * 37.0}

# Get sensor values.
poa = ws1[irradiance_poa_key]
tm = ws2[temperature_module_key]
tc = sapm_cell_from_module(tm, poa, deltaT)

# Initialize
ivcurves = {}
for key in ['i', 'v', 'tc', 'tm', 'ee', 'i_sc', 'v_oc', 'i_mp', 'v_mp',
            'timestamp','idx']:
    ivcurves[key] = []

n = 0

for k in range(len(df)):
    # for k in range(1):
    current = np.array(df.iloc[k, 1:]).astype('float')

    is_current_finite = np.isfinite(current)

    if np.sum(is_current_finite) > 10 and np.nanmax(current) > 0.1:
        power = current * voltage
        power[np.logical_not(is_current_finite)] = 0
        mpp_idx = np.argmax(power)
        #         print(mpp_idx)
        ivcurves['timestamp'].append(df.loc[k, 'timestamps'])
        ivcurves['i_mp'].append(current[mpp_idx])
        ivcurves['v_mp'].append(voltage[mpp_idx])
        ivcurves['i_sc'].append(current[0])

        ivcurves['v_oc'].append(np.max(voltage[is_current_finite]))
        ivcurves['i'].append(current[is_current_finite])
        ivcurves['v'].append(voltage[is_current_finite])
        ivcurves['tm'].append(tm[k])
        ivcurves['tc'].append(tc[k])
        ivcurves['ee'].append(poa[k])
        ivcurves['idx'].append(poa[k])
        n = n + 1

for key in ['tc', 'ee', 'i_sc', 'v_oc', 'i_mp', 'v_mp']:
    ivcurves[key] = np.array(ivcurves[key])


desoto = fit_desoto_sandia(ivcurves, specs)

print(desoto.keys())

print(desoto)


