"""
Example process of iv curves from NIST dataset.

Data available here:
https://pvdata.nist.gov/

"""

import os
import pandas as pd
import numpy as np
from pvlib.ivtools.sde import fit_sandia_simple
from pvlib.temperature import sapm_cell_from_module


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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

    if np.sum(is_current_finite) > 10 and np.nanmax(current) > 7:
        power = current * voltage
        power[np.logical_not(is_current_finite)] = 0
        mpp_idx = np.argmax(power)

        v_oc = np.max(voltage[is_current_finite])
        i_sc = current[0]

        # Plot IV curve.
        plt.plot(voltage, current)
        plt.plot(0, i_sc,'r.')
        plt.plot(v_oc,0,'r.')
        plt.show()

        v_mp_i_mp = (voltage[mpp_idx], current[mpp_idx])

        out = fit_sandia_simple(voltage=voltage, current=current,
                                v_mp_i_mp=v_mp_i_mp,
                                v_oc=v_oc,
                                i_sc=i_sc)



        break

"""
Traceback (most recent call last):
  File "/Users/toddkarin/opt/anaconda3/envs/pvpro/lib/python3.8/site-packages/IPython/core/interactiveshell.py", line 3441, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-2-0f850edcbc3d>", line 1, in <module>
    runfile('/Volumes/GoogleDrive/My Drive/LBL/projects/pvpro/examples/nist-iv03_process.py', wdir='/Volumes/GoogleDrive/My Drive/LBL/projects/pvpro/examples')
  File "/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev/_pydev_bundle/pydev_umd.py", line 197, in runfile
    pydev_imports.execfile(filename, global_vars, local_vars)  # execute the script
  File "/Applications/PyCharm.app/Contents/plugins/python/helpers/pydev/_pydev_imps/_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "/Volumes/GoogleDrive/My Drive/LBL/projects/pvpro/examples/nist-iv03_process.py", line 77, in <module>
    out = fit_sandia_simple(voltage=voltage, current=current,
  File "/Users/toddkarin/opt/anaconda3/envs/pvpro/lib/python3.8/site-packages/pvlib/ivtools/sde.py", line 166, in fit_sandia_simple
    beta0, beta1 = _sandia_beta0_beta1(voltage, current, vlim, v_oc)
  File "/Users/toddkarin/opt/anaconda3/envs/pvpro/lib/python3.8/site-packages/pvlib/ivtools/sde.py", line 218, in _sandia_beta0_beta1
    raise RuntimeError("Parameter extraction failed: beta0={}, beta1={}"
RuntimeError: Parameter extraction failed: beta0=nan, beta1=nan
"""