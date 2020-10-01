"""
Example estimation of single diode model parameters

@author: toddkarin
"""
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import synthetic data
df = pd.read_pickle('synth01_out.pkl')
import time
from pvpro.estimate import estimate_singlediode_params

delta_T = 3
df = df[:2000]


est, result = estimate_singlediode_params(
    poa=df['poa_meas'],
    temperature_module=df['temperature_module_meas'],
    vmp=df['v_dc'],
    imp=df['i_dc'],
    delta_T=delta_T,
    resistance_series_ref=0.4,
    figure=True,
    verbose=True,
    max_iter=50,
    optimize_Rs_Io=True
)

# Format output to compare
compare = pd.DataFrame(est, index=['Estimate'])

for k in est.keys():
    if k in df:
        compare.loc['True', k] = df[k].mean()
print(compare.transpose().to_string())

plt.figure(8)
plt.savefig('figures/vmp_fit.pdf',
            bbox_inches='tight')


plt.figure(11)
plt.savefig('figures/imp_fit.pdf',
            bbox_inches='tight')


plt.figure(101,figsize=(4,5))
plt.clf()
# plt.subplots(3,3)
n=1
for k in ['resistance_series_ref', 'saturation_current_ref']:
    plt.subplot(2,1,n)
    plt.plot(result.index,result[k],label='optimization')
    plt.plot(result.index, df[k].mean()*np.ones_like(result.index),label='true')
    plt.ylabel(k)
    plt.xlabel('Optimization Iteration')
    n=n+1
plt.show()
plt.savefig('figures/estimate_convergence.pdf',
            bbox_inches='tight')