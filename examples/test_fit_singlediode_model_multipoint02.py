"""
Example for fitting the Desoto single diode model

"""

import numpy as np
import pandas as pd
from pvpro.fit import fit_singlediode_model
import time
from pvlib.pvsystem import calcparams_desoto, singlediode
from pvlib.ivtools.sdm import fit_desoto_sandia
from pickle import dump,load


dfs = load(open('data/test_fit_singlediode_model_multipoint_out.pkl', "rb" ) )


summary = pd.DataFrame()
for model in ['fit_singlediode_model_error','fit_desoto_sandia_error']:
    for key in dfs[0].keys():
        values = [dfs[k].loc[model, key] for k in range(len(dfs))]
        # summary.loc[model, key + '_mean_abs_fractional_error'] = np.mean(values)
        # summary.loc[model, key + '_max_abs_fractional_error'] = np.max(values)
        summary.loc[key + '_P90_abs_percent_error',model] = np.percentile(values,90)
        # summary.loc[key + '_dev_of_error', model] = np.std(values - np.mean(values))


values = [dfs[k].loc['fit_desoto_sandia', 'evaluation_time']/dfs[k].loc['fit_singlediode_model', 'evaluation_time'] for k in range(len(dfs))]
# summary.loc[model, key + '_mean_abs_fractional_error'] = np.mean(values)
# summary.loc[model, key + '_max_abs_fractional_error'] = np.max(values)
summary.loc['median_speedup','fit_singlediode_model_error'] = np.percentile(values,50)


# summary.loc['P90 Percent Error',:] = summary.keys()
print(summary[['fit_singlediode_model_error','fit_desoto_sandia_error']])


summary.index

idx_out = [
       'photocurrent_ref_P90_abs_percent_error',
'log_saturation_current_ref_P90_abs_percent_error',
'diode_factor_P90_abs_percent_error',
       # 'saturation_current_ref_P90_abs_percent_error',
'resistance_series_ref_P90_abs_percent_error',
       'resistance_shunt_ref_P90_abs_percent_error',
       # 'nNsVth_ref_P90_abs_percent_error',
       # 'conductance_shunt_ref_P90_abs_percent_error',
       'i_sc_P90_abs_percent_error', 'v_oc_P90_abs_percent_error',
       'i_mp_P90_abs_percent_error', 'v_mp_P90_abs_percent_error',
       'p_mp_P90_abs_percent_error',
    # 'i_x_P90_abs_percent_error',
       # 'i_xx_P90_abs_percent_error',
       ]

with open('test_fit_table.tex','w') as f:
    f.write(summary.loc[idx_out].to_latex(float_format='{:.2%}'.format).replace('\_',' '))