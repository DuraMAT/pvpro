from rdtools.degradation import degradation_year_on_year
import numpy as np
import pandas as pd


def analyze_yoy(pfit):
    out = {}

    for k in ['photocurrent_ref', 'saturation_current_ref',
              'resistance_series_ref',
              'conductance_shunt_extra', 'diode_factor', 'i_sc_ref',
              'v_oc_ref', 'i_mp_ref', 'v_mp_ref', 'p_mp_ref', 'v_mp_ref_est',
              'i_mp_ref_est', 'p_mp_ref_est',
              'nNsVth_ref']:
        if k in pfit:
            Rd_pct, Rd_CI, calc_info = degradation_year_on_year(pd.Series(pfit[k]),
                                                                recenter=False)
            renorm = np.median(pfit[k])
            if renorm == 0:
                renorm = np.nan

            out[k] = {
                'change_per_year': Rd_pct * 1e-2,
                'percent_per_year': Rd_pct / renorm,
                'change_per_year_CI': np.array(Rd_CI) * 1e-2,
                'percent_per_year_CI': np.array(Rd_CI) / renorm,
                'calc_info': calc_info,
                'median': np.median(pfit[k])}

    return out


def calculate_rms_error(pfit, df,zero_mean=False):
    dft = pd.DataFrame()

    keys = ['diode_factor',
            'photocurrent_ref', 'saturation_current_ref',
            'resistance_series_ref',
            'conductance_shunt_extra', 'resistance_shunt_ref',
            'nNsVth_ref', 'i_sc_ref', 'v_oc_ref',
            'i_mp_ref', 'v_mp_ref', 'p_mp_ref']

    for k in range(len(pfit)):
        cax = np.logical_and(df.index >= pfit['t_start'].iloc[k],
                             df.index < pfit['t_end'].iloc[k])
        dfcurr_mean = df[cax][keys].mean()

        for key in dfcurr_mean.keys():
            dft.loc[pfit['t_start'].iloc[k], key] = dfcurr_mean[key]


    rms_error = {}
    for k in keys:
        if zero_mean:
            p1 = dft[k] - np.median(dft[k])
            p2 = pfit[k] - np.median(pfit[k])
        else:
            p1 = dft[k]
            p2 = pfit[k]
        rms_error[k] = np.sqrt(np.mean((p1-p2) ** 2))

    return rms_error