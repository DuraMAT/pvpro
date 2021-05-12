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
