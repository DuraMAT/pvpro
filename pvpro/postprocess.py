from rdtools.degradation import degradation_year_on_year
import numpy as np
import pandas as pd


def analyze_yoy(pfit : 'dataframe'):
    out = {}

    for k in ['photocurrent_ref', 'saturation_current_ref',
              'resistance_series_ref', 'resistance_shunt_ref',
              'conductance_shunt_extra', 'diode_factor', 'i_sc_ref',
              'v_oc_ref', 'i_mp_ref', 'v_mp_ref', 'p_mp_ref', 'v_mp_ref_est',
              'i_mp_ref_est', 'p_mp_ref_est',
              'nNsVth_ref']:
        if k in pfit:
            Rd_pct, Rd_CI, calc_info = degradation_year_on_year(pd.Series(pfit[k]),
                                                                recenter=False)
            renorm = np.nanmedian(pfit[k])
            if renorm == 0:
                renorm = np.nan

            out[k] = {
                'change_per_year': Rd_pct * 1e-2,
                'percent_per_year': Rd_pct / renorm,
                'change_per_year_CI': np.array(Rd_CI) * 1e-2,
                'percent_per_year_CI': np.array(Rd_CI) / renorm,
                'calc_info': calc_info,
                'median': np.nanmedian(pfit[k])}

    return out

def calculate_error_real(pfit : 'dataframe', df_ref : 'dataframe', nrolling : int = 1):
    keys = ['diode_factor',
            'photocurrent_ref', 'saturation_current_ref',
            'resistance_series_ref',
            'resistance_shunt_ref',
            'i_sc_ref', 'v_oc_ref',
            'i_mp_ref', 'v_mp_ref', 'p_mp_ref']

    all_error_df = pd.DataFrame(index=keys, columns=['rms', 'rms_rela', 'corr_coef'])

    for key in keys:
        para_ref = df_ref[key].rolling(nrolling).mean()
        para_pvpro = pfit[key].rolling(nrolling).mean()

        mask = np.logical_and(~np.isnan(para_pvpro), ~np.isnan(para_ref))
        corrcoef = np.corrcoef(para_pvpro[mask], para_ref[mask])
        rela_rmse = np.sqrt(np.mean((para_pvpro[mask]-para_ref[mask]) ** 2))/np.mean(para_pvpro[mask])

        all_error_df['rms_rela'][key] = rela_rmse
        all_error_df['corr_coef'][key] = corrcoef[0,1]

    return all_error_df

def calculate_error_synthetic(pfit : 'dataframe', df : 'dataframe', zero_mean : bool =False):
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

    all_error_df = pd.DataFrame(index=keys, columns=['rms', 'rms_rela', 'corr_coef'])
    
    for k in keys:
       
        p1 = dft[k]
        p2 = pfit[k]
        mask = ~np.isnan(p1) & ~np.isnan(p2)# remove nan value for corrcoef calculation
        
        all_error_df.loc[k, 'rms'] = np.sqrt(np.mean((p1[mask]-p2[mask]) ** 2))
        all_error_df.loc[k,'rms_rela'] = np.sqrt(np.nanmean(((p1[mask]-p2[mask])/p1[mask]) ** 2))
        all_error_df.loc[k, 'corr_coef'] = np.corrcoef(p1[mask], 
                                                  p2[mask])[0,1]



    return all_error_df