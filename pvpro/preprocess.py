
import numpy as np
from tqdm import tqdm
import pandas as pd

from pvlib.location import Location
from pvlib.solarposition import get_solarposition
from pvlib.irradiance import get_total_irradiance
from pvlib.tracking import singleaxis
from pvlib.clearsky import detect_clearsky
import pandas as pd

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model import HuberRegressor

def monotonic(ac_power, fractional_rate_limit=0.05):
    dP = np.diff(ac_power)
    dP = np.append(dP, dP[-1])
    boolean_mask = np.logical_or.reduce((
        np.logical_and(dP > 0, np.roll(dP, 1) > 0),
        np.logical_and(dP < 0, np.roll(dP, 1) < 0),
        np.abs(dP / ac_power) < fractional_rate_limit
    ))
    return boolean_mask


def find_huber_outliers(x, y, sample_weight=None,
                        fit_intercept=True,
                        epsilon=2.5):
    """
    Identify outliers based on a linear fit of current at maximum power point
    to plane-of-array irradiance.

    Parameters
    ----------
    poa
    current
    sample_weight
    epsilon

    Returns
    -------

    """

    mask = np.logical_and(np.isfinite(x), np.isfinite(y))

    if np.sum(mask) <= 2:
        print('Need more than two points for linear regression.')
        return outliers

    X = np.atleast_2d(x).transpose()
    y = np.array(y)

    huber = HuberRegressor(epsilon=epsilon,
                           fit_intercept=fit_intercept)
    huber.fit(np.atleast_2d(x[mask]).transpose(), y[mask],
              sample_weight=sample_weight[mask])

    #     outliers = huber.outliers_

    def is_outlier(x, y):
        X = np.atleast_2d(x).transpose()
        residual = np.abs(
            y - safe_sparse_dot(X, huber.coef_) - huber.intercept_)
        outliers = residual > huber.scale_ * huber.epsilon
        return outliers

    def is_inbounds(x, y):
        X = np.atleast_2d(x).transpose()
        residual = np.abs(
            y - safe_sparse_dot(X, huber.coef_) - huber.intercept_)
        outliers = residual <= huber.scale_ * huber.epsilon
        return outliers

    outliers = is_outlier(x, y)

    huber.is_outlier = is_outlier
    huber.is_inbounds = is_inbounds

    return outliers, huber


def find_linear_model_outliers_timeseries(x, y,
                                          boolean_mask=None,
                                          fit_intercept=True,
                                          points_per_iteration=20000,
                                          epsilon=2.5,
                                          ):
    outliers = np.zeros_like(x).astype('bool')

    #     poa = poa[boolean_mask]
    #     current = current[boolean_mask]

    #     lower_iter_idx = np.arange(0, len(x), points_per_iteration).astype('int')
    #     upper_iter_idx = lower_iter_idx + points_per_iteration
    #     if upper_iter_idx[-1] != len(x):
    #         upper_iter_idx[-1] = len(x)
    #         upper_iter_idx[-2] = len(x) - points_per_iteration

    isfinite = np.logical_and(np.isfinite(x), np.isfinite(y))
    lower_iter_idx = []
    upper_iter_idx = []

    lenx = len(x)
    n = 0
    isfinite_count = np.cumsum(isfinite)
    while True:
        if n == 0:
            lower_lim = 0
        else:
            lower_lim = upper_iter_idx[-1]

        upper_lim_finder = isfinite_count == int((n + 1) * points_per_iteration)

        if np.sum(upper_lim_finder) >= 1:
            upper_lim = np.where(upper_lim_finder)[0][0]
        else:
            upper_lim=np.nan

        if lower_lim < lenx and upper_lim < lenx:
            lower_iter_idx.append(lower_lim)
            upper_iter_idx.append(upper_lim)
        else:
            break

        n = n + 1
    num_iterations = len(lower_iter_idx)

    huber = []
    for k in range(num_iterations):
        cax = np.arange(lower_iter_idx[k], upper_iter_idx[k]).astype('int')
        # Filter
        outliers_iter, huber_iter = find_huber_outliers(
            x=x[cax],
            y=y[cax],
            sample_weight=boolean_mask[cax],
            fit_intercept=fit_intercept,
            epsilon=epsilon
        )

        outliers_iter = np.logical_and(outliers_iter, boolean_mask[cax])
        # inbounds_iter = np.logical_not(outliers_iter)
        # outliers[cax] = huber_iter.is_outlier(x[cax],y[cax])
        # inbounds[cax] = inbounds_iter
        outliers[cax] = outliers_iter

        huber.append(huber_iter)

    out = {
        'outliers': outliers,
        # 'inbounds': inbounds,
        'lower_iter_idx': lower_iter_idx,
        'upper_iter_idx': upper_iter_idx,
        'huber': huber,
    }
    return out


def find_clearsky_poa(df, lat, lon,
                      irradiance_poa_key='irradiance_poa_o_###',
                      mounting='fixed',
                      tilt=0,
                      azimuth=180,
                      altitude=0):
    loc = Location(lat, lon, altitude=altitude)

    CS = loc.get_clearsky(df.index)

    df['csghi'] = CS.ghi
    df['csdhi'] = CS.dhi
    df['csdni'] = CS.dni

    if mounting.lower() == "fixed":
        sun = get_solarposition(df.index, lat, lon)

        fixedpoa = get_total_irradiance(tilt, azimuth, sun.zenith,
                                        sun.azimuth,
                                        CS.dni, CS.ghi, CS.dhi)

        df['cspoa'] = fixedpoa.poa_global

    if mounting.lower() == "tracking":
        sun = get_solarposition(df.index, lat, lon)

        # default to axis_tilt=0 and axis_azimuth=180

        tracker_data = singleaxis(sun.apparent_zenith,
                                  sun.azimuth,
                                  axis_tilt=tilt,
                                  axis_azimuth=azimuth,
                                  max_angle=50,
                                  backtrack=True,
                                  gcr=0.35)

        track = get_total_irradiance(
            tracker_data['surface_tilt'],
            tracker_data['surface_azimuth'],
            sun.zenith, sun.azimuth,
            CS.dni, CS.ghi, CS.dhi)

        df['cspoa'] = track.poa_global

    # the following code is assuming clear sky poa has been generated per pvlib, aligned in the same
    # datetime index, and daylight savings or any time shifts were previously corrected
    # the inputs below were tuned for POA at a 15 minute frequency
    # note that detect_clearsky has a scaling factor but I still got slightly different results when I scaled measured poa first

    df['poa'] = df[irradiance_poa_key] / df[irradiance_poa_key].quantile(
        0.98) * df.cspoa.quantile(0.98)

    # inputs for detect_clearsky

    measured = df.poa.copy()
    clear = df.cspoa.copy()
    dur = 60
    lower_line_length = -41.416
    upper_line_length = 77.789
    var_diff = .00745
    mean_diff = 80
    max_diff = 90
    slope_dev = 3

    is_clear_results = detect_clearsky(measured.values,
                                       clear.values, df.index,
                                       dur,
                                       mean_diff, max_diff,
                                       lower_line_length,
                                       upper_line_length,
                                       var_diff, slope_dev,
                                       return_components=True)

    clearSeries = pd.Series(index=df.index, data=is_clear_results[0])

    clearSeries = clearSeries.reindex(index=df.index, method='ffill', limit=3)

    return clearSeries
