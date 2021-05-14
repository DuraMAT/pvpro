
import numpy as np
from tqdm import tqdm
import pandas as pd

from pvlib.location import Location
from pvlib.solarposition import get_solarposition
from pvlib.irradiance import get_total_irradiance
from pvlib.tracking import singleaxis
from pvlib.clearsky import detect_clearsky
from pvlib.temperature import sapm_cell_from_module
import pandas as pd

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model import HuberRegressor
from solardatatools import DataHandler

from pvpro.classify import classify_operating_mode

import warnings

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
        return [], []

    # X = np.atleast_2d(x).transpose()
    # y = np.array(y)

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
        'boolean_mask': boolean_mask
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


class Preprocessor():

    def __init__(self,
                 df,
                 system_name='Unknown',
                 voltage_key=None,
                 current_key=None,
                 temperature_module_key=None,
                 temperature_ambient_key=None,
                 irradiance_poa_key=None,
                 modules_per_string=None,
                 parallel_strings=None,
                 freq=None,
                 ):

        # Initialize datahandler object.

        self.dh = DataHandler(df)

        # self.df = df
        self.system_name = system_name
        # self.use_clear_times = use_clear_times

        self.voltage_key = voltage_key
        self.current_key = current_key
        # self.power_key = power_key
        self.temperature_module_key = temperature_module_key
        self.temperature_ambient_key = temperature_ambient_key
        self.irradiance_poa_key = irradiance_poa_key
        self.modules_per_string = modules_per_string
        self.parallel_strings = parallel_strings

        if freq is None:
            timedelta_minutes = np.median(np.diff(df.index)) / np.timedelta64(1,
                                                                              's') / 60
            freq = '{:.0f}min'.format(timedelta_minutes)
            print('Autodetected freq: {}'.format(freq))

    @property
    def df(self):
        """
        Store dataframe inside the DataHandler.

        Returns
        -------
        df : dataframe
            Time-series data
        """
        return self.dh.data_frame_raw

    @df.setter
    def df(self, value):
        """
        Set Dataframe by setting the version inside datahandler.

        Parameters
        ----------
        value : dataframe
            Time-series data

        Returns
        -------

        """
        self.dh.data_frame_raw = value

    def calculate_cell_temperature(self,delta_T=3):
        """
        Set cell temeperature in dataframe.

        Todo: move this functionality to preprocessing.

        Returns
        -------

        """
        # Calculate cell temperature
        self.df.loc[:, 'temperature_cell'] = sapm_cell_from_module(
            module_temperature=self.df[self.temperature_module_key],
            poa_global=self.df[self.irradiance_poa_key],
            deltaT=delta_T)

    def simulation_setup(self):
        """
        Perform "quick" preprocessing steps.


        Returns
        -------

        """

        # Remove nan from df.
        keys = [self.voltage_key,
                self.current_key,
                self.temperature_module_key,
                self.irradiance_poa_key]

        for k in keys:
            if not k in self.df.keys():
                raise Exception(
                    'Key "{}" not in dataframe. Check specification of '
                    'voltage_key, current_key, temperature_module_key and '
                    'irradiance_poa_key'.format(
                        k))

        # Make normalized power column.
        self.df['power_dc'] = self.df[self.voltage_key] * self.df[
            self.current_key] / self.modules_per_string / self.parallel_strings

        # Make cell temp column
        self.calculate_cell_temperature()



    def run_preprocess(self,
                       correct_tz=True,
                       data_sampling=None,
                       correct_dst=False,
                       fix_shifts=True,
                       classification_method='solar-data-tools',
                       max_val=None,
                       verbose=True):
        """
        Perform "time-consuming" preprocessing steps



        Parameters
        ----------
        correct_tz
        data_sampling
        run_solar_data_tools

        Returns
        -------

        """
        self.simulation_setup()
        if self.df[self.temperature_module_key].max() > 85:
            warnings.warn(
                'Maximum module temperature is larger than 85 C. Double check that input temperature is in Celsius, not Farenheight.')

        if type(data_sampling) != type(None):
            self.dh.data_sampling = data_sampling

        # Run solar-data-tools.

        if correct_dst:
            if verbose:
                print('Fixing daylight savings time shift...')
            self.dh.fix_dst()

        if verbose:
            print('Running solar data tools...')

        self.dh.run_pipeline(power_col='power_dc',
                             correct_tz=correct_tz,
                             extra_cols=[self.temperature_module_key,
                                         self.irradiance_poa_key,
                                         self.voltage_key,
                                         self.current_key],
                             verbose=False,
                             fix_shifts=fix_shifts,
                             max_val=max_val)

        if classification_method.lower() == 'solar-data-tools':
            self.dh.find_clipped_times()
            # Calculate boolean masks
            dh = self.dh
            dh.augment_data_frame(dh.boolean_masks.daytime, 'daytime')
            dh.augment_data_frame(dh.boolean_masks.clipped_times,
                                  'clipped_times')
            voltage_fill_nan = np.nan_to_num(
                dh.extra_matrices[self.voltage_key], nan=-9999)
            dh.augment_data_frame(voltage_fill_nan > 0.01 * np.nanquantile(
                dh.extra_matrices[self.voltage_key], 0.98), 'high_v')
            dh.augment_data_frame(
                dh.filled_data_matrix < 0.01 * dh.capacity_estimate,
                'low_p')
            dh.augment_data_frame(dh.daily_flags.no_errors, 'no_errors')
            dh.augment_data_frame(
                np.any([np.isnan(dh.extra_matrices[self.voltage_key]),
                        np.isnan(dh.extra_matrices[self.current_key]),
                        np.isnan(dh.extra_matrices[self.irradiance_poa_key]),
                        np.isnan(
                            dh.extra_matrices[self.temperature_module_key])],
                       axis=0),
                'missing_data')

            dh.data_frame_raw['missing_data'] = dh.data_frame_raw[
                'missing_data'].fillna(True, inplace=False)
            dh.data_frame_raw['low_p'] = dh.data_frame_raw['low_p'].fillna(True,
                                                                           inplace=False)
            dh.data_frame_raw['high_v'] = dh.data_frame_raw['high_v'].fillna(
                False, inplace=False)
            dh.data_frame_raw['daytime'] = dh.data_frame_raw['daytime'].fillna(
                False, inplace=False)
            dh.data_frame_raw['clipped_times'] = dh.data_frame_raw[
                'clipped_times'].fillna(False, inplace=False)

            # Apply operating class labels

            # 0: System at maximum power point.
            # 1: System at open circuit conditions.
            # 2: Clipped or curtailed. DC operating point is not necessarily at MPP.
            # -1: No power/inverter off
            # -2: Other

            for df in [dh.data_frame_raw, dh.data_frame]:
                df.loc[:, 'operating_cls'] = 0
                df.loc[np.logical_and(
                    np.logical_not(df['high_v']),
                    np.logical_not(df['daytime'])
                ), 'operating_cls'] = -1
                df.loc[np.logical_and(
                    df['high_v'],
                    np.logical_or(np.logical_not(df['daytime']), df['low_p'])
                ), 'operating_cls'] = 1
                df.loc[df['clipped_times'], 'operating_cls'] = 2
                df.loc[np.logical_or(
                    df['missing_data'],
                    np.logical_not(df['no_errors'])
                ), 'operating_cls'] = -2
            # Create matrix view of operating class labels for plotting
            dh.generate_extra_matrix('operating_cls',
                                     new_index=dh.data_frame.index)

        elif classification_method.lower() == 'simple':
            self.df['operating_cls'] = classify_operating_mode(
                voltage=self.df[self.voltage_key],
                current=self.df[self.current_key],
                freq=self.freq
            )
        else:
            raise Exception(
                '`classification_method` must be "solar-data-tools" or "simple"')
