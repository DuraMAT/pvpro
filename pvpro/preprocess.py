import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from array import array


from pvlib.location import Location
from pvlib.solarposition import get_solarposition
from pvlib.irradiance import get_total_irradiance
from pvlib.tracking import singleaxis
from pvlib.clearsky import detect_clearsky
from pvlib.temperature import sapm_cell_from_module

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model import HuberRegressor
from solardatatools import DataHandler
from pvanalytics.features import clipping

class Preprocessor():

    def __init__(self,
                 df : 'dataframe',
                 system_name : str ='Unknown',
                 voltage_dc_key : str =None,
                 current_dc_key : str =None,
                 temperature_module_key : str =None,
                 temperature_ambient_key : str =None,
                 irradiance_poa_key : str =None,
                 modules_per_string : int =None,
                 parallel_strings : int =None,
                 freq : str ='15min',
                 solver : str ="MOSEK",
                 techonology: str = None,
                 alpha_isc : float =None,
                 cells_in_series : int =None
                 ):

        # Initialize datahandler object.

        self.dh = DataHandler(df)
        self.system_name = system_name
        self.voltage_dc_key = voltage_dc_key
        self.current_dc_key = current_dc_key
        self.temperature_module_key = temperature_module_key
        self.temperature_ambient_key = temperature_ambient_key
        self.irradiance_poa_key = irradiance_poa_key
        self.modules_per_string = modules_per_string
        self.parallel_strings = parallel_strings
        self._ran_sdt = False
        self.freq = freq
        self.solver = solver
        self.technology = techonology
        self.cells_in_series = cells_in_series
        self.alpha_isc = alpha_isc # alpha_isc is in units of A/C

        keys = [self.voltage_dc_key,
                self.current_dc_key,
                self.temperature_module_key,
                self.irradiance_poa_key]

        # Check keys in df.
        for k in keys:
            if not k in self.df.keys():
                raise Exception("""Key '{}' not in dataframe. Check 
                specification of voltage_dc_key, current_dc_key, 
                temperature_module_key and irradiance_poa_key""".format(k)
                                )

        if self.df[self.temperature_module_key].max() > 85:
            warnings.warn("""Maximum module temperature is larger than 85 C. 
            Double check that input temperature is in Celsius, not Farenheight. 
            
            """)

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
    def df(self, value : str):
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

    def run_preprocess_sdt(self,
                correct_tz : bool =False,
                data_sampling : bool =None,
                correct_dst : bool =False,
                fix_shifts : bool =True,
                max_val : bool =None,
                verbose : bool =True):
        """
        Perform "time-consuming" preprocessing steps, using solarDataTool
        """

        if type(data_sampling) != type(None):
            self.dh.data_sampling = data_sampling

        # Run solar-data-tools.
        if correct_dst:
            if verbose:
                print('Fixing daylight savings time shift...')
            self.dh.fix_dst()

        if verbose:
            print('Running solar data tools...')

        # Make normalized power column.
        self.df['power_dc'] = self.df[self.voltage_dc_key] * self.df[
            self.current_dc_key] / self.modules_per_string / self.parallel_strings

        self.dh.run_pipeline(power_col='power_dc',
                            correct_tz=correct_tz,
                            extra_cols=[self.temperature_module_key,
                                        self.irradiance_poa_key,
                                        self.voltage_dc_key,
                                        self.current_dc_key],
                            verbose=verbose,
                            fix_shifts=fix_shifts,
                            max_val=max_val,
                            solver=self.solver)
        self._ran_sdt = True

    def classify_points_sdt(self):

        self.dh.find_clipped_times()

        # Calculate boolean masks
        dh = self.dh
        dh.augment_data_frame(dh.boolean_masks.daytime, 'daytime')
        dh.augment_data_frame(dh.boolean_masks.clipped_times,
                              'clipped_times')
        voltage_fill_nan = np.nan_to_num(
            dh.extra_matrices[self.voltage_dc_key], nan=-9999)
        dh.augment_data_frame(voltage_fill_nan > 0.01 * np.nanquantile(
            dh.extra_matrices[self.voltage_dc_key], 0.98), 'high_v')
        dh.augment_data_frame(
            dh.filled_data_matrix < 0.01 * dh.capacity_estimate,
            'low_p')
        dh.augment_data_frame(dh.daily_flags.no_errors, 'no_errors')
        dh.augment_data_frame(
            np.any([np.isnan(dh.extra_matrices[self.voltage_dc_key]),
                    np.isnan(dh.extra_matrices[self.current_dc_key]),
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
        dh.data_frame_raw['no_errors'] = dh.data_frame_raw[
            'no_errors'].fillna(True, inplace=False)

    def classify_points_pva(self):

        self.find_clipped_times_pva()

        voltage_fill_nan = np.nan_to_num(
            self.df[self.voltage_dc_key], nan=-9999)
        self.df.loc[:,'high_v'] = voltage_fill_nan > 0.01 * np.nanquantile(
            self.df[self.voltage_dc_key], 0.98)

        self.df.loc[:,'missing_data'] = np.logical_or.reduce((
            np.isnan(self.df[self.voltage_dc_key]),
            np.isnan(self.df[self.current_dc_key]),
            np.isnan(self.df[self.irradiance_poa_key]),
            np.isnan(self.df[self.temperature_module_key])))

        self.df.loc[:,'no_errors'] = np.logical_not(self.df['missing_data'])

        power_fill_nan = np.nan_to_num(
            self.df[self.voltage_dc_key] * self.df[self.current_dc_key], nan=1e10)
        self.df.loc[:, 'low_p'] = power_fill_nan < 0.01 * np.nanquantile(
            self.df[self.voltage_dc_key] * self.df[self.current_dc_key], 0.98)

        self.df.loc[:,'daytime'] = np.logical_not(self.df.loc[:, 'low_p'])

    def calculate_cell_temperature(self, delta_T : float =3,
                                   temperature_cell_key : str ='temperature_cell'):
        """
        Set cell temeperature in dataframe.
        """
        # Calculate cell temperature
        self.df.loc[:,temperature_cell_key] = sapm_cell_from_module(
            module_temperature=self.df[self.temperature_module_key],
            poa_global=self.df[self.irradiance_poa_key],
            deltaT=delta_T)

        print("Cell temperature assigned to '{}'".format(temperature_cell_key))

    def find_clipped_times_pva(self):

        # Make normalized power column.
        self.df['power_dc'] = self.df[self.voltage_dc_key] * self.df[
        self.current_dc_key] / self.modules_per_string / self.parallel_strings

        # Find clipped times.
        self.df['clipped_times'] = clipping.geometric(
            ac_power=self.df['power_dc'],
            freq=self.freq)

    def classify_operating_mode(self, voltage: array, current: array,
                            clipped_times : array =None,
                            freq : str ='15min'):
        """
        Parameters
        ----------
        voltage
        currente
        method

        Returns
        -------
        operating_cls : array

            Array of classifications of each time stamp.

            0: System at maximum power point.
            1: System at open circuit conditions.
            2: Clipped or curtailed. DC operating point is not necessarily at MPP.
            -1: No power/inverter off
            -2: Other
        """

        cls = np.zeros(np.shape(voltage)) - 1

        # Inverter on
        cls[np.logical_and(
            voltage > voltage.max() * 0.01,
            current > current.max() * 0.01,
        )] = 0

        # Nighttime, low voltage and irradiance (inverter off)
        cls[voltage < voltage.max() * 0.01] = -1

        # Open circuit condition
        cls[np.logical_and(current < current.max() * 0.01,
                        voltage > voltage.max() * 0.01)] = 1

        if clipped_times is None:
            clipped_times = clipping.geometric(
                ac_power=voltage * current,
                freq=freq)

        # Clipped data.
        cls[clipped_times] = 2

        return cls

    def build_operating_classification(self, df: pd.DataFrame):
        """
        Build array of classifications of each time stamp based on boolean arrays
        provided.

        Parameters
        ----------
        df : dataframe 
            Needs to have fields:

            - 'high_v':

            - 'daytime':

            - 'low_p':

            - 'clipped_times':

            - 'missing_data':

            - 'no_errors':

        Returns
        -------

        operating_cls : array

            integer array of operating_cls

                0: System at maximum power point.
                1: System at open circuit conditions.
                2: Clipped or curtailed. DC operating point is not necessarily at MPP.
                -1: No power/inverter off
                -2: Other


        """
        if isinstance(df, pd.DataFrame):
            operating_cls = np.zeros(len(df), dtype='int')
        else:
            operating_cls = np.zeros(df['high_v'].shape,dtype='int')

        # df.loc[:, 'operating_cls'] = 0
        operating_cls[np.logical_and(
            np.logical_not(df['high_v']),
            np.logical_not(df['daytime']))] = -1

        operating_cls[
            np.logical_and(
                df['high_v'],
                np.logical_or(np.logical_not(df['daytime']), df['low_p'])
            )] = 1

        operating_cls[df['clipped_times']] = 2

        operating_cls[
            np.logical_or(
                df['missing_data'],
                np.logical_not(df['no_errors'])
            )] = -2

        return operating_cls

    def build_operating_cls(self):
        """
        Adds a key to self.df of 'operating_cls'. This field classifies the
        operating point according to the following table:

        0: System at maximum power point.
        1: System at open circuit conditions.
        2: Clipped or curtailed. DC operating point is not necessarily at MPP.
        -1: No power/inverter off
        -2: Other

        """
        if self._ran_sdt:
            for df in [self.dh.data_frame_raw, self.dh.data_frame]:
                df.loc[:,'operating_cls'] = self.build_operating_classification(df)

            self.dh.generate_extra_matrix('operating_cls',
                          new_index=self.dh.data_frame.index)
        else:
            for df in [self.dh.data_frame_raw]:
                df.loc[:, 'operating_cls'] = self.build_operating_classification(df)

    def find_clear_times_sdt(self,
                         min_length : int =2,
                         smoothness_hyperparam : int =5000):
        """
        Find clear times.
        """
        self.dh.find_clear_times(min_length=min_length,
                                 smoothness_hyperparam=smoothness_hyperparam)
        self.dh.augment_data_frame(self.dh.boolean_masks.clear_times,
                                   'clear_time')

    def monotonic(self, y : array, fractional_rate_limit : float =0.05):
        """
        Find times when vector has a run of three increasing values,
        three decreasing values or is changing less than a fractional percent.

        Returns
        -------
        boolean_mask
            True if monotonic or rate of change is less than a fractional limit.

        """
        dP = np.diff(y)
        dP = np.append(dP, dP[-1])
        boolean_mask = np.logical_or.reduce((
            np.logical_and(dP > 0, np.roll(dP, 1) > 0),
            np.logical_and(dP < 0, np.roll(dP, 1) < 0),
            np.abs(dP / y ) < fractional_rate_limit
        ))
        return boolean_mask

    def find_monotonic_times(self, fractional_rate_limit : float =0.05):
        self.df['monotonic'] = self.monotonic(
            self.df[self.voltage_dc_key] * self.df[self.current_dc_key],
            fractional_rate_limit=fractional_rate_limit)

    def find_huber_outliers(self, x : array, y : array, sample_weight : bool =None,
                        fit_intercept : bool =True,
                        epsilon : float =2.5):
        """
        Identify outliers based on a linear fit of current at maximum power point
        to plane-of-array irradiance.
        """
        if sample_weight is None:
            sample_weight = np.ones_like(x)

        mask = np.logical_and(np.isfinite(x), np.isfinite(y))

        if np.sum(mask) <= 2:
            print('Need more than two points for linear regression.')
            return [], []

        huber = HuberRegressor(epsilon=epsilon,
                            fit_intercept=fit_intercept)
        huber.fit(np.atleast_2d(x[mask]).transpose(), y[mask],
                sample_weight=sample_weight[mask])


        def is_outlier(x : array, y : array):
            X = np.atleast_2d(x).transpose()
            residual = np.abs(
                y - safe_sparse_dot(X, huber.coef_) - huber.intercept_)
            outliers = residual > huber.scale_ * huber.epsilon
            return outliers

        def is_inbounds(x : array, y : array):
            X = np.atleast_2d(x).transpose()
            residual = np.abs(
                y - safe_sparse_dot(X, huber.coef_) - huber.intercept_)
            outliers = residual <= huber.scale_ * huber.epsilon
            return outliers

        outliers = is_outlier(x, y)

        huber.is_outlier = is_outlier
        huber.is_inbounds = is_inbounds

        return outliers, huber

    def find_linear_model_outliers_timeseries(self, x : array, y : array,
                                            boolean_mask : array = None,
                                            fit_intercept : bool = True,
                                            points_per_iteration : int =20000,
                                            epsilon : float =2.5,
                                            ):
        outliers = np.zeros_like(x).astype('bool')

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
                upper_lim = np.nan

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
            outliers_iter, huber_iter = self.find_huber_outliers(
                x=x[cax],
                y=y[cax],
                sample_weight=boolean_mask[cax],
                fit_intercept=fit_intercept,
                epsilon=epsilon
            )

            outliers_iter = np.logical_and(outliers_iter, boolean_mask[cax])
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

    def find_clearsky_poa(self, df : pd.DataFrame, lat : float, lon : float,
                        irradiance_poa_key : str ='irradiance_poa_o_###',
                        mounting : str ='fixed',
                        tilt : float =0,
                        azimuth : float =180,
                        altitude : float =0):
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

    def find_current_irradiance_outliers(self,
                                         boolean_mask : array =None,
                                         poa_lower_lim : float =10,
                                         epsilon : float =2.0,
                                         points_per_iteration : int =2000):


        filter = np.logical_and.reduce(
            (self.df['operating_cls'] == 0,
             self.df[self.irradiance_poa_key] > poa_lower_lim)
        )

        if boolean_mask is None:
            boolean_mask = filter
        else:
            boolean_mask = np.logical_and(boolean_mask, filter)

        current_irradiance_filter = self.find_linear_model_outliers_timeseries(
            x=self.df[self.irradiance_poa_key],
            y=self.df[self.current_dc_key] / self.parallel_strings,
            boolean_mask=boolean_mask,
            fit_intercept=False,
            epsilon=epsilon,
            points_per_iteration=points_per_iteration)

        self.df['current_irradiance_outliers'] = current_irradiance_filter[
            'outliers']

        return current_irradiance_filter

    def find_temperature_voltage_outliers(self,
                                         boolean_mask : bool =None,
                                         poa_lower_lim : float =10,
                                         voltage_lower_lim : float = 10,
                                         epsilon : float =2.0,
                                         points_per_iteration : int =2000):

        filter = np.logical_and.reduce(
            (self.df['operating_cls'] == 0,
             self.df[self.irradiance_poa_key] > poa_lower_lim,
             self.df[self.voltage_dc_key]/self.modules_per_string > voltage_lower_lim)
        )


        if boolean_mask is None:
            boolean_mask = filter
        else:
            boolean_mask = np.logical_and(boolean_mask, filter)
        if 'temperature_cell' not in self.df:
            raise Exception("""Need 'temperature_cell' in dataframe, 
            run 'calculate_cell_temperature' first. 
            
            """)

        voltage_temperature_filter = self.find_linear_model_outliers_timeseries(
            x=self.df['temperature_cell'],
            y=self.df[self.voltage_dc_key] / self.modules_per_string,
            boolean_mask=boolean_mask,
            fit_intercept=True,
            epsilon=epsilon,
            points_per_iteration=points_per_iteration)

        self.df['voltage_temperature_outliers'] = voltage_temperature_filter[
            'outliers']

        return voltage_temperature_filter


   