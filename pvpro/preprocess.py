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
                 solver : str ="MOSEK"
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
                            power_clip=np.inf,
                            method='fraction',
                            clipped_times=None,
                            freq='15min'):
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

    def build_operating_classification(self, df: 'dataframe or dict'):
        """
        Build array of classifications of each time stamp based on boolean arrays
        provided.

        Parameters
        ----------
        df : dataframe or dict
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
        self.df['monotonic'] = monotonic(
            self.df[self.voltage_dc_key] * self.df[self.current_dc_key],
            fractional_rate_limit=fractional_rate_limit)

    def find_huber_outliers(x : array, y : array, sample_weight : bool =None,
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

    def find_linear_model_outliers_timeseries(x : array, y : array,
                                            boolean_mask : bool =None,
                                            fit_intercept : bool =True,
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
            outliers_iter, huber_iter = find_huber_outliers(
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

    def find_clearsky_poa(df : 'dataframe', lat : float, lon : float,
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
                                         boolean_mask : bool =None,
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

        current_irradiance_filter = find_linear_model_outliers_timeseries(
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

        voltage_temperature_filter = find_linear_model_outliers_timeseries(
            x=self.df['temperature_cell'],
            y=self.df[self.voltage_dc_key] / self.modules_per_string,
            boolean_mask=boolean_mask,
            fit_intercept=True,
            epsilon=epsilon,
            points_per_iteration=points_per_iteration)

        self.df['voltage_temperature_outliers'] = voltage_temperature_filter[
            'outliers']

        return voltage_temperature_filter

    def plot_operating_cls(self, figsize : tuple =(12, 6)):

        if not 'operating_cls' in self.dh.extra_matrices:
            raise Exception("""Must call 'run_preprocess_sdt' first to use 
            this visualization.""")

        fig = plt.figure(figsize=figsize)

        # Build colormap
        colors = sns.color_palette("Paired")[:5]
        n_bins = 5  # Discretizes the interpolation into bins
        cmap_name = 'my_map'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        plt.imshow(self.dh.extra_matrices['operating_cls'], aspect='auto',
                   interpolation='none',
                   cmap=cmap,
                   vmin=-2.5, vmax=2.5)
        plt.colorbar()
        return fig

    """
    off-MPP functions

    """
    def detect_off_MPP(self, pvp, boolean_mask : array = None):

        """
        detect off-MPP based on Pmp error

        return
        ------
        off-MPP bool array
        
        """
        if boolean_mask:
            df=pvp.df[boolean_mask]
        else:
            df=pvp.df

        p_plot=pvp.p0

        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings

        # calculate error
        v_esti, i_esti = pvp.single_diode_predict(
            effective_irradiance=df[pvp.irradiance_poa_key][mask],
            temperature_cell=df[pvp.temperature_cell_key][mask],
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)
        rmse_vmp = mean_squared_error(v_esti, vmp)/37
        rmse_imp = mean_squared_error(i_esti, imp)/8.6

        # Pmp error
        pmp_error = abs(vmp*imp - v_esti*i_esti)
        vmp_error = abs(vmp-v_esti)
        imp_error = abs(imp-i_esti)

        # detect off-mpp and calculate off-mpp percentage
        offmpp = pmp_error>np.nanmean(pmp_error)+np.std(pmp_error)
        offmpp_ratio = offmpp.sum()/pmp_error.size*100  

        return offmpp
  
    def plot_Pmp_error_vs_time(self, pvp, boolean_mask : array, points_show : array= None, figsize : tuple =[4,3], 
                                 sys_name : str = None):

        """
        Plot Pmp error vs time, where the at-MPP and off-MPP points are highlighted
        
        """
        if points_show:
            points_show_bool = np.full(boolean_mask.sum(), False)
            points_show_bool[points_show] = True
            df=pvp.df[boolean_mask][points_show_bool]
        else:
            df=pvp.df[boolean_mask]

        p_plot=pvp.p0

        fig, ax = plt.subplots(figsize=figsize)

        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings

        # calculate error
        v_esti, i_esti = pvp.single_diode_predict(
            effective_irradiance=df[pvp.irradiance_poa_key][mask],
            temperature_cell=df[pvp.temperature_cell_key][mask],
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)
        rmse_vmp = mean_squared_error(v_esti, vmp)/37
        rmse_imp = mean_squared_error(i_esti, imp)/8.6

        # Pmp error
        pmp_error = abs(vmp*imp - v_esti*i_esti)
        vmp_error = abs(vmp-v_esti)
        imp_error = abs(imp-i_esti)

        # Plot At-MPP points
        ax.scatter(df.index[mask], pmp_error, s =1, color ='#8ACCF8', label = 'At-MPP')

        # detect off-mpp and calculate off-mpp percentage
        offmpp = pmp_error>np.nanmean(pmp_error)+np.std(pmp_error)
        offmpp_ratio = offmpp.sum()/pmp_error.size*100  
        plt.text(df.index[0], 240, 'Off-MPP ratio:\
                        \n{}%'.format(round(offmpp_ratio,2)),
                    fontsize=12)
        
        # Plot off-MPP points
        ax.scatter(df.index[mask][offmpp], pmp_error[offmpp], s =1, color ='#FFA222', label = 'Off-MPP')

        # plot mean Pmp error line
        ax.plot([df.index[mask][0], df.index[mask][-1]], [np.nanmean(pmp_error)]*2, 
                    '--', linewidth = 1, color='#0070C0', label = 'Mean Pmp error')

        import matplotlib.dates as mdates
        # h_fmt = mdates.DateFormatter('%y-%m')
        h_fmt = mdates.DateFormatter('%Y')
        xloc = mdates.YearLocator(1)
        ax.xaxis.set_major_locator(xloc)
        ax.xaxis.set_major_formatter(h_fmt)

        # fig.autofmt_xdate()
        plt.ylim([0, 300])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.ylabel('Pmp error (W)', fontsize=12, fontweight = 'bold')
        lgnd = plt.legend()
        lgnd.legendHandles[0]._sizes = [20]
        lgnd.legendHandles[1]._sizes = [20]
        plt.title(sys_name, fontsize=13, fontweight = 'bold')

        plt.gcf().set_dpi(150)
        plt.show()

    def deconvolve_Pmp_error_on_V_I (self, pvp, boolean_mask : array, points_show : array = None, figsize : tuple =[4.5,3], 
                                sys_name : str = None, date_text : str = None):

        points_show_bool = np.full(boolean_mask.sum(), False)
        points_show_bool[points_show] = True
        df=pvp.df[boolean_mask][points_show_bool]
        
        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings
        G = df[pvp.irradiance_poa_key][mask]
        Tm = df[pvp.temperature_cell_key][mask]

        # estimate
        v_esti, i_esti = pvp.single_diode_predict(
            effective_irradiance=G,
            temperature_cell=Tm,
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)

        # error
        pmp_error = abs(vmp*imp - v_esti*i_esti) 
        vmp_error = abs(vmp-v_esti)
        imp_error = abs(imp-i_esti)
        pmp_error = pmp_error + vmp_error*imp_error

        # contribution
        con_V = vmp_error*i_esti/pmp_error*100
        con_I = imp_error*v_esti/pmp_error*100

        fig, ax = plt.subplots(figsize=figsize)
        xtime = df.index[mask]

        # ax.fill_between(xtime, np.ones_like(con_V)*100, 0, alpha=1, color='#0070C0', edgecolor = 'white', linewidth = 2, label = 'Error of Vmp')
        ax.fill_between(xtime, con_I+con_V, 0, alpha=1, color='#0070C0', edgecolor = 'white', linewidth = 2, label = 'Error of Vmp')
        ax.fill_between(xtime, con_I, 0, alpha=1, color='#92D050', edgecolor = 'white', linewidth = 2, label = 'Error of Imp')

        import matplotlib.dates as mdates
        hours = mdates.HourLocator(interval = 1)
        h_fmt = mdates.DateFormatter('%Hh')
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)

        # text
        if not date_text:
            datetext = df.index[mask][0].strftime("%Y-%m-%d")
        text_show = sys_name + '\n' + datetext
        ax.text(xtime[1], 10, text_show)

        ax.tick_params(labelsize=13)
        ax.tick_params(labelsize=13)
        ax.set_xlabel('Time', fontsize=13)
        ax.set_ylabel('Deconvolution\n of Pmp error (%)', fontsize=13, fontweight = 'bold')
        plt.ylim([0,100])
        plt.legend(loc=7)
        plt.gcf().set_dpi(120)
        plt.show()

    def plot_Vmp_Imp_scatters_Pmp_error(self, pvp, boolean_mask : array, points_show : array = None, figsize : tuple =[4,3], show_only_offmpp : bool = False, 
                                sys_name : str = None, date_show : str = None):

        """
        Plot relative error (RE) of Vmp vs RE of Imp as scatters.
        The color of scatters corresponds to the RE of Pmp.
        
        """
        if points_show:
            points_show_bool = np.full(boolean_mask.sum(), False)
            points_show_bool[points_show] = True
            df=pvp.df[boolean_mask][points_show_bool]
        else:
            df=pvp.df[boolean_mask]
        p_plot=pvp.p0
        
        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings

        # calculate error
        v_esti, i_esti = pvp.single_diode_predict(
            effective_irradiance=df[pvp.irradiance_poa_key][mask],
            temperature_cell=df[pvp.temperature_cell_key][mask],
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)
        rmse_vmp = mean_squared_error(v_esti, vmp)/37
        rmse_imp = mean_squared_error(i_esti, imp)/8.6

        # Pmp error
        pmp_error = abs(vmp*imp - v_esti*i_esti)
        vmp_error = abs(vmp-v_esti)
        imp_error = abs(imp-i_esti)

        # calculate off-mpp percentage
        msk = np.full(pmp_error.size, True)
        if show_only_offmpp:
            msk = (pmp_error>np.nanmean(pmp_error)+np.std(pmp_error) ) & (pmp_error<300)

        # plot
        fig, ax = plt.subplots(figsize=figsize)

        h_sc = plt.scatter(vmp_error[msk]/37*100, imp_error[msk]/8.6*100, cmap='jet',
                s=10,  alpha = 0.8, c=pmp_error[msk])
                            
        pcbar = plt.colorbar(h_sc)
        pcbar.set_label('Pmp error (W)', fontsize = 13)

        if not date_show:
            date_show = df.index[mask][0].strftime("%Y-%m-%d")

        text_show = sys_name + '\n' + date_show
        plt.text (35,85, text_show)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlim([0,80])
        plt.ylim([0,100])
        plt.xlabel('RE of Vmp (%)', fontsize=13)
        plt.ylabel('RE of Imp (%)', fontsize=13)
        plt.title('Distribution of off-MPP points', fontweight = 'bold', fontsize=13)
        plt.gcf().set_dpi(120)
        plt.show()

    def plot_Vmp_Tm_Imp_G_vs_time (self, pvp, boolean_mask : array, points_show : array = None, figsize : tuple =[5,6]):

        points_show_bool = np.full(boolean_mask.sum(), False)
        points_show_bool[points_show] = True
        df=pvp.df[boolean_mask][points_show_bool]

        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings

        # calculate error
        v_esti, i_esti = pvp.single_diode_predict(
            effective_irradiance=df[pvp.irradiance_poa_key][mask],
            temperature_cell=df[pvp.temperature_cell_key][mask],
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)
        
        fig, ax = plt.subplots(2,1,figsize=figsize, sharex = True)

        """ plot Imp and G """

        ax11 = ax[0]
        ax12 = ax11.twinx()
        
        xtime = df.index[mask]

        ## plot G in right
        lns11 = ax11.fill_between(xtime, df[irradiance_poa_key][mask], 0, 
                        alpha=0.3, color='#FF95C2',
                        zorder = 2, label = 'G')
        ax11.yaxis.tick_right()
        ax11.yaxis.set_label_position("right")

        ## plot Imp in left
        lns12, = ax12.plot(xtime, imp, '-o', color = 'deepskyblue', zorder = 2.5, label = 'Measured Imp')
        lns13, = ax12.plot(xtime, i_esti, '--o',zorder = 3,label = 'Estimated Imp')
        ax12.yaxis.tick_left()
        ax12.yaxis.set_label_position("left")

        ax11.grid(linestyle = '--')
        ax11.tick_params(labelsize=13)
        ax12.tick_params(labelsize=13)
        ax12.set_ylabel('Imp (A)', fontsize=13, color = np.array([24,116,205])/256, fontweight = 'bold')
        ax11.set_ylabel('Irradiance (${W/m^2}$)', fontsize=13, color = '#C47398', fontweight = 'bold')
        ax11.set_title(' Vmp and Imp on {}'.format(xtime[0].strftime('%Y-%m-%d')), 
                    fontweight = 'bold', fontsize=13)

        # combine legends

        lns1 = (lns12, lns13, lns11)
        labs1 = [l.get_label() for l in lns1]
        ax12.legend(lns1, labs1, loc=1)

        """ plot Vmp and Tm """

        ax21 = ax[1]
        ax22 = ax21.twinx()
        
        xtime = df.index[mask]
        
        ## plot Tm in right
        lns21 = ax21.fill_between(xtime, df[temperature_module_key][mask], 0, 
                        alpha=0.4, color='#FFC000', edgecolor = None,
                        zorder = 2, label = 'Tm')
        ax21.yaxis.tick_right()
        ax21.yaxis.set_label_position("right")

        ## plot Imp in left
        lns22, = ax22.plot(xtime, vmp, '-o',zorder = 2.5, label = 'Measured Vmp', color = '#92D050')
        lns23, = ax22.plot(xtime, v_esti, '--o', color= '#009847',zorder = 3,label = 'Estimated Vmp')
        ax22.yaxis.tick_left()
        ax22.yaxis.set_label_position("left")
        
        ax21.grid(linestyle = '--')
        ax21.tick_params(labelsize=13)
        ax22.tick_params(labelsize=13)
        ax21.set_xlabel('Time', fontsize=13)
        ax22.set_ylabel('Vmp (A)', fontsize=13, color = '#009847', fontweight = 'bold')
        ax21.set_ylabel('Tm (℃)', fontsize=13, color = '#D8A402', fontweight = 'bold')
        

        import matplotlib.dates as mdates
        hours = mdates.HourLocator(interval = 1)
        h_fmt = mdates.DateFormatter('%Hh')
        ax21.xaxis.set_major_locator(hours)
        ax21.xaxis.set_major_formatter(h_fmt)

        # combine legends
        lns2 = (lns22, lns23, lns21)
        labs2 = [l.get_label() for l in lns2]
        ax22.legend(lns2, labs2, loc=7)


        plt.gcf().set_dpi(120)
        plt.show()

    def plot_Vmp_vs_Tm_Imp_vs_G (self, pvp, boolean_mask : array, points_show : array = None, figsize : tuple =[4,6]):

        points_show_bool = np.full(boolean_mask.sum(), False)
        points_show_bool[points_show] = True
        df=pvp.df[boolean_mask][points_show_bool]

        mask = np.array(df['operating_cls'] == 0)
        vmp = np.array(df.loc[mask, pvp.voltage_key]) / pvp.modules_per_string
        imp = np.array(df.loc[mask, pvp.current_key]) / pvp.parallel_strings
        G = df[pvp.irradiance_poa_key][mask]
        Tm = df[pvp.temperature_cell_key][mask]

        # estimate
        v_esti, i_esti = pvp.single_diode_predict(
            effective_irradiance=G,
            temperature_cell=Tm,
            operating_cls=np.zeros_like(df[pvp.irradiance_poa_key][mask]),
            params=p_plot)

        # error
        pmp_error = abs(vmp*imp - v_esti*i_esti)
        RE_vmp = abs(vmp-v_esti)/37*100
        RE_imp = abs(imp-i_esti)/8.6*100

        fig, ax = plt.subplots(2,1,figsize=figsize)

        ax1 = ax[0]
        ax1.scatter(G, RE_imp)
        ax1.grid(linestyle = '--')
        ax1.tick_params(labelsize=13)
        ax1.tick_params(labelsize=13)
        ax1.set_ylabel('RE_Imp (%)', fontsize=13, fontweight = 'bold')
        ax1.set_xlabel('G (${W/m^2}$)', fontsize=13, fontweight = 'bold')
        ax1.set_title('RE_Imp vs G', fontweight = 'bold', fontsize=13)

        ax2 = ax[1]
        ax2.scatter(Tm, RE_vmp, color = '#009847')
        ax2.grid(linestyle = '--')
        ax2.tick_params(labelsize=13)
        ax2.tick_params(labelsize=13)
        ax2.set_ylabel('RE_Vmp (%)', fontsize=13, fontweight = 'bold')
        ax2.set_xlabel('Tm (℃)', fontsize=13, fontweight = 'bold')
        ax2.set_title(' RE_Vmp vs Tm', fontweight = 'bold', fontsize=13)

        plt.gcf().set_dpi(120)
        plt.tight_layout()
        plt.show()


