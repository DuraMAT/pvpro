import numpy as np
from pvanalytics.features import clipping
import pandas as pd

def classify_operating_mode(voltage, current,
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


def build_operating_cls(df):
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
