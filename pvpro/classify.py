import numpy as np

def classify_operating_mode(voltage, current,
                            power_clip=np.inf,
                            method='fraction'):
    """

    Parameters
    ----------
    voltage
    current
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

    # Clipped data. Easy algorithm.
    cls[current * voltage > power_clip] = 2

    return cls

