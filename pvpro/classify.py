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
        -1: Unclassified
        0: System at maximum power point.
        1: System at open circuit conditions.
        2: Low irradiance nighttime.
        3: Clipped/curtailed operation. Not necessarily at mpp.

    """

    cls = np.zeros(np.shape(voltage)) - 1

    # Inverter on
    cls[np.logical_and(
        voltage > voltage.max() * 0.01,
        current > current.max() * 0.01,
    )] = 0

    # Nighttime, low voltage and irradiance.
    cls[np.logical_and(current < current.max() * 0.01,
                       voltage < voltage.max() * 0.01)] = 2

    # Inverter off
    cls[np.logical_and(current < current.max() * 0.01,
                       voltage > voltage.max() * 0.01)] = 1

    # Clipped data. Easy algorithm.
    cls[current * voltage > power_clip] = 3

    return cls

