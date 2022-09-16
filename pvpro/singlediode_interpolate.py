"""After a lot of testing, figured out that this method is not faster than
the 'newton' method of pvlib.singlediode.

"""

import numpy as np
from pvlib.pvsystem import singlediode
from tqdm import tqdm
import os
from scipy.interpolate import RegularGridInterpolator
from pvlib.singlediode import _lambertw_v_from_i, _lambertw_i_from_v

_interpolator_filename = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'singlediode_precalculated_values.npz')


def build_interpolator():
    Io_p_list = np.logspace(-16, -2, 25)
    Io_p_log_list = np.log(Io_p_list)

    Rs_p_list = np.linspace(0, 50, 26)
    Rsh_p_list = np.concatenate((np.array([1e-10]),
                                 np.logspace(-2, 1, 4, endpoint=False),
                                 np.logspace(1, 3, 15, endpoint=False),
                                 np.logspace(3, 6, 8, endpoint=True),
                                 np.array([1e10]),
                                 ))

    v_mp_p = np.zeros((len(Io_p_list), len(Rs_p_list), len(Rsh_p_list)))
    i_mp_p = np.zeros((len(Io_p_list), len(Rs_p_list), len(Rsh_p_list)))
    v_oc_p = np.zeros((len(Io_p_list), len(Rs_p_list), len(Rsh_p_list)))
    i_sc_p = np.zeros((len(Io_p_list), len(Rs_p_list), len(Rsh_p_list)))

    for j in tqdm(range(len(Io_p_list))):
        for k in range(len(Rs_p_list)):
            # for l in range(len(Rsh_p_list)):
            out = singlediode(photocurrent=1,
                              saturation_current=Io_p_list[j],
                              resistance_series=Rs_p_list[k],
                              resistance_shunt=Rsh_p_list,
                              nNsVth=1,
                              method='brentq'
                              )
            v_mp_p[j, k, :] = out['v_mp']
            i_mp_p[j, k, :] = out['i_mp']
            i_sc_p[j, k, :] = out['i_sc']
            v_oc_p[j, k, :] = out['v_oc']


    v_mp_p[v_mp_p<0] = 0
    i_mp_p[i_mp_p<0] = 0
    i_sc_p[i_sc_p<0] = 0
    v_oc_p[v_oc_p<0] = 0
    np.savez_compressed(_interpolator_filename,
                        Io_p_list=Io_p_list,
                        Rs_p_list=Rs_p_list,
                        Rsh_p_list=Rsh_p_list,
                        v_mp_p=v_mp_p,
                        i_mp_p=i_mp_p,
                        v_oc_p=v_oc_p,
                        i_sc_p=i_sc_p)




if not os.path.exists(_interpolator_filename):
    print('No interpolator data found, making file now...')
    build_interpolator()

_interpolator_data = np.load(_interpolator_filename)

_xyz = (np.log(_interpolator_data['Io_p_list']),
        _interpolator_data['Rs_p_list'],
        np.log(_interpolator_data['Rsh_p_list']))

_v_mp_p_interpolator = RegularGridInterpolator(
    _xyz,
    _interpolator_data['v_mp_p'],
    bounds_error=True,
    method='linear',
    fill_value=np.nan)
_v_oc_p_interpolator = RegularGridInterpolator(
    _xyz,
    _interpolator_data['v_oc_p'],
    bounds_error=True,
    method='linear',
    fill_value=np.nan)
_i_mp_p_interpolator = RegularGridInterpolator(
    _xyz,
    _interpolator_data['i_mp_p'],
    bounds_error=True,
    method='linear',
    fill_value=np.nan)




def singlediode_fast(photocurrent,
                     saturation_current,
                     resistance_series,
                     resistance_shunt,
                     nNsVth,
                     calculate_voc=False):
    io_p_log_max = _v_mp_p_interpolator.grid[0][-1]
    #
    io_p_log_min = _v_mp_p_interpolator.grid[0][0]
    # low_photocurrent = photocurrent*np.exp(io_p_log_min) < saturation_current
    # photocurrent[low_photocurrent] = saturation_current*np.exp(io_p_log_min)


    # io_p_log = np.atleast_2d(
    #     np.log(saturation_current / photocurrent).flatten())
    io_p_log = np.atleast_2d(
        np.where(photocurrent>1e-6, np.log(saturation_current / photocurrent), io_p_log_max).flatten())
    Rs_p = np.atleast_2d((resistance_series * photocurrent / nNsVth).flatten())
    Rsh_p_log = np.atleast_2d(np.log(resistance_shunt * photocurrent / nNsVth).flatten())

    # # Put in bounds.
    # io_p_log_min = _v_mp_p_interpolator.grid[0][0]
    # io_p_log_max = _v_mp_p_interpolator.grid[0][-1]
    # Rs_p_min = _v_mp_p_interpolator.grid[1][0]
    # Rs_p_max = _v_mp_p_interpolator.grid[1][-1]
    # Rsh_p_min = _v_mp_p_interpolator.grid[2][0]
    # Rsh_p_max = _v_mp_p_interpolator.grid[2][-1]
    #
    # io_p_log[io_p_log<io_p_log_min]=io_p_log_min
    # io_p_log[io_p_log>io_p_log_max]=io_p_log_max
    #
    # Rs_p[Rs_p<Rs_p_min]=Rs_p_min
    # Rs_p[Rs_p>Rs_p_max]=Rs_p_max
    #
    # Rsh_p[Rsh_p < Rsh_p_min] = Rsh_p_min
    # Rsh_p[Rsh_p > Rsh_p_max] = Rsh_p_max


    x = np.concatenate((io_p_log, Rs_p, Rsh_p_log), axis=0).transpose()


    try:
        v_mp_p = _v_mp_p_interpolator(x)
    except Exception as e:

        print('Range io_p_log: {:.2f} {:.2f}'.format(io_p_log.min(), io_p_log.max()))
        print('Allowed io_p_log range: {:.2f} {:.2f}'.format(
            _v_mp_p_interpolator.grid[0][0],
              _v_mp_p_interpolator.grid[0][-1]))

        print('Range Rs: {:.2f} {:.2f}'.format(Rs_p.min(), Rs_p.max()))
        print('Allowed Rs range: {:.2f} {:.2f}'.format(
            _v_mp_p_interpolator.grid[1][0],
              _v_mp_p_interpolator.grid[1][-1]))

        print('Range Rsh_p_log: {:.2f} {:.2f}'.format(Rsh_p.min(), Rsh_p.max()))
        print('Allowed Rsh_p range: {:.2f} {:.2f}'.format(
            _v_mp_p_interpolator.grid[2][0],
              _v_mp_p_interpolator.grid[2][-1]))

        raise e

    v_mp = v_mp_p * nNsVth

    # i_mp_p = _i_mp_p_interpolator(x)
    # i_mp = i_mp_p * photocurrent


    # Another way is to calculate the other mpp point using lambert w.
    # v_mp = _lambertw_v_from_i(resistance_shunt, resistance_series, nNsVth, i_mp,
    #                    saturation_current, photocurrent)
    i_mp = _lambertw_i_from_v(resistance_shunt, resistance_series, nNsVth, v_mp,
                       saturation_current, photocurrent)


    # low_photocurrent = (io_p_log< io_p_log_min).flatten()
    # v_mp[low_photocurrent]=0
    # i_mp[low_photocurrent]=0

    out = {'v_mp': v_mp,
            'i_mp': i_mp,
           }

    # Compute open circuit voltage
    if calculate_voc:
        v_oc = _lambertw_v_from_i(resistance_shunt, resistance_series, nNsVth, 0.,
                              saturation_current, photocurrent)

        # v_oc[low_photocurrent]=0
        out['v_oc'] = v_oc


    return out

