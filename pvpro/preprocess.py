from sklearn.linear_model import HuberRegressor
import numpy as np
from tqdm import tqdm
import pandas as pd

def find_irradiance_current_inbounds(poa, current, sample_weight=None,
                                     epsilon=2.5):


    X = np.atleast_2d(poa).transpose()
    y = np.array(current)

    huber = HuberRegressor(epsilon=epsilon).fit(X, y,sample_weight=sample_weight)

    outliers = huber.outliers_
    inbounds = np.logical_not(outliers)

    return inbounds, huber


def classify_irradiance_current_inbounds(poa, current,
                                         boolean_mask=None,
                                         points_per_iteration=2000,
                                         epsilon=2.5,
                                         ):

    inbounds = np.zeros_like(poa).astype('bool')

#     poa = poa[boolean_mask]
#     current = current[boolean_mask]

    lower_iter_idx = np.arange(0,len(poa),points_per_iteration).astype('int')
    upper_iter_idx = lower_iter_idx + points_per_iteration
    if upper_iter_idx[-1] != len(poa):
        upper_iter_idx[-1] = len(poa)
        upper_iter_idx[-2] = len(poa) - points_per_iteration

    num_iterations = len(lower_iter_idx)

    huber = []
    for k in range(num_iterations):
        cax = np.arange(lower_iter_idx[k], upper_iter_idx[k]).astype('int')

        # Filter
        inbounds_iter, huber_iter = find_irradiance_current_inbounds(
            poa=poa[cax],
            current=current[cax],
            sample_weight=boolean_mask[cax],
            epsilon=epsilon
        )

        inbounds[cax] = inbounds_iter
        huber.append(huber_iter)

        # outliers_iter = np.logical_not(inbounds_iter)
#         if k==0:
#             outliers = np.logical_not(inbounds_iter)
#             x_smooth = np.linspace(0,1200,100)

#             inbounds_and_unmasked = np.logical_and(inbounds_iter, boolean_mask[cax])
#             outliers_and_unmasked = np.logical_and(outliers, boolean_mask[cax])

#             plt.scatter(poa[cax][inbounds_and_unmasked],current[cax][inbounds_and_unmasked],s=1,c='b',label='Points to use')
#             plt.plot(x_smooth, huber_iter.coef_ * x_smooth + huber_iter.intercept_,'g')

#             plt.scatter(poa[cax][outliers_and_unmasked], current[cax][outliers_and_unmasked], s=1,
#                        c='r',label='Outliers')
#             plt.xlabel('POA (W/m^2)')
#             plt.ylabel('Current (A)')
#             plt.legend()


    out = {
        'inbounds': inbounds,
        'lower_iter_idx': lower_iter_idx,
        'upper_iter_idx': upper_iter_idx,
        'huber': huber,
    }
    return out

