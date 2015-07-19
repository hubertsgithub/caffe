import scipy.stats.mstats as mstats
import numpy as np


def nanrankdata(arr):
    '''
    Ranks data ignoring NaN values
    '''
    if np.all(np.isnan(arr)):
        return arr.copy()

    ranks = mstats.rankdata(np.ma.masked_invalid(arr))
    ranks[ranks == 0] = np.nan

    return ranks

