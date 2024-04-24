from config import *
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Tuple
# import warnings
# warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

def _mean_rank_ic(x: pd.DataFrame, y: pd.DataFrame) -> float:
    '''
    Compute the rank information correlation (IC) between the paired rows of two data frames and take average on the row ICs. 
    '''

    if all(x.index != y.index):
        raise ValueError("DataFrames must have the same columns.")

    ic_per_row = np.zeros(len(x))
    for i in range(len(x)):
        ic_per_row[i] = stats.spearmanr(x.iloc[i], y.iloc[i], nan_policy='omit')[0]

    return ic_per_row.mean()


def compute_log_reward(factor: pd.DataFrame) -> Tuple[float, float]:
    '''
    Compute the negative logarithm of reward.
    Reward is defined as squared IC (i.e., IC^2)
    '''

    # Proportion of nan's
    nan_proportion = factor.apply(lambda col: np.isnan(col).mean()).values.mean()

    if nan_proportion > 0.5:
        return 0, -100
    try:
        ic = _mean_rank_ic(factor, FOWARD_RETURN)
        return ic, (2 * np.log(np.abs(ic))).clip(-100) 
    except ValueError:
        return 0, -100