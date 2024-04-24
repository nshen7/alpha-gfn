import pandas as pd
from numba import jit

@jit(nopython=True)
def _ops_roll_std_arr(ar, window_size):
    '''
    Compute the rolling standard deviation per column of a NumPy array.
    
    Parameters:
        ar (np.ndarray): 2D Input array.
        window_size (int): Size of the rolling window.
        
    Returns:
        np.ndarray: 2D array of rolling standard deviations.
    '''
    n, d = ar.shape
    std_devs = []
    for i in range(n):
        win = ar[max(i+1-window_size, 0):(i+1)]
        win_std_devs = []
        for j in range(d):
            win_std_dev = np.std(win[:, j])
            win_std_devs.append(win_std_dev)
        std_devs.append(win_std_devs)
    return std_devs

def ops_roll_std(df: pd.DataFrame, window_size=5) -> pd.DataFrame:
    '''
    Compute the rolling standard deviation per column of a pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        window_size (int): Size of the rolling window.
        
    Returns:
        pd.DataFrame: DataFrame of rolling standard deviations per column.
    '''
    ar = df.values
    ar_std_devs = _ops_roll_std_arr(ar, window_size)
    df_std_devs = pd.DataFrame(ar_std_devs, index=df.index, columns=df.columns)

    return df_std_devs


@jit(nopython=True)
def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Compute the Spearman correlation coefficient between two NumPy arrays.
    
    Parameters:
        x (np.ndarray): Input array.
        y (np.ndarray): Input array.
        
    Returns:
        float: Spearman correlation coefficient.
    '''
    if len(x) < 2 or len(y) < 2:
        return np.nan
    
    rank_x = np.argsort(x).argsort()
    rank_y = np.argsort(y).argsort()
    
    covariance_matrix = np.cov(rank_x, rank_y)
    covariance_xy = covariance_matrix[0, 1] * (len(x)-1) / len(x) 

    std_dev_x = np.std(rank_x)
    std_dev_y = np.std(rank_y)
    
    spearman_corr = covariance_xy / (std_dev_x * std_dev_y)

    return spearman_corr

@jit(nopython=True)
def _ops_roll_corr_arr(ar_1, ar_2, window_size):
    '''
    Compute the rolling Spearman correlation coefficient between two NumPy arrays.
    
    Parameters:
        x (np.ndarray): 2D Input array.
        y (np.ndarray): 2D Input array.
        
    Returns:
        float: 2D array of Spearman correlation coefficient.
    '''
    n, d = ar_1.shape
    corrs = []
    for i in range(n):
        win_1 = ar_1[max(i+1-window_size, 0):(i+1)]
        win_2 = ar_2[max(i+1-window_size, 0):(i+1)]
        win_corrs = []
        for j in range(d):
            win_corr = _spearman_correlation(win_1[:,j], win_2[:,j])
            win_corrs.append(win_corr)
        corrs.append(win_corrs)
    return corrs

def ops_roll_corr(df_1: pd.DataFrame, df_2: pd.DataFrame, window_size=5) -> pd.DataFrame:
    '''
    Compute the rolling Spearman correlation coefficient between two pandas DataFrames. 
    Returns:
        float: pandas DataFrame of rolling Spearman correlation coefficients per column.
    '''
    if not all(df_1.index == df_2.index):
        ValueError
    if not all(df_1.columns == df_2.columns):
        ValueError    

    ar_1, ar_2 = df_1.values, df_2.values
    ar_corrs = _ops_roll_corr_arr(ar_1, ar_2, window_size)
    df_corrs = pd.DataFrame(ar_corrs, index=df_1.index, columns=df_2.columns)

    return df_corrs