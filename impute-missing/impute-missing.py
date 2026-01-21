import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    Works for 1D and 2D arrays.
    """
    X = np.array(X,dtype=float)

    is_1D = X.ndim == 1
    
    if is_1D:
        X = X.reshape(-1, 1)

    X_cpy = X.copy()
    
    for j in range(X_cpy.shape[1]):
        col = X_cpy[:,j]

        mask = np.isnan(col)

        if np.any(np.logical_not(mask)):
            
            if strategy == 'mean':
                statistic = np.nanmean(col)

            elif strategy == 'median':
                statistic = np.nanmedian(col)

            X_cpy[mask,j] = statistic
    
        else:
            X_cpy[mask,j] = 0


    if is_1D:
        X_cpy = X_cpy.ravel()

    return X_cpy
