import numpy as np
def z_score_normalize(X):
    """
    standardize features by removing the mean and scaling to unit variance
    X: shape(m, n)
    """

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma