import numpy as np
from costs import *

def least_squares(y, tx):
    """calculate the least squares."""
    w = np.linalg.solve(((tx.T).dot(tx)), (tx.T).dot(y))
    
    loss = compute_mse(y, tx, w);
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""    
    w_rr = np.linalg.solve(tx.T.dot(tx) + lambda_*np.identity(tx.shape[1]) , (tx.T).dot(y))
    
    loss = compute_mse(y, tx, w_rr);
    
    return w_rr, loss