# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid
"""Functions used to compute the loss."""

def compute_mse(y, tx, w):
    """Calculate the cost using mse."""
   
    # number of samples
    N = len(y)
    
    # compute the error
    e = y-tx.dot(w)
    
    # when loss is the MSE
    mse = (1/(2*N))*e.dot(e)
    
    return mse

def compute_mae(y, tx, w):
    """Calculate the cost using mae."""
   
    # number of samples
    N = len(y)
    
    # compute the error
    e = y-tx.dot(w)
    
    # when loss is the MAE
    mae = abs(e).mean()
    
    return mae

def compute_likelihood_log_reg(y, tx, w):
    """Compute the cost by negative log likelihood. """

    logLikelihood = 1
    for i in range(1,y.shape[0]):
        logLikelihood = logLikelihood + y[i]*np.log10(sigmoid(tx[i,:].T.dot(w))) + (1-y[i])*np.log10(1-sigmoid(tx[i,:].T.dot(w)))                     

    return -logLikelihood