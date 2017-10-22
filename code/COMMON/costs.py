# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid
"""Functions used to compute the loss."""

def compute_mse_reg(y, tx, w, lambda_=0):
    """Calculate the cost using mse, penalized when lambda_>0."""
   
    # number of samples
    N = len(y)
    
    # compute the error
    e = y-tx.dot(w)
    
    # when loss is the MSE
    mse = (1/(2*N))*e.dot(e)
    
    return mse

def compute_mae_reg(y, tx, w, lambda_=0):
    """Calculate the cost using mae, penalized when lambda_>0."""
   
    # number of samples
    N = len(y)
    
    # compute the error
    e = y-tx.dot(w)
    
    # when loss is the MAE
    mae = abs(e).mean()
    
    return mae

def compute_logLikelihood_reg(y, tx, w, lambda_=0):
    """Calculate the cost by negative log likelihood, penalized when lambda_>0. """

    # log-likelihood
    logLikelihood = 1
    for i in range(1,y.shape[0]):
        logLikelihood = logLikelihood + y[i]*np.log10(sigmoid(tx[i,:].T.dot(w))) + (1-y[i])*np.log10(1-sigmoid(tx[i,:].T.dot(w)))                     

    # regularized when lambda_>0
    reg_logLikelihood = logLikelihood + lambda_*w.T.dot(w)
    
    return -logLikelihood